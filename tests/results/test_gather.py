"""Tests for gathering ABFE results and checking their quality.

The fixtures are synthetic result files, written with the same gufe JSON codec
``openfe quickrun`` uses and following the structure OpenFE 1.12's analysis units produce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import KCAL, abfe_campaign, leg_outputs, write_result
from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.results.gather import gather_results, read_repeat, write_tsv
from openfe_api.results.qc import MIN_NEIGHBOR_OVERLAP, check_ligand
from openfe_api.results.runner import gather_campaign, result_paths

DATA = Path(__file__).parents[1] / "data"


def test_read_repeat_extracts_estimates(tmp_path: Path) -> None:
    path = write_result(tmp_path / "repeat1" / "results.json", estimate=-8.5)

    repeat = read_repeat(path)

    assert repeat.ok
    assert repeat.ligand == "mini"
    assert repeat.estimate == pytest.approx(-8.5)
    assert set(repeat.legs) == {"complex", "solvent"}
    assert repeat.legs["complex"].estimate == pytest.approx(-12.0)
    assert repeat.legs["complex"].standard_state_correction == pytest.approx(-1.2)


def test_read_repeat_ignores_setup_and_simulation_units(tmp_path: Path) -> None:
    path = write_result(tmp_path / "repeat1" / "results.json")

    repeat = read_repeat(path)

    assert "unknown" not in repeat.legs


def test_read_repeat_flags_a_failed_simulation(tmp_path: Path) -> None:
    path = write_result(tmp_path / "repeat1" / "results.json", failed=True)

    repeat = read_repeat(path)

    assert not repeat.ok
    assert repeat.failure is not None
    assert "exception" in repeat.failure


def test_read_repeat_flags_a_missing_estimate(tmp_path: Path) -> None:
    path = write_result(tmp_path / "repeat1" / "results.json", no_estimate=True)

    repeat = read_repeat(path)

    assert not repeat.ok
    assert repeat.failure is not None
    assert "did not finish" in repeat.failure


def test_read_repeat_rejects_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(OpenFEAPIError, match="could not be read"):
        read_repeat(path)


def test_gather_uses_the_spread_between_repeats(tmp_path: Path) -> None:
    paths = [
        write_result(tmp_path / "repeat1" / "results.json", estimate=-8.0),
        write_result(tmp_path / "repeat2" / "results.json", estimate=-9.0),
        write_result(tmp_path / "repeat3" / "results.json", estimate=-10.0),
    ]

    results, failed = gather_results(paths)

    assert failed == []
    assert len(results) == 1
    assert results[0].dg == pytest.approx(-9.0)
    assert results[0].uncertainty == pytest.approx(1.0)
    assert results[0].uncertainty_kind == "std"


def test_gather_falls_back_to_mbar_error_for_one_repeat(tmp_path: Path) -> None:
    paths = [write_result(tmp_path / "repeat1" / "results.json", estimate=-8.0)]

    results, _ = gather_results(paths)

    assert results[0].uncertainty_kind == "mbar"
    assert results[0].uncertainty == pytest.approx(np.sqrt(0.2**2 + 0.2**2))


def test_gather_separates_ligands(tmp_path: Path) -> None:
    paths = [
        write_result(tmp_path / "a" / "results.json", ligand="lig_a", estimate=-8.0),
        write_result(tmp_path / "b" / "results.json", ligand="lig_b", estimate=-6.0),
    ]

    results, _ = gather_results(paths)

    assert [result.name for result in results] == ["lig_a", "lig_b"]


def test_gather_reports_failed_repeats(tmp_path: Path) -> None:
    paths = [
        write_result(tmp_path / "repeat1" / "results.json", estimate=-8.0),
        write_result(tmp_path / "repeat2" / "results.json", failed=True),
    ]

    results, failed = gather_results(paths)

    assert len(results) == 1
    assert len(failed) == 1


def test_write_tsv_has_a_header_and_a_row(tmp_path: Path) -> None:
    paths = [write_result(tmp_path / "repeat1" / "results.json", estimate=-8.0)]
    results, _ = gather_results(paths)

    write_tsv(results, tmp_path / "results.tsv")

    lines = (tmp_path / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t")[0] == "ligand"
    assert lines[1].startswith("mini\t-8.000\t")


def test_quality_passes_for_a_healthy_result(tmp_path: Path) -> None:
    paths = [
        write_result(tmp_path / f"repeat{index}" / "results.json", estimate=-8.0 - index * 0.1)
        for index in (1, 2, 3)
    ]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    assert report.verdict == "pass"
    assert report.failures == []


def test_quality_fails_on_poor_overlap(tmp_path: Path) -> None:
    legs = {
        "complex": leg_outputs("complex", -12.0, neighbor_overlap=0.001),
        "solvent": leg_outputs("solvent", -4.0),
    }
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    overlap = next(check for check in report.checks if check.name == "mbar_overlap")
    assert overlap.failed
    assert str(MIN_NEIGHBOR_OVERLAP) in overlap.detail


def test_quality_fails_when_states_never_exchange(tmp_path: Path) -> None:
    legs = {
        "complex": leg_outputs("complex", -12.0, exchange=0.0),
        "solvent": leg_outputs("solvent", -4.0),
    }
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    exchange = next(check for check in report.checks if check.name == "replica_exchange")
    assert exchange.failed


def test_quality_fails_on_unconverged_forward_reverse(tmp_path: Path) -> None:
    legs = {
        "complex": leg_outputs("complex", -12.0, converged=False),
        "solvent": leg_outputs("solvent", -4.0),
    }
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    convergence = next(check for check in report.checks if check.name == "forward_reverse")
    assert convergence.failed


def test_quality_flags_a_wide_repeat_spread(tmp_path: Path) -> None:
    paths = [
        write_result(tmp_path / "repeat1" / "results.json", estimate=-8.0),
        write_result(tmp_path / "repeat2" / "results.json", estimate=-14.0),
    ]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    spread = next(check for check in report.checks if check.name == "repeat_spread")
    assert spread.failed
    assert report.verdict == "fail"


def test_quality_reports_unknown_when_data_is_absent(tmp_path: Path) -> None:
    legs = {"complex": {"simtype": "complex", "unit_estimate": -12.0 * KCAL}}
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    assert report.verdict == "unknown"
    assert all(not check.failed for check in report.checks)


def _campaign(tmp_path: Path, repeats: int = 2) -> Campaign:
    """Create a campaign whose single run has been submitted.

    Args:
        tmp_path: Pytest temporary directory.
        repeats: Repeats per run.

    Returns:
        The campaign.
    """
    campaign = abfe_campaign(tmp_path, repeats=repeats)
    campaign.set_state("mini", RunState.SUBMITTED)
    return campaign


def test_result_paths_finds_repeat_files(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    for repeat in (1, 2):
        write_result(campaign.directory / "runs" / "mini" / f"repeat{repeat}" / "results.json")

    found = result_paths(campaign)

    assert len(found["mini"]) == 2


def test_gather_campaign_writes_a_table_and_marks_done(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    for repeat, estimate in ((1, -8.0), (2, -8.4)):
        write_result(
            campaign.directory / "runs" / "mini" / f"repeat{repeat}" / "results.json",
            estimate=estimate,
        )

    outcome = gather_campaign(campaign)

    assert outcome.tsv is not None and outcome.tsv.is_file()
    assert outcome.results[0].dg == pytest.approx(-8.2)
    assert outcome.reports["mini"].verdict == "pass"
    assert campaign.run("mini").state is RunState.DONE


def test_gather_campaign_reports_missing_results(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = gather_campaign(campaign)

    assert outcome.missing == ["mini"]
    assert outcome.results == []


def test_gather_campaign_leaves_partial_runs_alone(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, repeats=3)
    write_result(campaign.directory / "runs" / "mini" / "repeat1" / "results.json")

    outcome = gather_campaign(campaign)

    assert len(outcome.results) == 1
    assert campaign.run("mini").state is RunState.SUBMITTED


def _forward_reverse(forward: list[float], reverse: list[float], error: float = 0.2) -> dict:
    """Build forward and reverse convergence data over ten data fractions.

    Args:
        forward: Forward estimates, one per fraction.
        reverse: Reverse estimates, one per fraction.
        error: MBAR error on every point.

    Returns:
        The analysis dictionary an analysis unit reports.
    """
    count = len(forward)
    return {
        "fractions": np.linspace(1 / count, 1.0, count),
        "forward_DGs": np.array(forward) * KCAL,
        "forward_dDGs": np.full(count, error) * KCAL,
        "reverse_DGs": np.array(reverse) * KCAL,
        "reverse_dDGs": np.full(count, error) * KCAL,
    }


def test_early_disagreement_alone_does_not_fail_convergence(tmp_path: Path) -> None:
    """Regression: judging every data fraction failed runs that had in fact converged.

    The first fractions disagree even for a converged run, and the last point is identical
    by construction, so only the second half of the data is judged.
    """
    legs = {
        "complex": leg_outputs("complex", -12.0),
        "solvent": leg_outputs("solvent", -4.0),
    }
    legs["solvent"]["forward_and_reverse_energies"] = _forward_reverse(
        forward=[-9.0, -6.0, -4.3, -4.1, -4.0, -4.0],
        reverse=[1.0, -2.0, -4.4, -4.0, -4.05, -4.0],
    )
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    convergence = next(check for check in report.checks if check.name == "forward_reverse")
    assert convergence.status == "pass", convergence.detail


def test_late_disagreement_fails_convergence(tmp_path: Path) -> None:
    legs = {
        "complex": leg_outputs("complex", -12.0),
        "solvent": leg_outputs("solvent", -4.0),
    }
    legs["solvent"]["forward_and_reverse_energies"] = _forward_reverse(
        forward=[-9.0, -6.0, -4.3, -4.1, -4.0, -4.0],
        reverse=[1.0, -2.0, -30.0, -25.0, -18.0, -4.0],
    )
    paths = [write_result(tmp_path / "repeat1" / "results.json", legs=legs)]
    results, _ = gather_results(paths)

    report = check_ligand(results[0])

    convergence = next(check for check in report.checks if check.name == "forward_reverse")
    assert convergence.failed
    assert "kcal/mol" in convergence.detail
