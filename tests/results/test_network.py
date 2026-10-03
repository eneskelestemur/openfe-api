"""Tests for turning finished RBFE phases into edge and ligand free energies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from helpers import network_campaign, write_phase_result
from openfe_api.campaign import RunState
from openfe_api.results.network import (
    gather_network,
    read_phase,
    write_edge_tsv,
    write_ligand_tsv,
)
from openfe_api.results.qc import check_edge, check_network
from openfe_api.results.runner import gather_campaign


def plan(edges: list[tuple[str, str]]) -> dict[str, Any]:
    """Build the network summary planning would have written.

    Args:
        edges: Ligand name pairs.

    Returns:
        The summary mapping.
    """
    return {
        "edges": [{"name": f"{a}_to_{b}", "ligand_a": a, "ligand_b": b} for a, b in edges],
        "dropped": {},
        "unreachable": [],
        "warnings": [],
    }


def write_edge(
    root: Path,
    name: str,
    solvent: float,
    complex_dg: float,
    repeats: int = 1,
    skip: str | None = None,
) -> None:
    """Write result files for both phases of one edge.

    Args:
        root: Campaign directory.
        name: Edge name.
        solvent: Solvent phase estimate in kcal/mol.
        complex_dg: Complex phase estimate in kcal/mol.
        repeats: Repeats to write per phase.
        skip: A phase to leave unwritten, simulating a run that has not finished.
    """
    for phase, estimate in (("solvent", solvent), ("complex", complex_dg)):
        if phase == skip:
            continue
        for repeat in range(1, repeats + 1):
            write_phase_result(
                root / "runs" / name / phase / f"repeat{repeat}" / "results.json",
                estimate + 0.1 * (repeat - 1),
            )


def test_an_edge_is_the_difference_of_its_phases(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0)

    results = gather_network(tmp_path, plan([("a", "b")]))

    assert len(results.usable_edges) == 1
    assert results.usable_edges[0].ddg == pytest.approx(-3.0)


def test_repeats_are_averaged_with_their_spread(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0, repeats=3)

    results = gather_network(tmp_path, plan([("a", "b")]))

    edge = results.usable_edges[0]
    assert edge.uncertainty_kind == "std"
    assert edge.ddg == pytest.approx(-3.0)


def test_a_single_repeat_reports_the_mbar_error(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0)

    edge = gather_network(tmp_path, plan([("a", "b")])).usable_edges[0]

    assert edge.uncertainty_kind == "mbar"
    assert edge.uncertainty == pytest.approx(0.283, abs=0.01)


def test_a_missing_phase_makes_the_edge_unusable(tmp_path: Path) -> None:
    """Both phases are needed: their difference is the relative binding free energy."""
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0, skip="complex")

    results = gather_network(tmp_path, plan([("a", "b")]))

    assert results.usable_edges == []
    assert results.edges[0].failure is not None
    assert "complex phase" in results.edges[0].failure
    assert any("complex phase" in warning for warning in results.warnings)


def test_a_failed_phase_is_reported_not_averaged_in(tmp_path: Path) -> None:
    write_phase_result(tmp_path / "runs" / "a_to_b" / "solvent" / "repeat1" / "results.json", -2.0)
    write_phase_result(
        tmp_path / "runs" / "a_to_b" / "complex" / "repeat1" / "results.json",
        0.0,
        failed=True,
    )

    results = gather_network(tmp_path, plan([("a", "b")]))

    assert results.usable_edges == []
    assert results.edges[0].phases["complex"][0].ok is False


def test_ligand_values_are_fitted_over_the_network(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)

    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c")]))

    values = {entry.name: entry.dg for entry in results.ligands}
    assert set(values) == {"a", "b", "c"}
    assert values["b"] - values["a"] == pytest.approx(1.0, abs=1e-6)
    assert values["c"] - values["b"] == pytest.approx(2.0, abs=1e-6)


def test_ligand_degree_counts_its_edges(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)

    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c")]))

    degrees = {entry.name: entry.degree for entry in results.ligands}
    assert degrees == {"a": 1, "b": 2, "c": 1}


def test_a_consistent_cycle_closes(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)
    write_edge(tmp_path, "a_to_c", solvent=0.0, complex_dg=3.0)

    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c"), ("a", "c")]))

    assert len(results.cycles) == 1
    assert results.cycles[0].closure == pytest.approx(0.0, abs=1e-6)


def test_an_inconsistent_cycle_is_measured(tmp_path: Path) -> None:
    """Hysteresis is the only internal quality check RBFE has without experimental data."""
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)
    write_edge(tmp_path, "a_to_c", solvent=0.0, complex_dg=3.5)

    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c"), ("a", "c")]))

    assert len(results.cycles) == 1
    assert abs(results.cycles[0].closure) == pytest.approx(0.5, abs=1e-6)
    assert set(results.cycles[0].ligands) == {"a", "b", "c"}


def test_a_disconnected_network_names_the_orphans(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "c_to_d", solvent=0.0, complex_dg=2.0, skip="complex")

    results = gather_network(tmp_path, plan([("a", "b"), ("c", "d")]))

    assert set(results.unreachable) == {"c", "d"}
    assert {entry.name for entry in results.ligands} == {"a", "b"}
    assert any("not connected" in warning for warning in results.warnings)


def test_nothing_finished_is_reported_without_crashing(tmp_path: Path) -> None:
    results = gather_network(tmp_path, plan([("a", "b")]))

    assert results.usable_edges == []
    assert results.ligands == []
    assert any("nothing to fit yet" in warning for warning in results.warnings)


def test_a_phase_result_carries_its_quality_data(tmp_path: Path) -> None:
    path = write_phase_result(tmp_path / "results.json", -3.0)

    outcome = read_phase(path, "complex", 1)

    assert outcome.overlap_matrix is not None
    assert outcome.exchange_matrix is not None
    assert outcome.forward_reverse is not None
    assert outcome.production_iterations == 4000


def test_a_result_without_an_estimate_is_a_failure(tmp_path: Path) -> None:
    path = write_phase_result(tmp_path / "results.json", -3.0, no_estimate=True)

    outcome = read_phase(path, "complex", 1)

    assert outcome.ok is False
    assert outcome.failure is not None
    assert "did not finish" in outcome.failure


def test_the_edge_tsv_records_failures_too(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0)
    write_edge(tmp_path, "c_to_d", solvent=-2.0, complex_dg=-5.0, skip="solvent")
    results = gather_network(tmp_path, plan([("a", "b"), ("c", "d")]))

    path = tmp_path / "edges.tsv"
    write_edge_tsv(results.edges, path)

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert [row["edge"] for row in rows] == ["a_to_b", "c_to_d"]
    assert rows[0]["DDG (kcal/mol)"] == "-3.000"
    assert rows[1]["DDG (kcal/mol)"] == ""
    assert "solvent phase" in rows[1]["note"]


def test_the_ligand_tsv_names_the_values_as_relative(tmp_path: Path) -> None:
    """A single fitted value means nothing on its own, so the header must say so."""
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    results = gather_network(tmp_path, plan([("a", "b")]))

    path = tmp_path / "ligands.tsv"
    write_ligand_tsv(results.ligands, path)

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "relative to network mean" in header


def test_each_ligand_is_fitted_once_over_the_whole_network(tmp_path: Path) -> None:
    """Fitting per edge instead of per network returns plausible but wrong values."""
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)

    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c")]))

    names = [entry.name for entry in results.ligands]
    assert len(names) == len(set(names)) == 3


def test_edge_checks_flag_a_missing_phase(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0, skip="complex")
    results = gather_network(tmp_path, plan([("a", "b")]))

    report = check_edge(results.edges[0])

    assert report.verdict == "fail"
    assert any(check.name == "both_phases" and check.failed for check in report.checks)


def test_edge_checks_flag_poor_overlap(tmp_path: Path) -> None:
    for phase in ("solvent", "complex"):
        write_phase_result(
            tmp_path / "runs" / "a_to_b" / phase / "repeat1" / "results.json",
            -2.0,
            neighbor_overlap=0.001,
        )
    results = gather_network(tmp_path, plan([("a", "b")]))

    report = check_edge(results.edges[0])

    assert any(check.name == "mbar_overlap" and check.failed for check in report.checks)


def test_edge_checks_pass_on_a_healthy_edge(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=-2.0, complex_dg=-5.0, repeats=2)
    results = gather_network(tmp_path, plan([("a", "b")]))

    report = check_edge(results.edges[0])

    assert report.verdict == "pass"


def test_the_network_check_flags_a_cycle_that_does_not_close(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "b_to_c", solvent=0.0, complex_dg=2.0)
    write_edge(tmp_path, "a_to_c", solvent=0.0, complex_dg=6.0)
    results = gather_network(tmp_path, plan([("a", "b"), ("b", "c"), ("a", "c")]))

    report = check_network(results)

    assert any(check.name == "cycle_closure" and check.failed for check in report.checks)


def test_the_network_check_says_when_there_is_no_cycle(tmp_path: Path) -> None:
    """A spanning network has nothing to close, which is a gap rather than a pass."""
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    results = gather_network(tmp_path, plan([("a", "b")]))

    report = check_network(results)

    closure = next(check for check in report.checks if check.name == "cycle_closure")
    assert closure.status == "unknown"
    assert "redundant network" in closure.detail


def test_the_network_check_flags_disconnection(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", solvent=0.0, complex_dg=1.0)
    write_edge(tmp_path, "c_to_d", solvent=0.0, complex_dg=2.0)
    results = gather_network(tmp_path, plan([("a", "b"), ("c", "d")]))

    report = check_network(results)

    assert any(check.name == "connectivity" and check.failed for check in report.checks)


def test_gather_campaign_dispatches_to_the_network(tmp_path: Path) -> None:
    campaign = network_campaign(tmp_path, "rbfe", plan([("a", "b")]))
    write_edge(campaign.directory, "a_to_b", solvent=-2.0, complex_dg=-5.0)

    outcome = gather_campaign(campaign)

    assert outcome.network is not None
    assert outcome.results == []
    assert outcome.network.usable_edges[0].ddg == pytest.approx(-3.0)
    assert "network" in outcome.reports
    assert outcome.tables["edges"].is_file()
    assert outcome.tables["ligands"].is_file()
    assert campaign.run("a_to_b").state is RunState.DONE


def test_the_mbar_error_is_used_not_the_across_repeat_spread(tmp_path: Path) -> None:
    """OpenFE reports the top-level uncertainty as the spread across its own repeats.

    openfe-api always runs one repeat per process, so that spread is over a single value and
    comes back as zero. Using it would report every single-repeat edge as having no
    uncertainty, and the network fit refuses a zero-uncertainty edge outright.
    """
    path = write_phase_result(
        tmp_path / "results.json", -3.0, uncertainty=1.4, reported_uncertainty=0.0
    )

    outcome = read_phase(path, "complex", 1)

    assert outcome.uncertainty == pytest.approx(1.4)


def test_an_edge_of_single_repeats_has_a_usable_uncertainty(tmp_path: Path) -> None:
    for phase, estimate in (("solvent", -2.0), ("complex", -5.0)):
        write_phase_result(
            tmp_path / "runs" / "a_to_b" / phase / "repeat1" / "results.json",
            estimate,
            uncertainty=1.4,
            reported_uncertainty=0.0,
        )

    results = gather_network(tmp_path, plan([("a", "b")]))

    edge = results.usable_edges[0]
    assert edge.uncertainty is not None and edge.uncertainty > 0.0
    assert results.ligands, "a zero uncertainty would have made the fit refuse this edge"


def test_an_unsolvable_fit_keeps_the_edge_values(tmp_path: Path) -> None:
    """A fit that cannot be solved must not discard the per-edge values already computed."""
    for phase, estimate in (("solvent", -2.0), ("complex", -5.0)):
        write_phase_result(
            tmp_path / "runs" / "a_to_b" / phase / "repeat1" / "results.json",
            estimate,
            uncertainty=0.0,
            reported_uncertainty=0.0,
        )

    results = gather_network(tmp_path, plan([("a", "b")]))

    assert results.usable_edges[0].ddg == pytest.approx(-3.0)
    assert results.ligands == []
    assert any("per-ligand fit could not be solved" in w for w in results.warnings)
