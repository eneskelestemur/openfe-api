"""Tests for turning finished SepTop edges into edge and ligand free energies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from helpers import network_campaign, write_septop_result
from openfe_api.campaign import RunState
from openfe_api.results.network import gather_network, write_edge_tsv, write_ligand_tsv
from openfe_api.results.qc import check_edge, check_network
from openfe_api.results.runner import gather_campaign
from openfe_api.results.septop import read_septop_edge

DATA = Path(__file__).parents[1] / "data"


def plan(edges: list[tuple[str, str]], hub: str = "a") -> dict[str, Any]:
    """Build the network summary planning would have written.

    Args:
        edges: Ligand name pairs.
        hub: The hub ligand's name.

    Returns:
        The summary mapping.
    """
    return {
        "protocol": "septop",
        "hub": hub,
        "edges": [{"name": f"{a}_to_{b}", "ligand_a": a, "ligand_b": b} for a, b in edges],
        "dropped": {},
        "unreachable": [],
        "warnings": [],
    }


def write_edge(root: Path, name: str, ddg: float, repeats: int = 1, **kwargs: Any) -> None:
    """Write result files for one edge.

    Args:
        root: Campaign directory.
        name: Edge name.
        ddg: The relative binding free energy in kcal/mol.
        repeats: Repeats to write.
        **kwargs: Passed through to the result writer.
    """
    for repeat in range(1, repeats + 1):
        write_septop_result(
            root / "runs" / name / f"repeat{repeat}" / "results.json", ddg=ddg, **kwargs
        )


def gather(root: Path, planned: dict[str, Any]) -> Any:
    """Gather a SepTop network from a campaign directory.

    Args:
        root: Campaign directory.
        planned: The network summary.

    Returns:
        The gathered results.
    """
    return gather_network(root, planned, read_septop_edge)


def test_an_edge_takes_its_value_from_the_file(tmp_path: Path) -> None:
    """OpenFE already forms the difference and applies both standard state corrections."""
    write_edge(tmp_path, "a_to_b", ddg=-1.5)

    results = gather(tmp_path, plan([("a", "b")]))

    assert results.edges[0].ddg == pytest.approx(-1.5)


def test_one_repeat_uses_the_phase_mbar_errors(tmp_path: Path) -> None:
    """The file's own uncertainty is the spread over one repeat, so it is structurally zero."""
    write_edge(tmp_path, "a_to_b", ddg=-1.5, uncertainty=0.3)

    edge = gather(tmp_path, plan([("a", "b")])).edges[0]

    assert edge.uncertainty_kind == "mbar"
    assert edge.uncertainty == pytest.approx(0.3 * 2**0.5)


def test_several_repeats_use_their_spread(tmp_path: Path) -> None:
    root = tmp_path
    for repeat, value in enumerate((-1.0, -2.0), start=1):
        write_septop_result(
            root / "runs" / "a_to_b" / f"repeat{repeat}" / "results.json", ddg=value
        )

    edge = gather(root, plan([("a", "b")])).edges[0]

    assert edge.ddg == pytest.approx(-1.5)
    assert edge.uncertainty_kind == "std"
    assert edge.uncertainty == pytest.approx(0.7071, abs=1e-3)


def test_both_phases_are_read_from_the_one_file(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.5)

    edge = gather(tmp_path, plan([("a", "b")])).edges[0]

    assert sorted(edge.phases) == ["complex", "solvent"]
    assert all(len(entries) == 1 for entries in edge.phases.values())


def test_a_star_recovers_the_per_ligand_differences(tmp_path: Path) -> None:
    """Fitted values are relative to the network mean, so only differences are meaningful."""
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c")]))

    by_name = {entry.name: entry.dg for entry in results.ligands}
    assert by_name["b"] - by_name["a"] == pytest.approx(-1.0, abs=1e-6)
    assert by_name["c"] - by_name["a"] == pytest.approx(-3.0, abs=1e-6)


def test_every_ligand_is_fitted_once(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c")]))

    names = [entry.name for entry in results.ligands]
    assert sorted(names) == ["a", "b", "c"]


def test_the_hub_degree_counts_every_edge(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c")]))

    assert next(entry.degree for entry in results.ligands if entry.name == "a") == 2


def test_a_redundant_network_closes_its_cycle(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)
    write_edge(tmp_path, "b_to_c", ddg=-2.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c"), ("b", "c")]))

    assert results.cycles
    assert abs(results.cycles[0].closure) == pytest.approx(0.0, abs=1e-6)
    assert check_network(results).verdict == "pass"


def test_an_inconsistent_cycle_is_flagged(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)
    write_edge(tmp_path, "b_to_c", ddg=3.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c"), ("b", "c")]))

    assert check_network(results).verdict == "fail"


def test_an_unfinished_edge_is_reported_not_fitted(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)

    results = gather(tmp_path, plan([("a", "b"), ("a", "c")]))

    unfinished = next(edge for edge in results.edges if edge.name == "a_to_c")
    assert not unfinished.ok
    assert "no repeat produced a usable" in (unfinished.failure or "")
    assert [entry.name for entry in results.ligands] != []


def test_a_failed_repeat_does_not_hide_a_good_one(tmp_path: Path) -> None:
    write_septop_result(
        tmp_path / "runs" / "a_to_b" / "repeat1" / "results.json", ddg=0.0, failed=True
    )
    write_septop_result(tmp_path / "runs" / "a_to_b" / "repeat2" / "results.json", ddg=-1.0)

    edge = gather(tmp_path, plan([("a", "b")])).edges[0]

    assert edge.ddg == pytest.approx(-1.0)


def test_a_missing_phase_fails_the_quality_check(tmp_path: Path) -> None:
    write_septop_result(
        tmp_path / "runs" / "a_to_b" / "repeat1" / "results.json", ddg=0.0, failed=True
    )

    edge = gather(tmp_path, plan([("a", "b")])).edges[0]
    report = check_edge(edge)

    assert report.verdict == "fail"
    assert any(check.name == "both_phases" and check.failed for check in report.checks)


def test_poor_overlap_is_found_in_either_phase(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0, neighbor_overlap=0.01)

    report = check_edge(gather(tmp_path, plan([("a", "b")])).edges[0])

    assert any(check.name == "mbar_overlap" and check.failed for check in report.checks)


def test_good_overlap_passes(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)

    report = check_edge(gather(tmp_path, plan([("a", "b")])).edges[0])

    assert not any(check.name == "mbar_overlap" and check.failed for check in report.checks)


def test_the_tables_are_written(tmp_path: Path) -> None:
    write_edge(tmp_path, "a_to_b", ddg=-1.0)
    write_edge(tmp_path, "a_to_c", ddg=-3.0)
    results = gather(tmp_path, plan([("a", "b"), ("a", "c")]))

    edges_path = tmp_path / "results" / "edges.tsv"
    ligands_path = tmp_path / "results" / "ligands.tsv"
    write_edge_tsv(results.edges, edges_path)
    write_ligand_tsv(results.ligands, ligands_path)

    with edges_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["edge"] for row in rows] == ["a_to_b", "a_to_c"]
    assert ligands_path.read_text(encoding="utf-8").count("\n") == 4


def test_gathering_marks_a_finished_edge_done(tmp_path: Path) -> None:
    campaign = network_campaign(tmp_path, "septop", plan([("a", "b")]))
    write_edge(campaign.directory, "a_to_b", ddg=-1.0)

    outcome = gather_campaign(campaign, write=False)

    assert outcome.network is not None
    assert campaign.run("a_to_b").state is RunState.DONE
