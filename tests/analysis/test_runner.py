"""Tests for reading a campaign into an analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import DATA, abfe_campaign, network_campaign, write_phase_result, write_result
from openfe_api.analysis.experimental import Measurement
from openfe_api.analysis.runner import analyze_campaign, read_predictions
from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.schema.request import validate_request


def _plan(edges: list[tuple[str, str]]) -> dict[str, Any]:
    """Build the network summary planning would have written."""
    return {
        "edges": [{"name": f"{a}_to_{b}", "ligand_a": a, "ligand_b": b} for a, b in edges],
        "dropped": {},
        "unreachable": [],
    }


def _write_edge(directory: Path, name: str, solvent: float, complex_dg: float) -> None:
    """Write both phases of one finished edge."""
    write_phase_result(directory / "runs" / name / "solvent" / "repeat1" / "results.json", solvent)
    write_phase_result(
        directory / "runs" / name / "complex" / "repeat1" / "results.json", complex_dg
    )


def _finished_abfe(tmp_path: Path) -> Campaign:
    """An ABFE campaign with one finished ligand."""
    campaign = abfe_campaign(tmp_path, repeats=1)
    write_result(campaign.directory / "runs" / "mini" / "repeat1" / "results.json", estimate=-9.4)
    return campaign


def _finished_network(tmp_path: Path) -> Campaign:
    """An RBFE campaign with one finished edge."""
    campaign = network_campaign(tmp_path, "rbfe", _plan([("a", "b")]))
    _write_edge(campaign.directory, "a_to_b", solvent=-2.0, complex_dg=-5.0)
    return campaign


def test_abfe_predictions_are_absolute(tmp_path: Path) -> None:
    predictions, relative = read_predictions(_finished_abfe(tmp_path))

    assert [entry.name for entry in predictions] == ["mini"]
    assert predictions[0].dg == pytest.approx(-9.4)
    assert relative is False


def test_network_predictions_are_relative(tmp_path: Path) -> None:
    predictions, relative = read_predictions(_finished_network(tmp_path))

    assert {entry.name for entry in predictions} == {"a", "b"}
    assert relative is True


def test_analysis_never_moves_the_campaign_on(tmp_path: Path) -> None:
    """Gathering marks finished runs done; analyzing the same campaign must not."""
    campaign = _finished_abfe(tmp_path)
    before = campaign.run("mini").state
    manifest = campaign.directory / "manifest.json"
    stamp = manifest.read_text(encoding="utf-8")

    analyze_campaign(campaign)

    assert campaign.run("mini").state is before
    assert Campaign.open(campaign.directory).run("mini").state is not RunState.DONE
    assert manifest.read_text(encoding="utf-8") == stamp


def test_without_measurements_only_the_ranking_is_written(tmp_path: Path) -> None:
    outcome = analyze_campaign(_finished_abfe(tmp_path))

    assert outcome.figures["ranking"].is_file()
    assert outcome.tables["predictions"].is_file()
    assert "correlation" not in outcome.figures
    assert outcome.correlation is None


def test_measurements_add_the_correlation(tmp_path: Path) -> None:
    campaign = _finished_abfe(tmp_path)

    outcome = analyze_campaign(campaign, {"mini": Measurement("mini", -9.0)})

    assert outcome.correlation is not None
    assert len(outcome.correlation.paired) == 1
    assert outcome.tables["correlation"].is_file()
    assert outcome.tables["metrics"].is_file()
    assert outcome.figures["correlation"].is_file()


def test_the_metrics_file_holds_the_numbers(tmp_path: Path) -> None:
    campaign = _finished_network(tmp_path)
    measured = {"a": Measurement("a", -9.0), "b": Measurement("b", -6.0)}

    outcome = analyze_campaign(campaign, measured)

    payload = json.loads(outcome.tables["metrics"].read_text(encoding="utf-8"))
    assert payload["paired"] == 2
    assert payload["aligned"] is True
    assert payload["mue_kcal_per_mol"] is not None


def test_the_metrics_record_provenance(tmp_path: Path) -> None:
    """Rule 20: every artifact records the versions that produced it."""
    campaign = _finished_abfe(tmp_path)

    outcome = analyze_campaign(campaign, {"mini": Measurement("mini", -9.0)})

    payload = json.loads(outcome.tables["metrics"].read_text(encoding="utf-8"))
    assert payload["provenance"]
    assert "openfe" in payload["provenance"]


def test_everything_lands_in_the_analysis_directory(tmp_path: Path) -> None:
    campaign = _finished_abfe(tmp_path)

    outcome = analyze_campaign(campaign, {"mini": Measurement("mini", -9.0)})

    assert outcome.directory == campaign.directory / "analysis"
    for path in (*outcome.figures.values(), *outcome.tables.values()):
        assert path.parent == outcome.directory


def test_a_measured_ligand_the_campaign_lacks_is_warned_about(tmp_path: Path) -> None:
    campaign = _finished_abfe(tmp_path)

    outcome = analyze_campaign(campaign, {"ghost": Measurement("ghost", -9.0)})

    assert any("not in the campaign" in warning for warning in outcome.warnings)
    assert "correlation" not in outcome.figures


def test_a_measured_table_can_be_given_as_a_file(tmp_path: Path) -> None:
    campaign = _finished_abfe(tmp_path)
    table = tmp_path / "exp.csv"
    table.write_text("name,dg\nmini,-9.0\n", encoding="utf-8")

    outcome = analyze_campaign(campaign, table)

    assert outcome.correlation is not None
    assert len(outcome.correlation.paired) == 1


def test_analysis_can_be_run_twice(tmp_path: Path) -> None:
    """It is read-only, so a second run with different values just overwrites the output."""
    campaign = _finished_abfe(tmp_path)

    analyze_campaign(campaign, {"mini": Measurement("mini", -9.0)})
    outcome = analyze_campaign(campaign, {"mini": Measurement("mini", -8.0)})

    assert outcome.figures["correlation"].is_file()


def test_a_plain_md_campaign_is_refused(tmp_path: Path) -> None:
    """Trajectory metrics are a later milestone; MD has no free energy to correlate."""
    request = validate_request(
        {
            "protocol": "md",
            "name": "md_analysis",
            "systems": [
                {
                    "name": "holo",
                    "structure": str(DATA / "mini_complex.cif"),
                    "protein": {"chains": ["A"]},
                    "ligands": [
                        {"name": "lig", "selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"}
                    ],
                }
            ],
            "execution": {"repeats": 1},
        }
    )

    with pytest.raises(OpenFEAPIError, match="no free energy"):
        read_predictions(Campaign.create(tmp_path / "md", request))


def test_a_campaign_with_nothing_finished_is_refused(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=1)

    with pytest.raises(InputValidationError, match="no finished results"):
        read_predictions(campaign)
