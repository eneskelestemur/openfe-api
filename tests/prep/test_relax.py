"""Tests for the optional structure relaxation stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import write_md_result
from openfe_api.campaign import Campaign
from openfe_api.cli import app
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.execution.runner import relax_campaign
from openfe_api.prep.relax import (
    RELAX_CLASS,
    RelaxReport,
    collect_relaxed,
    relax_targets,
    relax_tasks,
    relaxed_request,
)
from openfe_api.prep.runner import prepare_campaign
from openfe_api.prep.structure import AMINO_ACIDS, Structure
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"
runner = CliRunner()

FAST_CHARGES = {"partial_charge_settings.partial_charge_method": "nagl"}

PROFILE = ExecutionProfile(
    name="relax_gpu",
    backend="slurm",
    jobs_per_gpu=1,
    slurm=SlurmProfile(partition="gpu", gres="gpu:1", time="4:00:00"),
)


def complex_data(relax: bool = True) -> dict[str, Any]:
    """Build an ABFE request on the trimmed complex fixture.

    Args:
        relax: Whether to enable relaxation.

    Returns:
        Request data ready for validation.
    """
    data: dict[str, Any] = {
        "protocol": "abfe",
        "name": "relax_abfe",
        "complexes": [
            {
                "name": "mini",
                "structure": str(DATA / "mini_complex.cif"),
                "protein": {"chains": ["A"]},
                "ligand": {"selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"},
                "extra_ligand_copies": "drop",
            }
        ],
        "settings": {"preset": "screening", "overrides": dict(FAST_CHARGES)},
        "execution": {"repeats": 1},
    }
    if relax:
        data["relax"] = {"enabled": True, "length": "0.05 nanosecond"}
    return data


def campaign_for(data: dict[str, Any], tmp_path: Path) -> Campaign:
    """Create a campaign from request data.

    Args:
        data: Request data.
        tmp_path: Pytest temporary directory.

    Returns:
        The campaign.
    """
    return Campaign.create(tmp_path / "campaign", validate_request(data))


def test_relaxation_is_off_by_default(series_request_data: dict[str, Any]) -> None:
    """Every campaign so far prepared from the input structure as supplied."""
    assert validate_request(series_request_data).relax.enabled is False


def test_the_length_and_seed_are_recorded(series_request_data: dict[str, Any]) -> None:
    series_request_data["relax"] = {"enabled": True, "length": "0.5 nanosecond", "seed": 11}

    relax = validate_request(series_request_data).relax

    assert relax.length == "0.5 nanosecond"
    assert relax.seed == 11


def test_every_abfe_complex_carries_its_own_frame() -> None:
    data = complex_data()
    data["complexes"].append({**data["complexes"][0], "name": "second"})
    request = validate_request(data)
    assert isinstance(request, AbfeRequest)

    targets = relax_targets(request)

    assert [target.name for target in targets] == ["mini", "second"]
    assert [molecule.name for molecule in targets[0].molecules()] == ["mini"]


def test_only_the_reference_carries_a_series_frame(series_request_data: dict[str, Any]) -> None:
    """Every other ligand is placed into the reference's frame, so relaxing it is enough."""
    request = validate_request(series_request_data)
    assert isinstance(request, RbfeRequest)

    targets = relax_targets(request)

    assert [target.name for target in targets] == ["lig_ref"]


def test_every_md_system_carries_its_own_frame(md_request_data: dict[str, Any]) -> None:
    request = validate_request(md_request_data)
    assert isinstance(request, MdRequest)

    assert [target.name for target in relax_targets(request)] == ["complex_one"]


def test_relaxing_a_campaign_that_did_not_ask_for_it_is_refused(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(relax=False), tmp_path)

    with pytest.raises(OpenFEAPIError, match="does not ask for relaxation"):
        relax_campaign(campaign, PROFILE)


def test_preparing_before_the_relaxation_finished_is_refused(tmp_path: Path) -> None:
    """Preparing anyway would silently use the structure the user asked to replace."""
    campaign = campaign_for(complex_data(), tmp_path)

    with pytest.raises(InputValidationError, match="has not finished"):
        prepare_campaign(campaign, check_parameters=False)


@pytest.mark.slow
def test_relaxation_prepares_plans_and_queues(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)

    outcome = relax_campaign(campaign, PROFILE, check_parameters=False, dry_run=True)

    assert outcome.targets == ["mini"]
    relaxed = campaign.directory / "relaxed" / "mini"
    assert (relaxed / "input" / "prep_report.json").is_file()
    assert (relaxed / "md_mini.json").is_file()
    assert [task.cost_class for task in outcome.tasks] == [RELAX_CLASS]
    assert outcome.tasks[0].expects_estimate is False


@pytest.mark.slow
def test_the_relaxation_inherits_the_charge_method(tmp_path: Path) -> None:
    """A different method would key the cache differently and pay for charges twice."""
    campaign = campaign_for(complex_data(), tmp_path)

    relax_campaign(campaign, PROFILE, check_parameters=False, dry_run=True)

    metadata = json.loads(
        (campaign.directory / "prepared" / "charges" / "mini.json").read_text(encoding="utf-8")
    )
    assert metadata["method"] == "nagl"


@pytest.mark.slow
def test_a_finished_relaxation_is_not_run_again(tmp_path: Path) -> None:
    """Running the stage again while waiting for a job must cost nothing."""
    campaign = campaign_for(complex_data(), tmp_path)
    relax_campaign(campaign, PROFILE, check_parameters=False, dry_run=True)
    task = relax_tasks(campaign)[0]
    write_md_result(task.result_path)
    prepared = campaign.directory / "relaxed" / "mini" / "input" / "prep_report.json"
    written_at = prepared.stat().st_mtime_ns

    outcome = relax_campaign(campaign, PROFILE, check_parameters=False, dry_run=True)

    assert outcome.skipped == ["mini"]
    assert outcome.tasks == []
    assert prepared.stat().st_mtime_ns == written_at


def test_collecting_before_the_run_finished_says_so(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)

    with pytest.raises(InputValidationError, match="has not finished"):
        collect_relaxed(campaign, "mini")


def test_collecting_a_run_that_wrote_no_frame_says_so(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)
    result = campaign.directory / "relaxed" / "mini" / "run" / "results.json"
    write_md_result(result, failed=True)

    with pytest.raises(InputValidationError, match="wrote no equilibrated structure"):
        collect_relaxed(campaign, "mini")


def relaxed_frame(campaign: Campaign, with_ligand: bool = True) -> Path:
    """Write a relaxation result whose equilibrated frame is a real structure.

    The frame stands in for what the MD protocol writes: the protein and the molecules that
    were simulated, with the water stripped.

    Args:
        campaign: The campaign being relaxed.
        with_ligand: Whether the frame holds the ligand, as a finished relaxation would.

    Returns:
        The result file written.
    """
    structure = Structure.load(DATA / "mini_complex.cif")
    keep = {
        residue.index
        for residue in structure.residues()
        if residue.name.upper() in AMINO_ACIDS
        or (with_ligand and residue.name == "LIG" and residue.chain_id == "L")
    }

    run_dir = campaign.directory / "relaxed" / "mini" / "run"
    shared = run_dir / "shared_PlainMDSimulationUnit-abc_attempt_0"
    shared.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(keep, shared / "equil_npt.pdb")
    write_md_result(run_dir / "results.json")
    return run_dir / "results.json"


def test_the_collected_frame_maps_every_molecule(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)
    relaxed_frame(campaign)

    report = collect_relaxed(campaign, "mini")

    assert report.structure.is_file()
    assert list(report.selectors) == ["mini"]
    assert report.selectors["mini"].resname is not None


def test_the_collected_report_is_reused(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)
    relaxed_frame(campaign)
    collect_relaxed(campaign, "mini")

    stored = RelaxReport.read(campaign.directory / "relaxed" / "mini" / "relax_report.json")

    assert stored.name == "mini"


def test_a_frame_missing_a_molecule_fails_by_name(tmp_path: Path) -> None:
    """The count check catches a frame that does not hold what the system declared."""
    campaign = campaign_for(complex_data(), tmp_path)
    relaxed_frame(campaign, with_ligand=False)

    with pytest.raises(InputValidationError, match="molecule residue"):
        collect_relaxed(campaign, "mini")


def test_the_request_is_rewritten_to_read_the_relaxed_structure(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(), tmp_path)
    relaxed_frame(campaign)

    request = relaxed_request(campaign)

    assert isinstance(request, AbfeRequest)
    spec = request.complexes[0]
    assert spec.structure == campaign.directory / "relaxed" / "mini" / "system.pdb"
    assert spec.protein.path is None
    assert spec.ligand.path is None
    assert spec.ligand.selector is not None


def test_an_unrelaxed_request_is_returned_unchanged(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(relax=False), tmp_path)

    assert relaxed_request(campaign) is campaign.manifest.request


def test_the_cli_reports_what_it_queued(tmp_path: Path) -> None:
    campaign = campaign_for(complex_data(relax=False), tmp_path)

    result = runner.invoke(app, ["relax", str(campaign.directory)])

    assert result.exit_code == 1
    assert "does not ask for relaxation" in result.output


@pytest.mark.slow
def test_preparation_runs_against_the_relaxed_structure(tmp_path: Path) -> None:
    """The whole point: the relaxed frame re-enters preparation and is verified there."""
    campaign = campaign_for(complex_data(), tmp_path)
    relaxed_frame(campaign)

    outcome = prepare_campaign(campaign, check_parameters=False)

    report = outcome.reports["mini"]
    relaxed = campaign.directory / "relaxed" / "mini" / "system.pdb"
    assert report.ligand.source == relaxed
    assert report.protein.source == relaxed
    assert report.ligand.smiles == "C[C@H](O)c1ccccc1"


@pytest.mark.slow
def test_a_relaxed_frame_holding_the_wrong_molecule_fails_preparation(tmp_path: Path) -> None:
    """Preparation rebuilds from the declared SMILES, so a wrong residue cannot pass."""
    data = complex_data()
    data["complexes"][0]["ligand"]["smiles"] = "CCO"
    campaign = campaign_for(data, tmp_path)
    relaxed_frame(campaign)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert "mini" in outcome.failures
