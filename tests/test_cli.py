"""Tests for the command line interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import write_result
from openfe_api.campaign import MANIFEST_NAME, Campaign, RunState
from openfe_api.cli import app

runner = CliRunner()


@pytest.fixture
def request_file(write_request: Callable[..., Path], combined_request_data: dict[str, Any]) -> Path:
    """Write the combined-structure request to a YAML file.

    Args:
        write_request: Helper that writes request data to disk.
        combined_request_data: Request data for a combined structure file.

    Returns:
        Path to the written request file.
    """
    return write_request(combined_request_data)


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip()


def test_validate_accepts_a_good_request(request_file: Path) -> None:
    result = runner.invoke(app, ["validate", str(request_file)])

    assert result.exit_code == 0
    assert "Request is valid." in result.stdout
    assert "complex_one" in result.stdout


def test_validate_warns_about_neutralization(
    write_request: Callable[..., Path], combined_request_data: dict[str, Any]
) -> None:
    combined_request_data["neutralize_ligands"] = True
    path = write_request(combined_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "neutralization is ENABLED" in result.stdout.replace("\n", " ")


def test_validate_accepts_an_rbfe_request(
    write_request: Callable[..., Path], series_request_data: dict[str, Any]
) -> None:
    path = write_request(series_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    output = result.stdout.replace("\n", " ")
    assert result.exit_code == 0
    assert "Request is valid." in result.stdout
    assert "reference" in output
    assert "minimal_redundant" in output


def test_validate_reports_each_ligand_pose_source(
    write_request: Callable[..., Path], series_request_data: dict[str, Any], input_dir: Path
) -> None:
    """A docked ligand and a generated one must be distinguishable before anything runs."""
    series_request_data["ligands"][1]["path"] = str(input_dir / "ligand.sdf")
    series_request_data["ligands"][1]["pose"] = "keep"
    path = write_request(series_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "SMILES only" in result.stdout.replace("\n", " ")


def test_validate_warns_when_charge_correction_is_disabled(
    write_request: Callable[..., Path], series_request_data: dict[str, Any]
) -> None:
    series_request_data["charges"] = {"correct_single": False}
    path = write_request(series_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "Charge correction is DISABLED" in result.stdout.replace("\n", " ")


def test_validate_warns_when_large_charge_changes_are_allowed(
    write_request: Callable[..., Path], series_request_data: dict[str, Any]
) -> None:
    series_request_data["charges"] = {"allow_multi": True}
    path = write_request(series_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "more than one are ALLOWED" in result.stdout.replace("\n", " ")


def test_validate_rejects_a_bad_rbfe_request(
    write_request: Callable[..., Path], series_request_data: dict[str, Any]
) -> None:
    series_request_data["reference"] = "lig_absent"
    path = write_request(series_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 1
    assert "not one of the ligands" in result.stderr.replace("\n", " ")


def test_validate_rejects_a_bad_request(
    write_request: Callable[..., Path], combined_request_data: dict[str, Any]
) -> None:
    del combined_request_data["complexes"][0]["extra_ligand_copies"]
    path = write_request(combined_request_data)

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == 1
    assert "extra_ligand_copies" in result.stderr.replace("\n", " ")


def test_validate_reports_a_missing_request_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "absent.yaml")])

    assert result.exit_code == 1
    assert "request file not found" in result.stderr.replace("\n", " ")


def test_create_builds_a_campaign(request_file: Path, tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"

    result = runner.invoke(app, ["create", str(request_file), str(campaign_dir)])

    assert result.exit_code == 0
    assert (campaign_dir / MANIFEST_NAME).is_file()


def test_create_refuses_an_existing_campaign(request_file: Path, tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(request_file), str(campaign_dir)])

    result = runner.invoke(app, ["create", str(request_file), str(campaign_dir)])

    assert result.exit_code == 1
    assert "already exists" in result.stderr.replace("\n", " ")


def test_create_force_overwrites(request_file: Path, tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(request_file), str(campaign_dir)])

    result = runner.invoke(app, ["create", str(request_file), str(campaign_dir), "--force"])

    assert result.exit_code == 0


def test_status_shows_run_states(request_file: Path, tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(request_file), str(campaign_dir)])
    Campaign.open(campaign_dir).set_state("complex_one", RunState.PREPARED)

    result = runner.invoke(app, ["status", str(campaign_dir)])

    assert result.exit_code == 0
    assert "prepared" in result.stdout


def test_status_reports_a_missing_campaign(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "no campaign manifest found" in result.stderr.replace("\n", " ")


def test_unknown_log_level_is_rejected(request_file: Path) -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "validate", str(request_file)])

    assert result.exit_code != 0
    assert "unknown log level" in result.stderr.replace("\n", " ")


def test_log_file_receives_records(request_file: Path, tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    campaign_dir = tmp_path / "campaign"

    result = runner.invoke(
        app, ["--log-file", str(log_file), "create", str(request_file), str(campaign_dir)]
    )

    assert result.exit_code == 0
    assert "Created campaign" in log_file.read_text(encoding="utf-8")


def _mini_request(input_dir: Path) -> dict[str, Any]:
    """Build request data pointing at the mini complex fixture.

    Args:
        input_dir: Unused; present so the fixture ordering stays explicit.

    Returns:
        Request data ready to be written to YAML.
    """
    del input_dir
    data = Path(__file__).parent / "data"
    return {
        "protocol": "abfe",
        "name": "mini_campaign",
        "complexes": [
            {
                "name": "mini",
                "structure": str(data / "mini_complex.cif"),
                "protein": {"chains": ["A"]},
                "ligand": {"selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"},
                "extra_ligand_copies": "drop",
            }
        ],
        "execution": {"repeats": 1},
    }


def test_prep_prepares_a_campaign(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["prep", str(campaign_dir)])

    assert result.exit_code == 0
    assert "prepared" in result.stdout
    assert (campaign_dir / "prepared" / "mini" / "ligand.sdf").is_file()


def test_prep_reports_warnings(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["prep", str(campaign_dir)])

    assert "dropped an additional copy" in result.stdout.replace("\n", " ")


def test_prep_exits_non_zero_on_failure(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    data = _mini_request(input_dir)
    data["complexes"][0]["ligand"]["smiles"] = "CCO"
    path = write_request(data)
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["prep", str(campaign_dir)])

    assert result.exit_code == 1
    assert "failed" in result.stdout


def test_prep_only_selects_named_complexes(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["prep", str(campaign_dir), "--only", "absent"])

    assert result.exit_code == 1
    assert "no run named 'absent'" in result.stderr.replace("\n", " ")


def test_prep_reports_a_missing_campaign(tmp_path: Path) -> None:
    result = runner.invoke(app, ["prep", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "no campaign manifest found" in result.stderr.replace("\n", " ")


def test_plan_plans_a_prepared_campaign(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    data = _mini_request(input_dir)
    data["settings"] = {
        "preset": "screening",
        "overrides": {"partial_charge_settings.partial_charge_method": "nagl"},
    }
    path = write_request(data)
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])
    runner.invoke(app, ["prep", str(campaign_dir), "--skip-parameter-check"])

    result = runner.invoke(app, ["plan", str(campaign_dir)])

    assert result.exit_code == 0
    assert "planned" in result.stdout
    assert "ns total" in result.stdout.replace("\n", " ")
    assert (campaign_dir / "plans" / "abfe_mini.json").is_file()


def test_plan_fails_when_nothing_is_prepared(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["plan", str(campaign_dir)])

    assert result.exit_code == 1
    assert "not prepared yet" in result.stdout.replace("\n", " ")


def test_plan_reports_a_missing_campaign(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "no campaign manifest found" in result.stderr.replace("\n", " ")


def _planned_campaign(campaign_dir: Path) -> None:
    """Move a created campaign into the planned state with a stub plan file.

    Args:
        campaign_dir: Path to the campaign directory.
    """
    campaign = Campaign.open(campaign_dir)
    plan = campaign.directory / "plans" / "abfe_mini.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("{}", encoding="utf-8")
    campaign.set_state("mini", RunState.PREPARED)
    campaign.set_state("mini", RunState.PLANNED)


def test_submit_dry_run_writes_a_script(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    data = _mini_request(input_dir)
    data["execution"] = {"profile": "cluster", "repeats": 2}
    path = write_request(data)
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])
    _planned_campaign(campaign_dir)

    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "cluster:\n  backend: slurm\n  jobs_per_gpu: 2\n  slurm:\n    partition: gpu\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["submit", str(campaign_dir), "--profiles", str(profiles), "--dry-run"]
    )

    assert result.exit_code == 0
    assert (campaign_dir / "submit_standard.sh").is_file()


def test_submit_requires_a_plan(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["submit", str(campaign_dir)])

    assert result.exit_code == 1
    assert "no planned transformations" in result.stderr.replace("\n", " ")


def test_submit_reports_an_unknown_profile(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    data = _mini_request(input_dir)
    data["execution"] = {"profile": "absent", "repeats": 1}
    path = write_request(data)
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])
    _planned_campaign(campaign_dir)

    result = runner.invoke(app, ["submit", str(campaign_dir)])

    assert result.exit_code == 1
    assert "unknown execution profile 'absent'" in result.stderr.replace("\n", " ")


def test_gather_prints_results_and_writes_a_table(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])
    write_result(campaign_dir / "runs" / "mini" / "repeat1" / "results.json", estimate=-8.0)

    result = runner.invoke(app, ["gather", str(campaign_dir)])

    assert result.exit_code == 0
    assert "-8.00" in result.stdout
    assert (campaign_dir / "results" / "results.tsv").is_file()


def test_gather_reports_when_there_is_nothing(
    write_request: Callable[..., Path], input_dir: Path, tmp_path: Path
) -> None:
    path = write_request(_mini_request(input_dir))
    campaign_dir = tmp_path / "campaign"
    runner.invoke(app, ["create", str(path), str(campaign_dir)])

    result = runner.invoke(app, ["gather", str(campaign_dir)])

    assert result.exit_code == 1
    assert "No usable results" in result.stdout
