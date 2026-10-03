"""Tests for campaign directories and their state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openfe_api.campaign import MANIFEST_NAME, Campaign, RunState
from openfe_api.exceptions import (
    CampaignStateError,
    InputValidationError,
    OpenFEAPIError,
)
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.runner import plan_campaign
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import validate_request


@pytest.fixture
def campaign(tmp_path: Path, combined_request_data: dict[str, Any]) -> Campaign:
    """Create a campaign in a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.
        combined_request_data: Request data for a combined structure file.

    Returns:
        The created campaign.
    """
    request = validate_request(combined_request_data)
    return Campaign.create(tmp_path / "campaign", request)


def test_create_writes_manifest_and_directories(campaign: Campaign) -> None:
    assert (campaign.directory / MANIFEST_NAME).is_file()
    for subdirectory in ("prepared", "plans", "runs", "logs"):
        assert (campaign.directory / subdirectory).is_dir()

    assert campaign.manifest.name == "combined_campaign"
    assert campaign.manifest.protocol == "abfe"
    assert campaign.manifest.provenance["python"]


def test_create_seeds_one_pending_run_per_complex(campaign: Campaign) -> None:
    record = campaign.run("complex_one")

    assert record.state is RunState.PENDING
    assert record.repeats == 2
    assert record.job_id is None


def test_create_refuses_to_overwrite(tmp_path: Path, combined_request_data: dict[str, Any]) -> None:
    request = validate_request(combined_request_data)
    Campaign.create(tmp_path / "campaign", request)

    with pytest.raises(CampaignStateError, match="a campaign already exists"):
        Campaign.create(tmp_path / "campaign", request)


def test_create_overwrites_when_allowed(
    tmp_path: Path, combined_request_data: dict[str, Any]
) -> None:
    request = validate_request(combined_request_data)
    Campaign.create(tmp_path / "campaign", request)

    campaign = Campaign.create(tmp_path / "campaign", request, exist_ok=True)

    assert campaign.manifest.name == "combined_campaign"


def test_open_round_trips_the_manifest(campaign: Campaign) -> None:
    campaign.set_state("complex_one", RunState.PREPARED)

    reopened = Campaign.open(campaign.directory)

    assert reopened.run("complex_one").state is RunState.PREPARED
    request = reopened.manifest.request
    assert isinstance(request, AbfeRequest)
    assert request.complexes[0].ligand.smiles


def test_open_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(CampaignStateError, match="no campaign manifest found"):
        Campaign.open(tmp_path / "absent")


def test_open_rejects_unreadable_manifest(campaign: Campaign) -> None:
    (campaign.directory / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(CampaignStateError, match="unreadable manifest"):
        Campaign.open(campaign.directory)


def test_open_rejects_unsupported_schema_version(campaign: Campaign) -> None:
    manifest_path = campaign.directory / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignStateError, match="schema version 99 is not supported"):
        Campaign.open(campaign.directory)


def test_run_reports_unknown_names(campaign: Campaign) -> None:
    with pytest.raises(CampaignStateError, match="no run named 'absent'"):
        campaign.run("absent")


def test_set_state_follows_the_lifecycle(campaign: Campaign) -> None:
    for state in (RunState.PREPARED, RunState.PLANNED, RunState.SUBMITTED, RunState.RUNNING):
        campaign.set_state("complex_one", state)

    record = campaign.set_state("complex_one", RunState.DONE)

    assert record.state is RunState.DONE


def test_set_state_rejects_illegal_transitions(campaign: Campaign) -> None:
    with pytest.raises(CampaignStateError, match="cannot move from 'pending' to 'running'"):
        campaign.set_state("complex_one", RunState.RUNNING)


def test_done_is_terminal(campaign: Campaign) -> None:
    campaign.set_state("complex_one", RunState.PREPARED)
    campaign.set_state("complex_one", RunState.PLANNED)
    campaign.set_state("complex_one", RunState.SUBMITTED)
    campaign.set_state("complex_one", RunState.DONE)

    with pytest.raises(CampaignStateError, match="cannot move from 'done'"):
        campaign.set_state("complex_one", RunState.RUNNING)


def test_failed_runs_can_be_retried(campaign: Campaign) -> None:
    campaign.set_state("complex_one", RunState.FAILED, message="prep failed")
    record = campaign.set_state("complex_one", RunState.PREPARED)

    assert record.state is RunState.PREPARED
    assert record.message is None


def test_repeating_a_state_is_allowed(campaign: Campaign) -> None:
    campaign.set_state("complex_one", RunState.PREPARED)
    record = campaign.set_state("complex_one", RunState.PREPARED, message="re-prepared")

    assert record.message == "re-prepared"


def test_set_state_records_job_id_and_persists(campaign: Campaign) -> None:
    campaign.set_state("complex_one", RunState.PREPARED)
    campaign.set_state("complex_one", RunState.PLANNED)
    campaign.set_state("complex_one", RunState.SUBMITTED, job_id="12345")

    reopened = Campaign.open(campaign.directory)

    assert reopened.run("complex_one").job_id == "12345"


def test_run_directory_is_created(campaign: Campaign) -> None:
    directory = campaign.run_directory("complex_one")

    assert directory == campaign.directory / "runs" / "complex_one"
    assert directory.is_dir()


def test_run_directory_rejects_unknown_runs(campaign: Campaign) -> None:
    with pytest.raises(CampaignStateError, match="no run named 'absent'"):
        campaign.run_directory("absent")


def test_counts_covers_every_state(campaign: Campaign) -> None:
    counts = campaign.counts()

    assert counts[RunState.PENDING] == 1
    assert set(counts) == set(RunState)


def test_create_accepts_an_rbfe_request(
    tmp_path: Path, series_request_data: dict[str, Any]
) -> None:
    """An RBFE campaign starts with no runs, because its runs are edges found at plan time."""
    request = validate_request(series_request_data)

    rbfe_campaign = Campaign.create(tmp_path / "series", request)

    assert rbfe_campaign.manifest.protocol == "rbfe"
    assert rbfe_campaign.manifest.runs == {}
    assert (rbfe_campaign.directory / MANIFEST_NAME).is_file()


def test_rbfe_manifest_round_trips(tmp_path: Path, series_request_data: dict[str, Any]) -> None:
    request = validate_request(series_request_data)
    Campaign.create(tmp_path / "series", request)

    reopened = Campaign.open(tmp_path / "series")

    reloaded = reopened.manifest.request
    assert isinstance(reloaded, RbfeRequest)
    assert reloaded.reference_ligand().name == "lig_ref"
    assert reloaded.alignment.pose == "mcs"


def test_preparing_an_rbfe_campaign_rejects_only(
    tmp_path: Path, series_request_data: dict[str, Any]
) -> None:
    """An RBFE campaign has no complexes to select, so '--only' must say so."""
    request = validate_request(series_request_data)
    rbfe_campaign = Campaign.create(tmp_path / "series", request)

    with pytest.raises(OpenFEAPIError, match="'--only' selects complexes"):
        prepare_campaign(rbfe_campaign, only=["lig_b"])


def test_planning_an_unprepared_rbfe_campaign_says_so(
    tmp_path: Path, series_request_data: dict[str, Any]
) -> None:
    request = validate_request(series_request_data)
    rbfe_campaign = Campaign.create(tmp_path / "series", request)

    with pytest.raises(InputValidationError, match="not prepared yet"):
        plan_campaign(rbfe_campaign)


def test_planning_an_rbfe_campaign_rejects_only(
    tmp_path: Path, series_request_data: dict[str, Any]
) -> None:
    request = validate_request(series_request_data)
    rbfe_campaign = Campaign.create(tmp_path / "series", request)

    with pytest.raises(OpenFEAPIError, match="'--only' selects complexes"):
        plan_campaign(rbfe_campaign, only=["lig_b"])
