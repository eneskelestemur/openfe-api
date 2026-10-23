"""Tests for handing the analysis to Slurm."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import LOCAL_PROFILE, SLURM_PROFILE, abfe_campaign
from openfe_api.analysis.submit import analysis_command, submit_analysis
from openfe_api.exceptions import OpenFEAPIError


def test_the_submitted_command_carries_the_options_it_was_given(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    measured = tmp_path / "affinities.csv"

    command = analysis_command(campaign, measured, dark=True, trajectories=True, workers=8)

    assert command[:2] == ["openfe-api", "analyze"]
    assert str(campaign.directory.resolve()) in command
    assert "--dark" in command
    assert command[-3:] == ["--trajectories", "--workers", "8"]
    assert str(measured.resolve()) in command


def test_a_foreground_option_is_left_out_when_unset(tmp_path: Path) -> None:
    command = analysis_command(
        abfe_campaign(tmp_path), None, dark=False, trajectories=False, workers=4
    )

    assert "--measured" not in command
    assert "--dark" not in command
    assert "--workers" not in command


def test_a_dry_run_writes_the_script_without_submitting(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    submission = submit_analysis(
        campaign, SLURM_PROFILE, trajectories=True, workers=12, dry_run=True
    )

    assert submission.job_id is None
    script = submission.script.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --qos=gpu_access" in script
    assert "#SBATCH --array" not in script
    assert "module load cuda" in script
    assert "openfe-api analyze" in script


def test_the_job_asks_for_a_core_per_worker(tmp_path: Path) -> None:
    """Workers that outnumber the profile's cores would otherwise share them."""
    campaign = abfe_campaign(tmp_path)

    submission = submit_analysis(campaign, SLURM_PROFILE, workers=16, dry_run=True)

    assert submission.cpus == 16
    assert "#SBATCH --cpus-per-task=16" in submission.script.read_text(encoding="utf-8")


def test_a_profile_without_slurm_is_refused(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    with pytest.raises(OpenFEAPIError, match="no 'slurm' section"):
        submit_analysis(campaign, LOCAL_PROFILE, dry_run=True)
