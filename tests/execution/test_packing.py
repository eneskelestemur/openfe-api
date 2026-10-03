"""Tests for the warning about packing repeats onto one GPU."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from helpers import LOCAL_PROFILE, abfe_campaign
from openfe_api.execution.runner import MPS_CONTEXT_LIMIT, packing_warning, submit_campaign
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile


def test_packing_warns_about_the_context_pile_up(tmp_path: Path) -> None:
    """A leg leaves a context per lambda window, so packing asks for far more than it looks."""
    campaign = abfe_campaign(tmp_path)
    packed = ExecutionProfile(name="packed", backend="local", jobs_per_gpu=3)

    warning = packing_warning(campaign, packed)

    assert warning is not None
    assert "90 contexts" in warning  # 30 complex windows x 3 repeats
    assert str(MPS_CONTEXT_LIMIT) in warning


def test_no_packing_warning_for_one_repeat_per_gpu(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    assert packing_warning(campaign, LOCAL_PROFILE) is None


def test_packing_warning_is_logged_on_submit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = abfe_campaign(tmp_path)
    packed = ExecutionProfile(
        name="packed", backend="slurm", jobs_per_gpu=3, slurm=SlurmProfile(partition="gpu")
    )
    # An entry point may have detached the package logger from the root one.
    monkeypatch.setattr(logging.getLogger("openfe_api"), "propagate", True)

    with caplog.at_level(logging.WARNING, logger="openfe_api.execution.runner"):
        submit_campaign(campaign, packed, dry_run=True)

    assert any("contexts at once" in record.getMessage() for record in caplog.records)
