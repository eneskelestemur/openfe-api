"""Tests for running a SepTop campaign: one transformation per edge, and GPU packing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from helpers import mini_series_data
from openfe_api.campaign import Campaign, RunState
from openfe_api.execution.base import STANDARD_CLASS, build_tasks
from openfe_api.execution.runner import packing_warning
from openfe_api.execution.slurm import group_tasks, render_script, submit_to_slurm
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile
from openfe_api.schema.request import validate_request

SLURM_PROFILE = ExecutionProfile(
    name="test_slurm",
    backend="slurm",
    jobs_per_gpu=1,
    slurm=SlurmProfile(partition="gpu", gres="gpu:1", time="2-00:00:00"),
)


def campaign_with_edges(tmp_path: Path, edges: list[str], repeats: int = 1) -> Campaign:
    """Create a planned SepTop campaign without running the planner.

    Args:
        tmp_path: Pytest temporary directory.
        edges: Edge names.
        repeats: Repeats per edge.

    Returns:
        The campaign, with one plan file per edge.
    """
    request = validate_request(
        mini_series_data("septop", "septop_campaign", execution={"repeats": repeats})
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    plans = campaign.directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": "septop",
        "hub": "ref",
        "edges": [{"name": name, "ligand_a": "ref", "ligand_b": name} for name in edges],
        "dropped": {},
        "unreachable": [],
        "warnings": [],
    }
    for name in edges:
        (plans / f"septop_{name}.json").write_text("{}", encoding="utf-8")
    (plans / "network.json").write_text(json.dumps(summary), encoding="utf-8")

    campaign.add_runs(edges, repeats=repeats, state=RunState.PREPARED)
    for name in edges:
        campaign.set_state(name, RunState.PLANNED)
    return campaign


def test_an_edge_is_one_task_per_repeat(tmp_path: Path) -> None:
    """The protocol builds both phases itself, so an edge is one process, as ABFE is."""
    campaign = campaign_with_edges(tmp_path, ["a_to_b"], repeats=2)

    tasks = build_tasks(campaign)

    assert sorted(task.label for task in tasks) == ["a_to_b/repeat1", "a_to_b/repeat2"]
    assert {task.phase for task in tasks} == {None}


def test_the_task_runs_the_septop_transformation(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, ["a_to_b"])

    task = build_tasks(campaign)[0]

    assert task.transformation.name == "septop_a_to_b.json"
    assert task.work_dir == campaign.directory / "runs" / "a_to_b" / "repeat1"


def test_every_edge_shares_one_cost_class(tmp_path: Path) -> None:
    """SepTop refuses a charge change outright, so no edge is more expensive than another."""
    campaign = campaign_with_edges(tmp_path, ["a_to_b", "a_to_c"])

    assert {task.cost_class for task in build_tasks(campaign)} == {STANDARD_CLASS}


def test_one_array_covers_the_campaign(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, ["a_to_b", "a_to_c"])

    submission = submit_to_slurm(
        build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True
    )

    assert len(submission.arrays) == 1
    assert submission.n_tasks == 2


def test_the_generated_script_is_valid_bash(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, ["a_to_b"])

    script = render_script(
        group_tasks(build_tasks(campaign), 1), SLURM_PROFILE, campaign.directory, STANDARD_CLASS
    )
    path = tmp_path / "submit.sh"
    path.write_text(script, encoding="utf-8")

    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0


def test_an_unplanned_edge_produces_no_tasks(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, ["a_to_b"])
    campaign.add_runs(["a_to_d"], repeats=1, state=RunState.PREPARED)

    assert {task.run_name for task in build_tasks(campaign)} == {"a_to_b"}


def test_sharing_a_gpu_is_warned_about_at_two_jobs(tmp_path: Path) -> None:
    """The solvent phase runs 27 windows, so two jobs already exceed the MPS ceiling."""
    campaign = campaign_with_edges(tmp_path, ["a_to_b"])
    profile = SLURM_PROFILE.model_copy(update={"jobs_per_gpu": 2})

    warning = packing_warning(campaign, profile)

    assert warning is not None
    assert "27 lambda windows" in warning
    assert "past the 48 contexts" in warning


def test_one_job_per_gpu_is_never_warned_about(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, ["a_to_b"])

    assert packing_warning(campaign, SLURM_PROFILE) is None
