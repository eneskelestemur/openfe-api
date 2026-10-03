"""Tests for running an RBFE campaign: edge and phase tasks, and cost classes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from helpers import mini_series_data
from openfe_api.campaign import Campaign, RunState
from openfe_api.execution.base import CHARGE_CLASS, STANDARD_CLASS, build_tasks
from openfe_api.execution.runner import packing_warning
from openfe_api.execution.slurm import group_tasks, render_script, submit_to_slurm
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile
from openfe_api.schema.request import validate_request

SLURM_PROFILE = ExecutionProfile(
    name="test_slurm",
    backend="slurm",
    jobs_per_gpu=2,
    slurm=SlurmProfile(
        partition="gpu",
        gres="gpu:1",
        time="1-00:00:00",
        time_by_class={CHARGE_CLASS: "4-00:00:00"},
    ),
)


def campaign_with_edges(
    tmp_path: Path, edges: list[tuple[str, bool]], repeats: int = 1
) -> Campaign:
    """Create a planned RBFE campaign without running the planner.

    Args:
        tmp_path: Pytest temporary directory.
        edges: Edge names paired with whether each applies a charge correction.
        repeats: Repeats per edge.

    Returns:
        The campaign, with a plan file per edge and phase.
    """
    request = validate_request(
        mini_series_data("rbfe", "edge_campaign", execution={"repeats": repeats})
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    plans = campaign.directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"edges": [], "dropped": {}, "unreachable": [], "warnings": []}
    for name, corrected in edges:
        for phase in ("solvent", "complex"):
            (plans / f"rbfe_{name}_{phase}.json").write_text("{}", encoding="utf-8")
        summary["edges"].append({"name": name, "corrected": corrected})

    (plans / "network.json").write_text(json.dumps(summary), encoding="utf-8")
    campaign.add_runs([name for name, _ in edges], repeats=repeats, state=RunState.PREPARED)
    for name, _ in edges:
        campaign.set_state(name, RunState.PLANNED)
    return campaign


def test_each_edge_becomes_two_tasks_per_repeat(tmp_path: Path) -> None:
    """Both phases must run: their difference is the relative binding free energy."""
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)], repeats=2)

    tasks = build_tasks(campaign)

    assert len(tasks) == 4
    assert sorted(task.label for task in tasks) == [
        "a_to_b/complex/repeat1",
        "a_to_b/complex/repeat2",
        "a_to_b/solvent/repeat1",
        "a_to_b/solvent/repeat2",
    ]


def test_phases_get_separate_working_directories(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])

    directories = {task.work_dir for task in build_tasks(campaign)}

    assert len(directories) == 2
    assert all("a_to_b" in str(directory) for directory in directories)


def test_a_neutral_edge_is_the_standard_class(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])

    assert {task.cost_class for task in build_tasks(campaign)} == {STANDARD_CLASS}


def test_a_corrected_edge_is_the_charge_class(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", True)])

    assert {task.cost_class for task in build_tasks(campaign)} == {CHARGE_CLASS}


def test_classes_are_submitted_as_separate_arrays(tmp_path: Path) -> None:
    """One array shares one wall-time limit, so a long edge needs its own array."""
    campaign = campaign_with_edges(tmp_path, [("cheap", False), ("expensive", True)])

    submission = submit_to_slurm(
        build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True
    )

    assert sorted(array.cost_class for array in submission.arrays) == [
        CHARGE_CLASS,
        STANDARD_CLASS,
    ]
    assert submission.n_tasks == 4


def test_each_array_gets_its_own_wall_time(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("cheap", False), ("expensive", True)])

    submit_to_slurm(build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True)

    standard = (campaign.directory / f"submit_{STANDARD_CLASS}.sh").read_text()
    charge = (campaign.directory / f"submit_{CHARGE_CLASS}.sh").read_text()
    assert "--time=1-00:00:00" in standard
    assert "--time=4-00:00:00" in charge


def test_a_class_without_its_own_time_uses_the_default(tmp_path: Path) -> None:
    profile = ExecutionProfile(
        name="plain",
        backend="slurm",
        jobs_per_gpu=1,
        slurm=SlurmProfile(partition="gpu", time="2:00:00"),
    )
    campaign = campaign_with_edges(tmp_path, [("expensive", True)])

    submit_to_slurm(build_tasks(campaign), profile, campaign.directory, dry_run=True)

    assert "--time=2:00:00" in (campaign.directory / f"submit_{CHARGE_CLASS}.sh").read_text()


def test_groups_never_mix_cost_classes(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("cheap", False), ("expensive", True)])

    submission = submit_to_slurm(
        build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True
    )

    for array in submission.arrays:
        for group in array.groups:
            assert {task.cost_class for task in group} == {array.cost_class}


def test_the_generated_scripts_are_valid_bash(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("cheap", False), ("expensive", True)])
    tasks = build_tasks(campaign)

    for cost_class in (STANDARD_CLASS, CHARGE_CLASS):
        selected = [task for task in tasks if task.cost_class == cost_class]
        script = render_script(
            group_tasks(selected, SLURM_PROFILE.jobs_per_gpu),
            SLURM_PROFILE,
            campaign.directory,
            cost_class,
        )
        path = tmp_path / f"{cost_class}.sh"
        path.write_text(script, encoding="utf-8")
        checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, check=False)
        assert checked.returncode == 0


def test_log_paths_separate_the_classes(tmp_path: Path) -> None:
    """Two arrays run at once, so their logs must not overwrite each other."""
    campaign = campaign_with_edges(tmp_path, [("cheap", False), ("expensive", True)])

    submit_to_slurm(build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True)

    standard = (campaign.directory / f"submit_{STANDARD_CLASS}.sh").read_text()
    assert f"openfe_{STANDARD_CLASS}_%A_%a.out" in standard


def test_packing_is_not_warned_about_for_an_rbfe_campaign(tmp_path: Path) -> None:
    """RBFE runs 11 windows, so two repeats stay inside the MPS context limit."""
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])

    warning = packing_warning(campaign, SLURM_PROFILE)

    assert warning is not None
    assert "11 lambda windows" in warning
    assert "past the 48 contexts" not in warning


def test_heavy_packing_is_still_warned_about(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])
    profile = SLURM_PROFILE.model_copy(update={"jobs_per_gpu": 6})

    warning = packing_warning(campaign, profile)

    assert warning is not None
    assert "past the 48 contexts" in warning


def test_one_job_per_gpu_is_never_warned_about(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])
    profile = SLURM_PROFILE.model_copy(update={"jobs_per_gpu": 1})

    assert packing_warning(campaign, profile) is None


def test_an_unplanned_edge_produces_no_tasks(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])
    campaign.add_runs(["c_to_d"], repeats=1, state=RunState.PREPARED)

    labels = {task.run_name for task in build_tasks(campaign)}

    assert labels == {"a_to_b"}


def test_the_script_waits_only_on_the_simulations(tmp_path: Path) -> None:
    """A bare wait would also wait for whatever the setup commands left running.

    A monitoring process would hold the array element open long after the work finished.
    """
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])
    tasks = build_tasks(campaign)

    script = render_script(group_tasks(tasks, 2), SLURM_PROFILE, campaign.directory, STANDARD_CLASS)

    assert "PIDS+=($!)" in script
    assert 'for pid in "${PIDS[@]}"; do wait "$pid"' in script
    assert "\n    wait\n" not in script


def test_a_failing_simulation_still_fails_the_element(tmp_path: Path) -> None:
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])

    script = render_script(
        group_tasks(build_tasks(campaign), 2),
        SLURM_PROFILE,
        campaign.directory,
        STANDARD_CLASS,
    )

    assert 'exit "$STATUS"' in script


def test_the_script_holds_absolute_paths(tmp_path: Path) -> None:
    """Slurm may start the job anywhere, so relative paths are not safe.

    A path relative to the submission directory breaks on a requeue, or when the campaign is
    submitted from another directory.
    """
    campaign = campaign_with_edges(tmp_path, [("a_to_b", False)])

    script = render_script(
        group_tasks(build_tasks(campaign), 1),
        SLURM_PROFILE,
        campaign.directory,
        STANDARD_CLASS,
    )

    campaign_lines = [line for line in script.splitlines() if "runs/" in line or "plans/" in line]
    assert campaign_lines
    for line in campaign_lines:
        assert str(tmp_path) in line, line
