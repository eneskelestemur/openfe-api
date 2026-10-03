"""Tests for grouping tasks into a Slurm array and rendering its batch script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helpers import SLURM_PROFILE, abfe_campaign
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.base import Task, build_tasks
from openfe_api.execution.slurm import group_tasks, render_script, submit_to_slurm


def test_group_tasks_packs_repeats_together(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=6)
    tasks = build_tasks(campaign)

    groups = group_tasks(tasks, jobs_per_gpu=3)

    assert [len(group) for group in groups] == [3, 3]
    assert [task.repeat for task in groups[0]] == [1, 2, 3]


def test_group_tasks_fills_every_slot_across_ligands() -> None:
    """A ligand with fewer repeats than slots must not waste the rest of the GPU."""
    tasks = [
        Task(
            run_name=name,
            repeat=repeat,
            transformation=Path("plan.json"),
            work_dir=Path(f"{name}/repeat{repeat}"),
            result_path=Path(f"{name}/repeat{repeat}/results.json"),
        )
        for name in ("lig_a", "lig_b")
        for repeat in (1, 2, 3)
    ]

    groups = group_tasks(tasks, jobs_per_gpu=4)

    assert [len(group) for group in groups] == [4, 2]
    assert [task.run_name for task in groups[0]] == ["lig_a", "lig_a", "lig_a", "lig_b"]


def test_group_tasks_with_more_slots_than_tasks(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=3)

    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=5)

    assert [len(group) for group in groups] == [3]


def test_group_tasks_without_packing(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=3)

    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=1)

    assert [len(group) for group in groups] == [1, 1, 1]


def test_rendered_script_is_valid_bash(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=3)

    script = render_script(groups, SLURM_PROFILE, campaign.directory)
    path = tmp_path / "submit.sh"
    path.write_text(script, encoding="utf-8")

    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, check=False)
    assert checked.returncode == 0


def test_rendered_script_carries_the_profile(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=3)

    script = render_script(groups, SLURM_PROFILE, campaign.directory)

    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --qos=gpu_access" in script
    assert "#SBATCH --array=0-0" in script
    assert "module load cuda" in script


def test_rendered_script_enables_mps_with_cleanup(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=3)

    script = render_script(groups, SLURM_PROFILE, campaign.directory)

    assert "nvidia-cuda-mps-control -d" in script
    assert "trap cleanup_mps EXIT" in script


def test_rendered_script_omits_mps_for_one_job_per_gpu(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    profile = SLURM_PROFILE.model_copy(update={"jobs_per_gpu": 1})
    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=1)

    script = render_script(groups, profile, campaign.directory)

    assert "nvidia-cuda-mps-control" not in script


def test_rendered_script_resumes_only_when_a_cache_exists(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    groups = group_tasks(build_tasks(campaign), jobs_per_gpu=3)

    script = render_script(groups, SLURM_PROFILE, campaign.directory)

    assert 'RESUME="--resume"' in script
    assert "quickrun_cache" in script


def test_dry_run_writes_a_script_without_submitting(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    submission = submit_to_slurm(
        build_tasks(campaign), SLURM_PROFILE, campaign.directory, dry_run=True
    )

    assert submission.job_id is None
    assert submission.script is not None
    assert submission.script.is_file()
    assert submission.n_tasks == 3


def test_submitting_nothing_is_an_error(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)

    with pytest.raises(OpenFEAPIError, match="no tasks to submit"):
        submit_to_slurm([], SLURM_PROFILE, campaign.directory)
