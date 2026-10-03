"""Tests for running tasks on this machine: devices, interrupts and logs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import openfe_api.execution.runner as runner_module
from helpers import (
    LOCAL_PROFILE,
    SLURM_PROFILE,
    SUCCESSFUL_RESULT,
    abfe_campaign,
    write_result_file,
)
from openfe_api.campaign import RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.base import Task, build_tasks
from openfe_api.execution.local import detect_gpus, run_locally
from openfe_api.execution.runner import submit_campaign
from openfe_api.schema.profiles import ExecutionProfile


def _fake_command(exit_code: int) -> Any:
    """Build a command factory that runs a trivial process instead of OpenFE.

    Args:
        exit_code: Exit code the fake process should return.

    Returns:
        A command factory suitable for ``run_locally``.
    """

    def factory(task: Task, resume: bool) -> list[str]:
        del resume
        if exit_code != 0:
            return [sys.executable, "-c", f"import sys; sys.exit({exit_code})"]
        script = (
            "import sys, pathlib;"
            f"pathlib.Path({str(task.result_path)!r}).write_text({SUCCESSFUL_RESULT!r});"
            "sys.exit(0)"
        )
        return [sys.executable, "-c", script]

    return factory


def test_run_locally_runs_every_task(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    tasks = build_tasks(campaign)

    result = run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(0))

    assert sorted(result.completed) == ["mini/repeat1", "mini/repeat2", "mini/repeat3"]
    assert result.failed == {}
    assert all(task.is_complete for task in tasks)


def test_run_locally_records_failures(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    tasks = build_tasks(campaign)

    result = run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(3))

    assert result.completed == []
    assert set(result.failed) == {"mini/repeat1", "mini/repeat2", "mini/repeat3"}
    assert "exit code 3" in result.failed["mini/repeat1"]


def test_run_locally_skips_completed_tasks(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    tasks = build_tasks(campaign)
    run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(0))

    again = run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(3))

    assert len(again.skipped) == 3
    assert again.failed == {}


def test_run_locally_force_reruns(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    tasks = build_tasks(campaign)
    run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(0))

    again = run_locally(tasks, LOCAL_PROFILE, force=True, command_factory=_fake_command(0))

    assert len(again.completed) == 3


def test_run_locally_writes_a_log(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    tasks = build_tasks(campaign)

    run_locally(tasks, LOCAL_PROFILE, command_factory=_fake_command(0))

    assert (tasks[0].work_dir / "quickrun.log").is_file()


def test_submit_campaign_requires_a_plan(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)

    with pytest.raises(OpenFEAPIError, match="no planned transformations"):
        submit_campaign(campaign, LOCAL_PROFILE)


def test_submit_campaign_to_slurm_records_the_job(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    outcome = submit_campaign(campaign, SLURM_PROFILE, dry_run=True)

    assert outcome.backend == "slurm"
    assert outcome.job_id is None
    assert campaign.run("mini").state is RunState.PLANNED


def _self_interrupting_command(task: Task, resume: bool) -> list[str]:
    """Build a command that stops itself with SIGINT, as Ctrl+C in a terminal does.

    Args:
        task: The task being run.
        resume: Unused; present to match the command factory signature.

    Returns:
        The command.
    """
    del task, resume
    return [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGINT)"]


def test_an_interrupt_stops_the_batch(tmp_path: Path) -> None:
    """Regression: an interrupted repeat freed its slot and the next repeat started."""
    campaign = abfe_campaign(tmp_path, repeats=3)
    tasks = build_tasks(campaign)

    result = run_locally(tasks, LOCAL_PROFILE, command_factory=_self_interrupting_command)

    assert result.completed == []
    assert result.failed == {}
    assert sorted(result.interrupted) == ["mini/repeat1", "mini/repeat2", "mini/repeat3"]


def test_an_interrupt_is_not_recorded_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = abfe_campaign(tmp_path, repeats=2)

    def interrupted_run(tasks: list[Task], profile: ExecutionProfile, force: bool = False) -> Any:
        return run_locally(tasks, profile, force=force, command_factory=_self_interrupting_command)

    monkeypatch.setattr(runner_module, "run_locally", interrupted_run)

    outcome = submit_campaign(campaign, LOCAL_PROFILE)

    record = campaign.run("mini")
    assert len(outcome.interrupted) == 2
    assert record.state is RunState.FAILED
    assert record.message is not None and "run submit again to resume" in record.message


def test_detect_gpus_uses_the_allocated_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert detect_gpus() == ["2", "3"]


def test_detect_gpus_keeps_uuid_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: UUID allocations were dropped, and every GPU on the node used instead.

    Schedulers may hand out GPUs by UUID. Falling back to nvidia-smi then spreads repeats
    over GPUs the job does not own, which on a shared node belong to someone else.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-3f1a,GPU-9c02")

    assert detect_gpus() == ["GPU-3f1a", "GPU-9c02"]


def test_detect_gpus_falls_back_when_nothing_is_allocated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("openfe_api.execution.local.shutil.which", lambda _: None)

    assert detect_gpus() == ["0"]


def test_each_repeat_is_pinned_to_its_own_device(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=2)
    profile = ExecutionProfile(name="two", backend="local", gpus=[4, 5], jobs_per_gpu=1)

    def record_device(task: Task, resume: bool) -> list[str]:
        del resume
        return [
            sys.executable,
            "-c",
            (
                "import os, pathlib; "
                f"pathlib.Path({str(task.work_dir / 'device')!r})"
                ".write_text(os.environ['CUDA_VISIBLE_DEVICES'])"
            ),
        ]

    run_locally(build_tasks(campaign), profile, command_factory=record_device)

    devices = {
        (task.work_dir / "device").read_text(encoding="utf-8") for task in build_tasks(campaign)
    }
    assert devices == {"4", "5"}


def test_a_failed_repeat_is_rerun_while_a_finished_one_is_skipped(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=2)
    finished, failed = build_tasks(campaign)
    write_result_file(finished, estimate=-8.0)
    write_result_file(failed, estimate=None)

    result = run_locally(build_tasks(campaign), LOCAL_PROFILE, command_factory=_fake_command(0))

    assert result.skipped == [finished.label]
    assert result.completed == [failed.label]


def _gpu_exhausted(task: Task, resume: bool) -> list[str]:
    """Build a command that fails the way a GPU out of contexts does."""
    del task, resume
    return [
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('OpenMMException: No compatible CUDA device is available'); "
            "sys.exit(1)"
        ),
    ]


def test_a_gpu_exhaustion_failure_explains_packing(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=1)
    packed = ExecutionProfile(name="packed", backend="local", gpus=[0], jobs_per_gpu=3)

    result = run_locally(build_tasks(campaign), packed, command_factory=_gpu_exhausted)

    assert "3 repeats were sharing it" in result.failed["mini/repeat1"]
    assert "jobs_per_gpu" in result.failed["mini/repeat1"]


def test_no_packing_hint_without_packing(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=1)

    result = run_locally(build_tasks(campaign), LOCAL_PROFILE, command_factory=_gpu_exhausted)

    assert "jobs_per_gpu" not in result.failed["mini/repeat1"]
