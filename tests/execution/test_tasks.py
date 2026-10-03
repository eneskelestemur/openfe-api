"""Tests for turning a planned campaign into tasks, and knowing when one is done."""

from __future__ import annotations

from pathlib import Path

from helpers import SUCCESSFUL_RESULT, abfe_campaign, write_result_file
from openfe_api.execution.base import build_tasks, quickrun_command


def test_build_tasks_makes_one_task_per_repeat(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, repeats=3)

    tasks = build_tasks(campaign)

    assert [task.repeat for task in tasks] == [1, 2, 3]
    assert tasks[0].label == "mini/repeat1"


def test_build_tasks_skips_unplanned_runs(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)

    assert build_tasks(campaign) == []


def test_build_tasks_honours_only(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)

    assert build_tasks(campaign, only=["absent"]) == []
    assert len(build_tasks(campaign, only=["mini"])) == 3


def test_task_knows_when_it_is_complete(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    task = build_tasks(campaign)[0]

    assert task.is_complete is False
    task.result_path.parent.mkdir(parents=True, exist_ok=True)
    task.result_path.write_text(SUCCESSFUL_RESULT, encoding="utf-8")
    assert task.is_complete is True


def test_task_detects_a_resume_cache(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    task = build_tasks(campaign)[0]

    assert task.can_resume is False
    cache = task.work_dir / "quickrun_cache"
    cache.mkdir(parents=True)
    (cache / "dag-cache-abc.json").write_text("{}", encoding="utf-8")
    assert task.can_resume is True


def test_quickrun_command_shape(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    task = build_tasks(campaign)[0]

    command = quickrun_command(task, resume=False)

    assert command[:2] == ["openfe", "quickrun"]
    assert "--resume" not in command
    assert quickrun_command(task, resume=True)[-1] == "--resume"


def test_a_failed_repeat_is_not_complete(tmp_path: Path) -> None:
    """Regression: quickrun writes a result file even when the run fails.

    Treating that file as a finished repeat meant a failed repeat was skipped forever.
    """
    campaign = abfe_campaign(tmp_path)
    task = build_tasks(campaign)[0]

    write_result_file(task, estimate=None)

    assert task.result_path.is_file()
    assert task.is_complete is False


def test_a_successful_repeat_is_complete(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path)
    task = build_tasks(campaign)[0]

    write_result_file(task, estimate=-8.0)

    assert task.is_complete is True
