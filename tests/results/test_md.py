"""Tests for collecting what a plain MD campaign produced."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from helpers import write_md_result
from openfe_api.campaign import Campaign, RunState
from openfe_api.execution.base import build_tasks
from openfe_api.execution.runner import packing_warning
from openfe_api.results.gather import result_is_successful
from openfe_api.results.md import gather_md, read_md_repeat, write_md_tsv
from openfe_api.results.runner import gather_campaign
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"

PROFILE = ExecutionProfile(
    name="packed",
    backend="slurm",
    jobs_per_gpu=4,
    slurm=SlurmProfile(partition="gpu", gres="gpu:1", time="1-00:00:00"),
)


def md_campaign(tmp_path: Path, systems: list[str], repeats: int = 1) -> Campaign:
    """Create a planned MD campaign without running the planner.

    Args:
        tmp_path: Pytest temporary directory.
        systems: System names.
        repeats: Repeats per system.

    Returns:
        The campaign, with one plan file per system.
    """
    request = validate_request(
        {
            "protocol": "md",
            "name": "md_campaign",
            "systems": [
                {
                    "name": name,
                    "structure": str(DATA / "mini_complex.cif"),
                    "protein": {"chains": ["A"]},
                    "ligands": [
                        {"name": "lig", "selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"}
                    ],
                }
                for name in systems
            ],
            "execution": {"repeats": repeats},
        }
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    plans = campaign.directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    for name in systems:
        (plans / f"md_{name}.json").write_text("{}", encoding="utf-8")
        campaign.set_state(name, RunState.PREPARED)
        campaign.set_state(name, RunState.PLANNED)
    return campaign


def test_a_finished_md_repeat_is_recognized_as_finished(tmp_path: Path) -> None:
    """MD writes a null estimate by design, which used to read as an unfinished run."""
    campaign = md_campaign(tmp_path, ["mini"])
    task = build_tasks(campaign)[0]
    write_md_result(task.result_path)

    assert task.expects_estimate is False
    assert task.is_complete


def test_an_interrupted_md_repeat_is_not_finished(tmp_path: Path) -> None:
    campaign = md_campaign(tmp_path, ["mini"])
    task = build_tasks(campaign)[0]
    write_md_result(task.result_path, no_trajectory=True)

    assert not task.is_complete


def test_a_failed_md_repeat_is_not_finished(tmp_path: Path) -> None:
    campaign = md_campaign(tmp_path, ["mini"])
    task = build_tasks(campaign)[0]
    write_md_result(task.result_path, failed=True)

    assert not task.is_complete


def test_a_null_estimate_still_fails_a_protocol_that_has_one(tmp_path: Path) -> None:
    """The MD rule must not leak into the protocols whose estimate is the answer."""
    path = write_md_result(tmp_path / "results.json")

    assert result_is_successful(path, expects_estimate=True) is False
    assert result_is_successful(path, expects_estimate=False) is True


def test_a_system_is_one_task_per_repeat(tmp_path: Path) -> None:
    campaign = md_campaign(tmp_path, ["mini"], repeats=3)

    tasks = build_tasks(campaign)

    assert sorted(task.label for task in tasks) == [
        "mini/repeat1",
        "mini/repeat2",
        "mini/repeat3",
    ]
    assert {task.transformation.name for task in tasks} == {"md_mini.json"}


def test_packing_is_never_warned_about_for_md(tmp_path: Path) -> None:
    """One simulation holds one context, so there is no pile-up to warn about."""
    campaign = md_campaign(tmp_path, ["mini"])

    assert packing_warning(campaign, PROFILE) is None


def test_the_artifacts_are_collected(tmp_path: Path) -> None:
    write_md_result(tmp_path / "runs" / "mini" / "repeat1" / "results.json")

    results = gather_md(tmp_path, ["mini"])

    artifacts = results[0].repeats[0].artifacts
    assert sorted(artifacts) == [
        "minimized",
        "npt_structure",
        "nvt_structure",
        "topology",
        "trajectory",
    ]
    assert results[0].ok


def test_a_missing_checkpoint_is_simply_absent(tmp_path: Path) -> None:
    """A run shorter than the checkpoint interval writes none, and reports None for it."""
    path = write_md_result(tmp_path / "results.json")

    assert "checkpoint" not in read_md_repeat(path, 1).artifacts


def test_a_failed_repeat_is_reported_with_a_reason(tmp_path: Path) -> None:
    write_md_result(tmp_path / "runs" / "mini" / "repeat1" / "results.json", failed=True)

    results = gather_md(tmp_path, ["mini"])

    assert not results[0].ok
    assert "raised an exception" in (results[0].repeats[0].failure or "")


def test_a_system_with_no_results_is_reported_as_such(tmp_path: Path) -> None:
    results = gather_md(tmp_path, ["mini"])

    assert results[0].repeats == []
    assert not results[0].ok


def test_the_table_is_written(tmp_path: Path) -> None:
    write_md_result(tmp_path / "runs" / "mini" / "repeat1" / "results.json")
    write_md_result(tmp_path / "runs" / "mini" / "repeat2" / "results.json", failed=True)
    results = gather_md(tmp_path, ["mini"])

    path = tmp_path / "results" / "simulations.tsv"
    write_md_tsv(results, path)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["status"] for row in rows] == ["finished", "failed"]
    assert rows[0]["trajectory"].endswith("simulation.xtc")


def test_gathering_marks_a_finished_system_done(tmp_path: Path) -> None:
    campaign = md_campaign(tmp_path, ["mini"])
    campaign.set_state("mini", RunState.SUBMITTED)
    write_md_result(campaign.directory / "runs" / "mini" / "repeat1" / "results.json")

    outcome: Any = gather_campaign(campaign, write=True)

    assert outcome.md[0].ok
    assert campaign.run("mini").state is RunState.DONE
    assert (campaign.directory / "results" / "simulations.tsv").is_file()


def test_gathering_reports_a_system_with_nothing_yet(tmp_path: Path) -> None:
    campaign = md_campaign(tmp_path, ["mini"])

    outcome = gather_campaign(campaign, write=False)

    assert outcome.missing == ["mini"]
