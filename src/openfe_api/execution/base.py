"""The unit of execution: one transformation, one repeat.

OpenFE's own ``protocol_repeats`` runs every repeat inside a single process. openfe-api
instead plans with one repeat per transformation and runs each repeat as its own
``openfe quickrun`` process, so repeats can be spread across GPUs, packed onto one GPU
through MPS, and resumed individually after a wall-time limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.protocols.rbfe import PHASES
from openfe_api.results.gather import result_is_successful
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import CampaignRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = [
    "CHARGE_CLASS",
    "STANDARD_CLASS",
    "Task",
    "build_tasks",
    "quickrun_command",
]

STANDARD_CLASS = "standard"
"""Cost class of an ordinary transformation."""

CHARGE_CLASS = "charge"
"""Cost class of a charge-corrected RBFE edge: 22 lambda windows sampled for 20 ns each."""


@dataclass(frozen=True)
class Task:
    """One repeat of one transformation.

    Attributes:
        run_name: Name of the complex or edge this repeat belongs to.
        repeat: One-based repeat index.
        transformation: The transformation JSON to run.
        work_dir: Directory for simulation output and the resume cache.
        result_path: JSON file the results are written to.
        phase: Which leg this task runs, for a protocol whose legs are separate
            transformations. None when the protocol builds both legs itself, as ABFE does.
        cost_class: Which wall-time class this task belongs to. Tasks of different classes
            are submitted as separate job arrays, so a long edge does not inherit a short
            edge's time limit.
        expects_estimate: Whether this task's protocol produces a free energy. Plain MD does
            not, so its completion is judged by the trajectory instead.
    """

    run_name: str
    repeat: int
    transformation: Path
    work_dir: Path
    result_path: Path
    phase: str | None = None
    cost_class: str = STANDARD_CLASS
    expects_estimate: bool = True

    @property
    def label(self) -> str:
        """Return a short identifier for the task.

        Returns:
            A string such as ``mini/repeat2``, or ``a_to_b/solvent/repeat2`` when the task
            runs one named leg.
        """
        if self.phase is None:
            return f"{self.run_name}/repeat{self.repeat}"
        return f"{self.run_name}/{self.phase}/repeat{self.repeat}"

    @property
    def is_complete(self) -> bool:
        """Whether this repeat has finished successfully.

        A failed run still writes a result file, recording a null estimate, so the file's
        presence alone would make a failed repeat look finished and stop it being retried.
        What counts as finished depends on the protocol: see ``expects_estimate``.

        Returns:
            True if the result file holds a usable estimate.
        """
        return self.result_path.is_file() and result_is_successful(
            self.result_path, expects_estimate=self.expects_estimate
        )

    @property
    def can_resume(self) -> bool:
        """Whether an interrupted run of this repeat can be resumed.

        ``openfe quickrun`` writes a plan to a cache directory before execution starts and
        removes it on completion, so a cache that is still present means the run stopped
        part way through.

        Returns:
            True if a resume cache is present.
        """
        cache = self.work_dir / "quickrun_cache"
        return cache.is_dir() and any(cache.glob("dag-cache-*.json"))


def _charge_classes(campaign: Campaign) -> set[str]:
    """Return the names of edges whose plan applied an explicit charge correction."""
    summary = campaign.directory / "plans" / "network.json"
    if not summary.is_file():
        return set()
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {
        edge["name"]
        for edge in payload.get("edges", [])
        if isinstance(edge, dict) and edge.get("corrected")
    }


def _layout(request: CampaignRequest) -> tuple[str, list[str | None]]:
    """Return the transformation file prefix and the phases a protocol runs separately."""
    match request:
        case AbfeRequest():
            return "abfe", [None]
        case RbfeRequest():
            return "rbfe", list(PHASES)
        case SepTopRequest():
            return "septop", [None]
        case MdRequest():
            return "md", [None]
        case _:
            raise OpenFEAPIError(f"execution does not handle the '{request.protocol}' protocol")


def build_tasks(campaign: Campaign, only: list[str] | None = None) -> list[Task]:
    """Build the tasks for every planned run in a campaign.

    An ABFE complex and a SepTop edge are each one transformation, because the protocol builds
    both of their phases. An RBFE edge is two, one per phase, and both must run for the edge
    to have a result.

    Args:
        campaign: The campaign to build tasks for.
        only: Names of runs to include. If None, every planned run is included.

    Returns:
        Tasks in run order, then phase order, then repeat order. Runs that are not planned
        are skipped.

    Raises:
        OpenFEAPIError: If the campaign's protocol has no execution layout.
    """
    selected = set(only) if only is not None else None
    request = campaign.manifest.request
    prefix, phases = _layout(request)
    corrected = _charge_classes(campaign) if isinstance(request, RbfeRequest) else set()
    expects_estimate = not isinstance(request, MdRequest)
    tasks: list[Task] = []

    for name, record in campaign.manifest.runs.items():
        if selected is not None and name not in selected:
            continue
        if record.state is RunState.PENDING or record.state is RunState.PREPARED:
            continue

        cost_class = CHARGE_CLASS if name in corrected else STANDARD_CLASS

        for phase in phases:
            if phase is None:
                transformation = campaign.directory / "plans" / f"{prefix}_{name}.json"
                run_dir = campaign.directory / "runs" / name
            else:
                transformation = campaign.directory / "plans" / f"{prefix}_{name}_{phase}.json"
                run_dir = campaign.directory / "runs" / name / phase
            if not transformation.is_file():
                continue

            for repeat in range(1, record.repeats + 1):
                work_dir = run_dir / f"repeat{repeat}"
                tasks.append(
                    Task(
                        run_name=name,
                        repeat=repeat,
                        transformation=transformation,
                        work_dir=work_dir,
                        result_path=work_dir / "results.json",
                        phase=phase,
                        cost_class=cost_class,
                        expects_estimate=expects_estimate,
                    )
                )

    return tasks


def quickrun_command(task: Task, resume: bool) -> list[str]:
    """Build the ``openfe quickrun`` command line for a task.

    Args:
        task: The task to run.
        resume: Whether to pass ``--resume``, continuing from the cached plan.

    Returns:
        The command as a list of arguments.
    """
    command = [
        "openfe",
        "quickrun",
        str(task.transformation),
        "-d",
        str(task.work_dir),
        "-o",
        str(task.result_path),
    ]
    if resume:
        command.append("--resume")
    return command
