"""Submitting repeats to Slurm as a job array.

Each array element runs one group of repeats: with MPS enabled a group is several repeats
of the same transformation sharing one GPU, matching how these runs are packed in practice.
The generated script decides at runtime whether to resume, so a requeued job continues
from its cache rather than starting over.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.base import Task, quickrun_command
from openfe_api.execution.mps import mps_shell_lines
from openfe_api.log import get_logger
from openfe_api.schema.profiles import ExecutionProfile

__all__ = [
    "ArraySubmission",
    "SlurmSubmission",
    "group_tasks",
    "render_script",
    "submit_to_slurm",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArraySubmission:
    """One job array, holding the tasks of a single cost class.

    Attributes:
        cost_class: The class whose tasks this array runs.
        job_id: The Slurm job identifier, or None for a dry run.
        script: The generated batch script.
        groups: Task groups, one per array element.
    """

    cost_class: str
    job_id: str | None
    script: Path
    groups: list[list[Task]]

    @property
    def tasks(self) -> list[Task]:
        """Every task in this array.

        Returns:
            Tasks in array order.
        """
        return [task for group in self.groups for task in group]


@dataclass(frozen=True)
class SlurmSubmission:
    """The outcome of submitting a campaign, as one array per cost class.

    Attributes:
        arrays: The submitted arrays, one per cost class that had work.
    """

    arrays: list[ArraySubmission]

    @property
    def n_tasks(self) -> int:
        """Number of repeats submitted.

        Returns:
            The total across every array.
        """
        return sum(len(array.tasks) for array in self.arrays)

    @property
    def job_id(self) -> str | None:
        """The submitted job identifiers.

        Returns:
            A comma-separated list, or None when nothing was submitted.
        """
        ids = [array.job_id for array in self.arrays if array.job_id is not None]
        return ",".join(ids) if ids else None

    @property
    def script(self) -> Path | None:
        """The first generated script, for reporting a dry run.

        Returns:
            The script path, or None if no array was rendered.
        """
        return self.arrays[0].script if self.arrays else None


def group_tasks(tasks: list[Task], jobs_per_gpu: int) -> list[list[Task]]:
    """Group tasks into array elements, filling every GPU slot.

    Groups are filled to ``jobs_per_gpu`` regardless of which transformation a repeat
    belongs to, so a request for more slots than a ligand has repeats still uses the whole
    GPU. Tasks arrive ordered by run and then repeat, so repeats of one ligand stay
    together wherever a group is large enough to hold them.

    Args:
        tasks: The tasks to group, in run then repeat order.
        jobs_per_gpu: Tasks per group.

    Returns:
        Groups in task order, every group full except possibly the last.
    """
    return [tasks[start : start + jobs_per_gpu] for start in range(0, len(tasks), jobs_per_gpu)]


def _directives(
    profile: ExecutionProfile,
    campaign_dir: Path,
    n_groups: int,
    cost_class: str,
) -> list[str]:
    """Build the ``#SBATCH`` lines for one array."""
    if profile.slurm is None:
        raise OpenFEAPIError(f"profile '{profile.name}' has no 'slurm' section")

    slurm = profile.slurm
    logs = campaign_dir / "logs"
    lines = [
        f"#SBATCH --job-name=openfe-api-{cost_class}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --time={slurm.time_by_class.get(cost_class, slurm.time)}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.memory}",
        f"#SBATCH --array=0-{n_groups - 1}",
        f"#SBATCH --output={logs}/openfe_{cost_class}_%A_%a.out",
        f"#SBATCH --error={logs}/openfe_{cost_class}_%A_%a.err",
    ]
    if slurm.gres:
        lines.append(f"#SBATCH --gres={slurm.gres}")
    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    lines.extend(f"#SBATCH {directive}" for directive in slurm.extra_directives)
    return lines


def render_script(
    groups: list[list[Task]],
    profile: ExecutionProfile,
    campaign_dir: Path,
    cost_class: str = "standard",
) -> str:
    """Render the batch script for one job array.

    Args:
        groups: Task groups, one per array element.
        profile: The execution profile.
        campaign_dir: Campaign directory, used for log paths.
        cost_class: The class this array runs.

    Returns:
        The script text.

    Raises:
        OpenFEAPIError: If the profile has no Slurm section.
    """
    lines = [
        "#!/bin/bash",
        *_directives(profile, campaign_dir, len(groups), cost_class),
        "",
        "set -euo pipefail",
        "",
    ]
    lines.extend(profile.setup_commands)
    lines.append("")

    use_mps = profile.jobs_per_gpu > 1
    start, stop = mps_shell_lines("${TMPDIR:-/tmp}/openfe-api-mps-${SLURM_JOB_ID}")
    if use_mps:
        # Stopped from a trap, so the daemon goes even when the job hits its wall-time limit.
        lines.extend(start)
        lines.append("")
        lines.append("cleanup_mps() {")
        lines.extend(f"    {line}" for line in stop)
        lines.extend(["}", "trap cleanup_mps EXIT", ""])

    lines.append('case "${SLURM_ARRAY_TASK_ID}" in')
    for index, group in enumerate(groups):
        lines.append(f"  {index})")
        lines.append("    PIDS=()")
        for task in group:
            command = " ".join(quickrun_command(task, resume=False))
            lines.extend(
                [
                    f'    mkdir -p "{task.work_dir}"',
                    '    RESUME=""',
                    f'    if [ -d "{task.work_dir}/quickrun_cache" ]; then RESUME="--resume"; fi',
                    f"    {command} $RESUME &",
                    "    PIDS+=($!)",
                ]
            )
        # Wait on the simulation pids, not every background job: a bare wait would also wait
        # on whatever the profile's setup commands left running.
        lines.extend(
            [
                "    STATUS=0",
                '    for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=$?; done',
                '    exit "$STATUS"',
                "    ;;",
            ]
        )
    lines.extend(
        [
            "  *)",
            '    echo "no work for array index ${SLURM_ARRAY_TASK_ID}" >&2',
            "    exit 1",
            "    ;;",
            "esac",
            "",
        ]
    )
    return "\n".join(lines)


def submit_to_slurm(
    tasks: list[Task],
    profile: ExecutionProfile,
    campaign_dir: Path,
    dry_run: bool = False,
) -> SlurmSubmission:
    """Write a batch script per cost class and submit each as its own job array.

    Tasks of different cost classes go in separate arrays because a Slurm array shares one
    wall-time limit, and a charge-corrected edge needs roughly four times a neutral one.

    Args:
        tasks: The tasks to run.
        profile: The execution profile, which must configure Slurm.
        campaign_dir: Campaign directory; scripts and logs are written under it.
        dry_run: Whether to write the scripts without submitting them.

    Returns:
        The submission, whose job identifiers are None for a dry run.

    Raises:
        OpenFEAPIError: If there is nothing to submit, ``sbatch`` is unavailable, or a
            submission is rejected.
    """
    if not tasks:
        raise OpenFEAPIError("no tasks to submit; plan the campaign first")

    by_class: dict[str, list[Task]] = {}
    for task in tasks:
        by_class.setdefault(task.cost_class, []).append(task)

    (campaign_dir / "logs").mkdir(parents=True, exist_ok=True)
    arrays: list[ArraySubmission] = []

    for cost_class in sorted(by_class):
        groups = group_tasks(by_class[cost_class], profile.jobs_per_gpu)
        script_path = campaign_dir / f"submit_{cost_class}.sh"
        script_path.write_text(
            render_script(groups, profile, campaign_dir, cost_class), encoding="utf-8"
        )
        script_path.chmod(0o755)

        if dry_run:
            logger.info(
                "Dry run: wrote %s for %d %s task(s)",
                script_path,
                len(by_class[cost_class]),
                cost_class,
            )
            arrays.append(
                ArraySubmission(
                    cost_class=cost_class, job_id=None, script=script_path, groups=groups
                )
            )
            continue

        arrays.append(
            ArraySubmission(
                cost_class=cost_class,
                job_id=_sbatch(script_path),
                script=script_path,
                groups=groups,
            )
        )
        logger.info(
            "Submitted %s class as job %s: %d array element(s)",
            cost_class,
            arrays[-1].job_id,
            len(groups),
        )

    return SlurmSubmission(arrays=arrays)


def _sbatch(script_path: Path) -> str:
    """Submit a script and return its job identifier."""
    if shutil.which("sbatch") is None:
        raise OpenFEAPIError(
            "sbatch was not found on this machine. Run where Slurm is available, or use "
            "--dry-run to write the script and submit it yourself."
        )

    try:
        completed = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise OpenFEAPIError(f"sbatch rejected the submission: {error.stderr.strip()}") from error

    return completed.stdout.strip().split(";")[0]
