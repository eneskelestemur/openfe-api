"""Running the analysis as a Slurm job instead of in the foreground.

Measuring the trajectories of a production campaign is hours of reading dense binary, which
belongs on a compute node. The job runs the same ``analyze`` command, so what lands in
``analysis/`` is what the foreground run would have written.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from openfe_api.campaign import LOGS_DIR, Campaign
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.slurm import batch_directives, sbatch
from openfe_api.log import get_logger
from openfe_api.schema.profiles import ExecutionProfile, SlurmProfile

__all__ = [
    "ANALYSIS_CLASS",
    "AnalysisSubmission",
    "analysis_command",
    "render_analysis_script",
    "submit_analysis",
]

logger = get_logger(__name__)

SCRIPT_NAME = "submit_analysis.sh"

ANALYSIS_CLASS = "analysis"
"""Cost class the wall time is looked up under, so a site can give analysis its own limit."""


def _slurm(profile: ExecutionProfile) -> SlurmProfile:
    """Return a profile's Slurm settings, refusing one that has none."""
    if profile.slurm is None:
        raise OpenFEAPIError(
            f"profile '{profile.name}' has no 'slurm' section, so there is nothing to submit "
            "the analysis to. Run it in the foreground, or name a Slurm profile."
        )
    return profile.slurm


@dataclass(frozen=True)
class AnalysisSubmission:
    """A submitted analysis job.

    Attributes:
        job_id: The Slurm job identifier, or None for a dry run.
        script: The generated batch script.
        cpus: Cores the job asked for.
    """

    job_id: str | None
    script: Path
    cpus: int


def analysis_command(
    campaign: Campaign,
    measured: Path | None,
    dark: bool,
    trajectories: bool,
    workers: int,
) -> list[str]:
    """Build the ``openfe-api analyze`` command line the job will run.

    Args:
        campaign: The campaign to analyze.
        measured: Table of measured free energies, if one was given.
        dark: Whether to draw for a dark background.
        trajectories: Whether to measure ligand and protein movement.
        workers: How many trajectories to measure at once.

    Returns:
        The command as a list of arguments.
    """
    command = ["openfe-api", "analyze", str(campaign.directory.resolve())]
    if measured is not None:
        command += ["--measured", str(measured.resolve())]
    if dark:
        command.append("--dark")
    if trajectories:
        command += ["--trajectories", "--workers", str(workers)]
    return command


def render_analysis_script(
    campaign: Campaign, profile: ExecutionProfile, command: list[str], cpus: int
) -> str:
    """Render the batch script for an analysis job.

    Args:
        campaign: The campaign being analyzed.
        profile: The execution profile, which must configure Slurm.
        command: The command line to run, as returned by :func:`analysis_command`.
        cpus: Cores to request.

    Returns:
        The script text.

    Raises:
        OpenFEAPIError: If the profile has no Slurm section.
    """
    slurm = _slurm(profile)
    time = slurm.time_by_class.get(ANALYSIS_CLASS, slurm.time)

    lines = [
        "#!/bin/bash",
        *batch_directives(profile, ANALYSIS_CLASS, campaign.directory, time, cpus),
        "",
        "set -euo pipefail",
        "",
        *profile.setup_commands,
        "",
        shlex.join(command),
        "",
    ]
    return "\n".join(lines)


def submit_analysis(
    campaign: Campaign,
    profile: ExecutionProfile,
    measured: Path | None = None,
    dark: bool = False,
    trajectories: bool = False,
    workers: int = 1,
    dry_run: bool = False,
) -> AnalysisSubmission:
    """Write the batch script for an analysis and submit it.

    The job asks for at least as many cores as there are workers, so the parallelism the
    command was given is the parallelism it gets.

    Args:
        campaign: The campaign to analyze.
        profile: The execution profile, which must configure Slurm.
        measured: Table of measured free energies, if one was given.
        dark: Whether to draw for a dark background.
        trajectories: Whether to measure ligand and protein movement.
        workers: How many trajectories to measure at once.
        dry_run: Whether to write the script without submitting it.

    Returns:
        The submission, whose job identifier is None for a dry run.

    Raises:
        OpenFEAPIError: If the profile has no Slurm section, or sbatch rejects the job.
    """
    cpus = max(workers, _slurm(profile).cpus_per_task)
    command = analysis_command(campaign, measured, dark, trajectories, workers)

    (campaign.directory / LOGS_DIR).mkdir(parents=True, exist_ok=True)
    script_path = campaign.directory / SCRIPT_NAME
    script_path.write_text(
        render_analysis_script(campaign, profile, command, cpus), encoding="utf-8"
    )
    script_path.chmod(0o755)

    if dry_run:
        logger.info("Dry run: wrote %s", script_path)
        return AnalysisSubmission(job_id=None, script=script_path, cpus=cpus)

    job_id = sbatch(script_path)
    logger.info("Submitted the analysis as job %s with %d core(s)", job_id, cpus)
    return AnalysisSubmission(job_id=job_id, script=script_path, cpus=cpus)
