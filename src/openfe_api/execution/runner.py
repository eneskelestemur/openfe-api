"""Running a campaign's tasks through the configured backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.execution.base import Task, build_tasks
from openfe_api.execution.local import run_locally
from openfe_api.execution.slurm import submit_to_slurm
from openfe_api.log import get_logger
from openfe_api.prep.relax import RelaxOutcome, prepare_relaxation, relax_tasks
from openfe_api.protocols.abfe import build_settings as abfe_settings
from openfe_api.protocols.rbfe import build_settings as rbfe_settings
from openfe_api.protocols.septop import build_settings as septop_settings
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.profiles import ExecutionProfile, ProfileLibrary
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = [
    "MPS_CONTEXT_LIMIT",
    "SubmitOutcome",
    "load_profile",
    "packing_warning",
    "submit_campaign",
]

MPS_CONTEXT_LIMIT = 48
"""Client CUDA contexts MPS allows per device, as NVIDIA documents for Volta and later."""

logger = get_logger(__name__)


@dataclass
class SubmitOutcome:
    """What a submission did.

    Attributes:
        backend: Backend that ran or queued the work.
        tasks: The tasks involved.
        job_id: Scheduler job identifier, for the Slurm backend.
        script: Generated batch script, for the Slurm backend.
        completed: Labels of tasks that finished, for the local backend.
        failed: Failure messages keyed by task label, for the local backend.
        skipped: Labels of tasks that already had results.
        interrupted: Labels of tasks stopped by an interrupt, for the local backend.
        arrays: Generated batch script per cost class, for the Slurm backend.
    """

    backend: str
    tasks: list[Task]
    job_id: str | None = None
    script: Path | None = None
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)
    arrays: dict[str, Path] = field(default_factory=dict)


def load_profile(name: str | None, profiles_file: Path | None) -> ExecutionProfile:
    """Load the execution profile a campaign asks for.

    Args:
        name: Profile name from the request. ``local`` is used if None.
        profiles_file: YAML file holding profiles. The built-in library is used if None.

    Returns:
        The profile.

    Raises:
        InputValidationError: If the file or the named profile cannot be loaded.
    """
    library = (
        ProfileLibrary.from_yaml(profiles_file)
        if profiles_file is not None
        else ProfileLibrary.default()
    )
    return library.get(name or "local")


def _replica_count(campaign: Campaign) -> int | None:
    """Return the largest lambda window count any phase of this campaign runs."""
    request = campaign.manifest.request
    try:
        match request:
            case AbfeRequest():
                abfe = abfe_settings(request.settings)
                return max(
                    abfe.complex_simulation_settings.n_replicas,
                    abfe.solvent_simulation_settings.n_replicas,
                )
            case RbfeRequest():
                return rbfe_settings(request.settings).simulation_settings.n_replicas
            case SepTopRequest():
                septop = septop_settings(request.settings)
                return max(
                    septop.complex_simulation_settings.n_replicas,
                    septop.solvent_simulation_settings.n_replicas,
                )
            case MdRequest():
                # One simulation, one context: there is no pile-up to warn about.
                return 1
            case _:
                return None
    except OpenFEAPIError:
        return None


def packing_warning(campaign: Campaign, profile: ExecutionProfile) -> str | None:
    """Warn when sharing a GPU would ask it for more contexts than it is likely to give.

    A multistate leg does not hold one GPU context. Minimization rebuilds each replica's
    state without its barostat, which drops the alchemical state that would otherwise make
    the replicas share a context, and OpenFE's context cache is unbounded, so a leg leaves
    roughly one context per lambda window behind.

    Args:
        campaign: The campaign about to run.
        profile: The execution profile that decides the packing.

    Returns:
        The warning, or None if the packing asks for nothing unusual.
    """
    if profile.jobs_per_gpu <= 1:
        return None

    replicas = _replica_count(campaign)
    if replicas is None or replicas <= 1:
        return None

    contexts = replicas * profile.jobs_per_gpu
    warning = (
        f"Profile '{profile.name}' puts {profile.jobs_per_gpu} repeats on each GPU. A leg "
        f"of this campaign has {replicas} lambda windows and leaves about one GPU context "
        f"per window alive after minimization, so the card is asked for roughly {contexts} "
        "contexts at once."
    )
    if contexts > MPS_CONTEXT_LIMIT:
        warning += (
            f" That is past the {MPS_CONTEXT_LIMIT} contexts MPS is documented to serve per "
            "device. An ABFE campaign has failed exactly this way, with 'No compatible CUDA "
            "device is available' part way into the complex leg, while two SepTop edges "
            "asking for 54 completed on a 46 GB card, so whether it is fatal depends on the "
            "device and the protocol."
        )
    warning += (
        " Measure it with scripts/diagnose_gpu_packing.py, or set 'jobs_per_gpu: 1' on the profile."
    )
    return warning


def relax_campaign(
    campaign: Campaign,
    profile: ExecutionProfile,
    check_parameters: bool = True,
    processors: int = 1,
    dry_run: bool = False,
) -> RelaxOutcome:
    """Prepare, plan and run a campaign's structure relaxations.

    A relaxation is not one of the campaign's runs, so nothing here touches the manifest's
    run states; the stage's own artifacts in ``relaxed/`` record what has happened.

    Args:
        campaign: The campaign to relax.
        profile: Execution profile selecting the backend and its resources.
        check_parameters: Whether to check that each prepared protein parameterizes.
        processors: Processes to use for partial charge generation.
        dry_run: For Slurm, write the batch script without submitting it.

    Returns:
        A record of what was prepared, skipped, and run or queued.

    Raises:
        OpenFEAPIError: If the campaign does not ask for a relaxation, or the backend fails.
        InputValidationError: If a target cannot be prepared.
    """
    targets = prepare_relaxation(campaign, check_parameters=check_parameters, processors=processors)
    outcome = RelaxOutcome(targets=[target.name for target in targets])

    tasks = relax_tasks(campaign)
    outcome.skipped = [task.run_name for task in tasks if task.is_complete]
    pending = [task for task in tasks if not task.is_complete]
    outcome.tasks = pending
    if not pending:
        logger.info("Every relaxation has already finished")
        return outcome

    if profile.backend == "slurm":
        submission = submit_to_slurm(pending, profile, campaign.directory, dry_run=dry_run)
        outcome.job_id = submission.job_id
        outcome.script = submission.script
        return outcome

    run_locally(pending, profile)
    return outcome


def submit_campaign(
    campaign: Campaign,
    profile: ExecutionProfile,
    only: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> SubmitOutcome:
    """Run or queue every planned repeat in a campaign.

    Args:
        campaign: The campaign to run.
        profile: Execution profile selecting the backend and its resources.
        only: Names of runs to submit. If None, every planned run is submitted.
        force: Whether to rerun repeats that already have results.
        dry_run: For Slurm, write the batch script without submitting it.

    Returns:
        A record of what was run or queued.

    Raises:
        OpenFEAPIError: If there is nothing to run, or the backend fails.
    """
    tasks = build_tasks(campaign, only=only)
    if not tasks:
        raise OpenFEAPIError("no planned transformations to run; run 'openfe-api plan' first")

    warning = packing_warning(campaign, profile)
    if warning is not None:
        logger.warning(warning)

    if profile.backend == "slurm":
        submission = submit_to_slurm(tasks, profile, campaign.directory, dry_run=dry_run)
        if not dry_run:
            for array in submission.arrays:
                for name in {task.run_name for task in array.tasks}:
                    campaign.set_state(name, RunState.SUBMITTED, job_id=array.job_id)
        return SubmitOutcome(
            backend="slurm",
            tasks=tasks,
            job_id=submission.job_id,
            script=submission.script,
            arrays={array.cost_class: array.script for array in submission.arrays},
        )

    for name in {task.run_name for task in tasks}:
        campaign.set_state(name, RunState.SUBMITTED)
    result = run_locally(tasks, profile, force=force)

    for name in sorted({task.run_name for task in tasks}):
        run_tasks = [task for task in tasks if task.run_name == name]
        failures = [task.label for task in run_tasks if task.label in result.failed]
        stopped = [task.label for task in run_tasks if task.label in result.interrupted]
        if stopped and not failures:
            campaign.set_state(
                name,
                RunState.FAILED,
                message=f"interrupted with {len(stopped)} repeat(s) unfinished; "
                "run submit again to resume",
            )
        elif failures:
            campaign.set_state(
                name,
                RunState.FAILED,
                message=f"{len(failures)} of {len(run_tasks)} repeat(s) failed",
            )
        elif all(task.is_complete for task in run_tasks):
            campaign.set_state(name, RunState.DONE)

    return SubmitOutcome(
        backend="local",
        tasks=tasks,
        completed=result.completed,
        failed=result.failed,
        skipped=result.skipped,
        interrupted=result.interrupted,
    )
