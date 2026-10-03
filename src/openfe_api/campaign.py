"""Campaign directories: the single source of truth for a campaign's state.

A campaign directory holds the validated request, a manifest tracking every run, and the
artifacts produced by preparation, planning and execution. The CLI and the service both
read and write the same directory, so work submitted through one is visible to the other.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from openfe_api.exceptions import CampaignStateError
from openfe_api.log import get_logger
from openfe_api.provenance import collect_provenance
from openfe_api.schema.request import CampaignRequest

__all__ = [
    "INPUTS_DIR",
    "LOGS_DIR",
    "MANIFEST_NAME",
    "PLANS_DIR",
    "PREPARED_DIR",
    "RELAXED_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "SUBDIRECTORIES",
    "Campaign",
    "CampaignManifest",
    "RunRecord",
    "RunState",
]

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

INPUTS_DIR = "inputs"
PREPARED_DIR = "prepared"
PLANS_DIR = "plans"
RUNS_DIR = "runs"
RELAXED_DIR = "relaxed"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

SUBDIRECTORIES = (
    INPUTS_DIR,
    PREPARED_DIR,
    PLANS_DIR,
    RUNS_DIR,
    RELAXED_DIR,
    RESULTS_DIR,
    LOGS_DIR,
)
"""Every directory a campaign may hold, in the order the stages fill them."""


class RunState(StrEnum):
    """Lifecycle state of a single run.

    Attributes:
        PENDING: Declared in the request, nothing done yet.
        PREPARED: Inputs validated and prepared files written.
        PLANNED: Transformation JSON written and validated by the protocol.
        SUBMITTED: Handed to an execution backend.
        RUNNING: Reported as running by the backend.
        DONE: Completed with results.
        FAILED: Failed; ``RunRecord.message`` says why.
    """

    PENDING = "pending"
    PREPARED = "prepared"
    PLANNED = "planned"
    SUBMITTED = "submitted"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.PREPARED, RunState.FAILED}),
    RunState.PREPARED: frozenset({RunState.PLANNED, RunState.FAILED}),
    RunState.PLANNED: frozenset({RunState.SUBMITTED, RunState.FAILED}),
    RunState.SUBMITTED: frozenset({RunState.RUNNING, RunState.DONE, RunState.FAILED}),
    RunState.RUNNING: frozenset({RunState.SUBMITTED, RunState.DONE, RunState.FAILED}),
    RunState.FAILED: frozenset({RunState.PREPARED, RunState.PLANNED, RunState.SUBMITTED}),
    RunState.DONE: frozenset(),
}


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class RunRecord(BaseModel):
    """State of one run within a campaign: a complex for ABFE, an edge for RBFE.

    Attributes:
        name: Run name. A complex name for ABFE, or ``<ligand_a>_to_<ligand_b>`` for RBFE.
        state: Current lifecycle state.
        repeats: Number of independent repeats requested.
        job_id: Backend job identifier, once submitted.
        message: Failure reason or other status detail.
        updated_at: When the record last changed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    state: RunState = RunState.PENDING
    repeats: int = Field(ge=1)
    job_id: str | None = None
    message: str | None = None
    updated_at: datetime = Field(default_factory=_utcnow)


class CampaignManifest(BaseModel):
    """Serialized state of a campaign directory.

    Attributes:
        schema_version: Manifest format version.
        name: Campaign name.
        protocol: Protocol being run.
        created_at: When the campaign was created.
        updated_at: When the manifest last changed.
        provenance: Tool versions recorded at creation time.
        request: The validated request the campaign was created from.
        runs: Run records keyed by complex name.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    name: str
    protocol: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: dict[str, str] = Field(default_factory=dict)
    request: CampaignRequest
    runs: dict[str, RunRecord] = Field(default_factory=dict)


class Campaign:
    """A campaign directory on disk.

    Attributes:
        directory: Root directory of the campaign.
        manifest: In-memory manifest; call :meth:`save` to persist changes.
    """

    def __init__(self, directory: Path, manifest: CampaignManifest) -> None:
        """Initialize a campaign from an already-loaded manifest.

        Use :meth:`create` or :meth:`open` instead of calling this directly.

        Args:
            directory: Root directory of the campaign.
            manifest: The campaign's manifest.
        """
        # Absolute, because a batch script runs from whatever directory the scheduler chose.
        self.directory = directory.resolve()
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        directory: Path,
        request: CampaignRequest,
        exist_ok: bool = False,
    ) -> Campaign:
        """Create a campaign directory from a validated request.

        Args:
            directory: Directory to create. Parent directories are created as needed.
            request: The validated request.
            exist_ok: Whether to overwrite an existing manifest in ``directory``.

        Returns:
            The new campaign.

        Raises:
            CampaignStateError: If a manifest already exists and ``exist_ok`` is False.
        """
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists() and not exist_ok:
            raise CampaignStateError(
                f"a campaign already exists at {directory}; pass exist_ok to overwrite it"
            )

        for subdirectory in (PREPARED_DIR, PLANS_DIR, RUNS_DIR, LOGS_DIR):
            (directory / subdirectory).mkdir(parents=True, exist_ok=True)

        manifest = CampaignManifest(
            name=request.name,
            protocol=request.protocol,
            provenance=collect_provenance(),
            request=request,
            runs={
                name: RunRecord(name=name, repeats=request.execution.repeats)
                for name in request.initial_run_names()
            },
        )
        campaign = cls(directory, manifest)
        campaign.save()
        logger.info("Created campaign '%s' at %s", manifest.name, directory)
        return campaign

    @classmethod
    def open(cls, directory: Path) -> Campaign:
        """Open an existing campaign directory.

        Args:
            directory: Root directory of the campaign.

        Returns:
            The loaded campaign.

        Raises:
            CampaignStateError: If the manifest is missing, unreadable, or written by an
                incompatible schema version.
        """
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise CampaignStateError(f"no campaign manifest found at {manifest_path}")

        try:
            manifest = CampaignManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise CampaignStateError(f"{manifest_path}: unreadable manifest: {error}") from error

        if manifest.schema_version != SCHEMA_VERSION:
            raise CampaignStateError(
                f"{manifest_path}: manifest schema version {manifest.schema_version} is not "
                f"supported by this version of openfe-api (expected {SCHEMA_VERSION})"
            )

        return cls(directory, manifest)

    def save(self) -> None:
        """Write the manifest to disk atomically, under an exclusive lock."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest.updated_at = _utcnow()
        payload = self.manifest.model_dump_json(indent=2)

        with self._lock():
            temporary = self.directory / f"{MANIFEST_NAME}.tmp"
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.directory / MANIFEST_NAME)

    def add_runs(self, names: list[str], repeats: int, state: RunState = RunState.PENDING) -> None:
        """Register runs discovered during planning, keeping any that already exist.

        An RBFE campaign's runs are its edges, which are only known once the network has been
        planned, so the planner adds them rather than :meth:`create`. Their inputs were
        prepared before the network existed, so they start at ``state``.

        Args:
            names: Run names to register.
            repeats: Repeats each new run will execute.
            state: State to give a newly registered run.
        """
        for name in names:
            if name not in self.manifest.runs:
                self.manifest.runs[name] = RunRecord(name=name, repeats=repeats, state=state)
        self.save()

    def run(self, name: str) -> RunRecord:
        """Return the record for one run.

        Args:
            name: Run name.

        Returns:
            The run record.

        Raises:
            CampaignStateError: If the campaign has no run with that name.
        """
        try:
            return self.manifest.runs[name]
        except KeyError:
            known = ", ".join(sorted(self.manifest.runs)) or "<none>"
            raise CampaignStateError(
                f"no run named '{name}' in campaign '{self.manifest.name}'; known runs: {known}"
            ) from None

    def set_state(
        self,
        name: str,
        state: RunState,
        job_id: str | None = None,
        message: str | None = None,
    ) -> RunRecord:
        """Move a run to a new state and persist the change.

        Repeating the current state is allowed and only refreshes the record.

        Args:
            name: Complex name.
            state: Target state.
            job_id: Backend job identifier to record.
            message: Failure reason or status detail.

        Returns:
            The updated run record.

        Raises:
            CampaignStateError: If the run does not exist, or the transition is not allowed.
        """
        record = self.run(name)
        if state is not record.state and state not in _ALLOWED_TRANSITIONS[record.state]:
            raise CampaignStateError(
                f"run '{name}' cannot move from '{record.state.value}' to '{state.value}'"
            )

        record.state = state
        record.updated_at = _utcnow()
        if job_id is not None:
            record.job_id = job_id
        record.message = message
        self.save()
        logger.debug("Run '%s' is now '%s'", name, state.value)
        return record

    def run_directory(self, name: str) -> Path:
        """Return the working directory for a run, creating it if needed.

        Args:
            name: Complex name.

        Returns:
            Path to the run's directory under ``runs/``.

        Raises:
            CampaignStateError: If the campaign has no run with that name.
        """
        self.run(name)
        directory = self.directory / RUNS_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def counts(self) -> dict[RunState, int]:
        """Count runs by state.

        Returns:
            A mapping of every state to the number of runs in it.
        """
        counts = dict.fromkeys(RunState, 0)
        for record in self.manifest.runs.values():
            counts[record.state] += 1
        return counts

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Hold an exclusive lock on the campaign directory."""
        lock_path = self.directory / f"{MANIFEST_NAME}.lock"
        with open(lock_path, "w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
