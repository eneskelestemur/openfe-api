"""HTTP service for running campaigns.

A thin layer over the same library the CLI uses. Preparation, planning and submission are
long-running, so they are started in the background and their progress is read back from
the campaign, which both the service and the CLI share.
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from re import fullmatch
from typing import Annotated, Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from openfe_api import __version__
from openfe_api.campaign import INPUTS_DIR, MANIFEST_NAME, Campaign
from openfe_api.config import ServiceSettings, get_settings
from openfe_api.exceptions import CampaignStateError, OpenFEAPIError
from openfe_api.execution.runner import load_profile, submit_campaign
from openfe_api.files import DEFAULT_GROUPS, archive_size, list_files, resolve_file, stream_archive
from openfe_api.log import get_logger
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.runner import plan_campaign
from openfe_api.results.qc import QualityReport
from openfe_api.results.runner import gather_campaign
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.common import MOLECULE_SUFFIXES, NAME_PATTERN, STRUCTURE_SUFFIXES
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import CampaignRequest, validate_request
from openfe_api.schema.septop import SepTopRequest

__all__ = ["app", "create_app"]

logger = get_logger(__name__)

OPERATION_FILE = "last_operation.json"
UPLOAD_SUFFIXES = STRUCTURE_SUFFIXES | MOLECULE_SUFFIXES
MAX_UPLOAD_FILES = 100
CHUNK_BYTES = 1024 * 1024

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _parse_upload_request(document: str) -> CampaignRequest:
    """Validate the JSON request that accompanies an upload.

    Args:
        document: The request, as a JSON document.

    Returns:
        The validated request.

    Raises:
        HTTPException: 422 if it is not valid JSON, or not a valid request.
    """
    try:
        data = json.loads(document)
    except ValueError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"'campaign' is not valid JSON: {error}"
        ) from error
    try:
        return validate_request(data)
    except OpenFEAPIError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error


def _check_upload_names(names: list[str], referenced: set[str]) -> None:
    """Check the uploaded filenames against the ones the request refers to.

    Args:
        names: Filenames as the client sent them.
        referenced: Input paths named by the request.

    Raises:
        HTTPException: 413 if there are too many files, 422 if a name is not a bare
            coordinate filename or the two sets differ.
    """
    if len(names) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"at most {MAX_UPLOAD_FILES} input files may be uploaded",
        )
    if len(set(names)) != len(names):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "duplicate upload filenames")

    # Both sides are checked: an uploaded name becomes a filename on disk, and a referenced
    # path would otherwise let the request read somewhere else on the server.
    for name in set(names) | referenced:
        if not fullmatch(NAME_PATTERN, name) or Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"{name!r} is not a bare coordinate filename; an upload names its own files, "
                f"with one of {sorted(UPLOAD_SUFFIXES)}",
            )
    if set(names) != referenced:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"uploaded files must match the ones the request names: uploaded {sorted(set(names))}, "
            f"referenced {sorted(referenced)}",
        )


def _store_uploads(files: list[UploadFile], names: list[str], into: Path, limit: int) -> None:
    """Write the uploaded files into a directory, streaming and counting bytes.

    Args:
        files: The uploads.
        names: Their filenames, in the same order.
        into: Directory to write them into.
        limit: Largest total size accepted.

    Raises:
        HTTPException: 413 if the total exceeds the limit, 422 if a file is empty.
    """
    total = 0
    for upload, name in zip(files, names, strict=True):
        written = 0
        with (into / name).open("xb") as destination:
            while chunk := upload.file.read(CHUNK_BYTES):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"upload exceeds the {limit} byte limit",
                    )
                written += len(chunk)
                destination.write(chunk)
        if not written:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{name!r} is empty")


def _campaign_relative(campaign: Campaign, path: Path) -> str:
    """Render an artifact path as a location inside the campaign.

    Args:
        campaign: The campaign the file belongs to.
        path: Absolute path to the file.

    Returns:
        The path relative to the campaign, or the absolute path if the file somehow sits
        outside it.
    """
    root = campaign.directory.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        return str(resolved)
    return str(PurePosixPath(resolved.relative_to(root)))


def _lock_for(name: str) -> threading.Lock:
    """Return the lock guarding one campaign's long-running operations."""
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


class OperationRecord(BaseModel):
    """The state of the last long-running operation on a campaign.

    Attributes:
        operation: Which operation ran.
        status: ``running``, ``done`` or ``failed``.
        started_at: When it started.
        finished_at: When it finished, if it has.
        message: Summary, or the error for a failed operation.
    """

    model_config = ConfigDict(extra="forbid")

    operation: str
    status: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    message: str | None = None


class RunSummary(BaseModel):
    """One run's state.

    Attributes:
        name: Complex name.
        state: Lifecycle state.
        repeats: Repeats requested.
        job_id: Scheduler job identifier, if submitted.
        message: Failure reason or status detail.
    """

    name: str
    state: str
    repeats: int
    job_id: str | None = None
    message: str | None = None


class CampaignSummary(BaseModel):
    """A campaign and what is happening to it.

    Attributes:
        name: Campaign name.
        protocol: Protocol being run.
        directory: Where the campaign lives.
        runs: The runs it holds.
        last_operation: The last long-running operation, if any.
    """

    name: str
    protocol: str
    directory: Path
    runs: list[RunSummary]
    last_operation: OperationRecord | None = None


class EdgeResultSummary(BaseModel):
    """One edge's relative binding free energy and quality verdict.

    Attributes:
        edge: Edge label.
        ligand_a: End state A ligand name.
        ligand_b: End state B ligand name.
        ddg: Relative binding free energy in kcal/mol, or None if the edge is unfinished.
        uncertainty: Uncertainty in kcal/mol.
        uncertainty_kind: How the uncertainty was obtained.
        quality: Overall verdict of the edge's quality checks.
        note: Why the edge has no result, if it has none.
    """

    edge: str
    ligand_a: str
    ligand_b: str
    ddg: float | None
    uncertainty: float | None
    uncertainty_kind: str | None
    quality: str
    note: str | None


class CycleSummary(BaseModel):
    """How far one cycle of the network fails to close.

    Attributes:
        ligands: The ligands around the cycle.
        closure: Sum of the cycle's relative free energies, in kcal/mol.
        per_edge: Closure spread over the cycle's edges, in kcal/mol.
    """

    ligands: list[str]
    closure: float
    per_edge: float


class FittedLigandSummary(BaseModel):
    """One ligand's fitted free energy.

    Attributes:
        ligand: Ligand name.
        dg_relative: Free energy in kcal/mol, relative to the network mean rather than
            absolute. A single value on its own is not meaningful.
        uncertainty: Uncertainty from the fit, in kcal/mol.
        edges: Edges connecting this ligand to the network.
    """

    ligand: str
    dg_relative: float
    uncertainty: float
    edges: int


class NetworkResultSummary(BaseModel):
    """Everything gathered from an RBFE campaign.

    Attributes:
        edges: One entry per planned edge.
        ligands: Fitted free energies for the ligands the network reaches.
        cycles: Cycle closures found in the network.
        unreachable: Ligands with no path to the rest of the network.
        checks: The network-level checks, as name, status and detail.
    """

    edges: list[EdgeResultSummary]
    ligands: list[FittedLigandSummary]
    cycles: list[CycleSummary]
    unreachable: list[str]
    checks: list[dict[str, str]]


class LigandResultSummary(BaseModel):
    """One ligand's binding free energy and quality verdict.

    Attributes:
        ligand: Ligand name.
        dg: Binding free energy in kcal/mol.
        uncertainty: Uncertainty in kcal/mol.
        uncertainty_kind: How the uncertainty was obtained.
        repeats: Repeats that contributed.
        quality: Overall verdict of the quality checks.
        checks: Each check as name, status and detail.
    """

    ligand: str
    dg: float
    uncertainty: float
    uncertainty_kind: str
    repeats: int
    quality: str
    checks: list[dict[str, str]]


class CampaignFileSummary(BaseModel):
    """One file a campaign holds.

    Attributes:
        path: Where it sits inside the campaign. Pass this to the download endpoint.
        group: Which group it belongs to.
        size: Size in bytes.
        modified: When it was last written.
    """

    path: str
    group: str
    size: int
    modified: datetime


class SimulationSummary(BaseModel):
    """One repeat of one plain MD system.

    Attributes:
        system: System name.
        repeat: One-based repeat index.
        finished: Whether the repeat finished.
        artifacts: Files the run produced, keyed by what each one is. Each value is a
            path inside the campaign, ready to pass to the download endpoint.
        note: Why the repeat is unusable, if it is.
    """

    system: str
    repeat: int
    finished: bool
    artifacts: dict[str, str]
    note: str | None = None


def _write_operation(directory: Path, record: OperationRecord) -> None:
    """Record the state of a long-running operation."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / OPERATION_FILE).write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _read_operation(directory: Path) -> OperationRecord | None:
    """Read the last operation record, if one exists."""
    path = directory / OPERATION_FILE
    if not path.is_file():
        return None
    try:
        return OperationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _open_campaign(name: str, settings: ServiceSettings) -> Campaign:
    """Open a campaign by name.

    Args:
        name: Campaign name, as it arrived in the path.
        settings: Service settings.

    Returns:
        The campaign.

    Raises:
        HTTPException: 404 if the name is not a campaign name, or names no campaign.
    """
    # Joined onto the campaigns root, so only a bare name is safe.
    if not fullmatch(NAME_PATTERN, name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no campaign named {name!r}")
    try:
        return Campaign.open(settings.campaign_dir(name))
    except CampaignStateError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


def _summarize(campaign: Campaign) -> CampaignSummary:
    """Build the summary returned for a campaign."""
    return CampaignSummary(
        name=campaign.manifest.name,
        protocol=campaign.manifest.protocol,
        directory=campaign.directory,
        runs=[
            RunSummary(
                name=record.name,
                state=record.state.value,
                repeats=record.repeats,
                job_id=record.job_id,
                message=record.message,
            )
            for record in campaign.manifest.runs.values()
        ],
        last_operation=_read_operation(campaign.directory),
    )


def _run_in_background(name: str, directory: Path, operation: str, work: Any) -> None:
    """Run one long operation, recording whether it succeeded."""
    lock = _lock_for(name)
    if not lock.acquire(blocking=False):
        logger.warning("Campaign '%s' is busy; skipping duplicate %s", name, operation)
        return

    record = OperationRecord(operation=operation, status="running")
    _write_operation(directory, record)
    try:
        message = work()
        record.status = "done"
        record.message = message
    except Exception as error:
        logger.exception("Operation %s failed for campaign '%s'", operation, name)
        record.status = "failed"
        record.message = str(error)
    finally:
        record.finished_at = datetime.now(UTC)
        _write_operation(directory, record)
        lock.release()


def get_service_settings(request: Request) -> ServiceSettings:
    """Return the settings the running application was built with.

    Args:
        request: The incoming request.

    Returns:
        The service settings held on the application state.
    """
    return request.app.state.settings


SettingsDep = Annotated[ServiceSettings, Depends(get_service_settings)]


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    """Build the service application.

    Args:
        settings: Settings to use. Read from the environment if None.

    Returns:
        The application.
    """
    application = FastAPI(
        title="openfe-api",
        version=__version__,
        summary="Run OpenFE free energy experiments on HPC and server VMs.",
    )
    application.state.settings = settings or get_settings()

    @application.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is up.

        Returns:
            The service status and version.
        """
        return {"status": "ok", "version": __version__}

    @application.get("/campaigns")
    def list_campaigns(settings: SettingsDep) -> list[str]:
        """List the campaigns the service knows about.

        Args:
            settings: Service settings.

        Returns:
            Campaign names, sorted.
        """
        if not settings.root.is_dir():
            return []
        return sorted(
            entry.name for entry in settings.root.iterdir() if (entry / "manifest.json").is_file()
        )

    @application.post("/campaigns", status_code=status.HTTP_201_CREATED)
    def create_campaign(
        request: CampaignRequest,
        settings: SettingsDep,
        force: bool = False,
    ) -> CampaignSummary:
        """Create a campaign from a request.

        Relative input paths are resolved against the service's working directory, so a
        container should be given absolute paths to its mounted data.

        Args:
            request: The campaign request.
            settings: Service settings.
            force: Whether to overwrite an existing campaign of the same name.

        Returns:
            The new campaign's summary.

        Raises:
            HTTPException: If an input file is missing, or the campaign already exists.
        """
        try:
            request.resolve_paths(Path.cwd())
        except OpenFEAPIError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

        directory = settings.campaign_dir(request.name)
        with _lock_for(request.name):
            try:
                campaign = Campaign.create(directory, request, exist_ok=force)
            except CampaignStateError as error:
                raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

        return _summarize(campaign)

    @application.post("/campaigns/upload", status_code=status.HTTP_201_CREATED)
    def upload_campaign(
        settings: SettingsDep,
        campaign: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()],
    ) -> CampaignSummary:
        """Create a campaign from uploaded input files and a JSON request.

        For a client that has no access to the service's filesystem. The request names its
        inputs by bare filename, the files are uploaded alongside it, and the campaign owns
        its own copy of them under ``inputs/``.

        Args:
            settings: Service settings.
            campaign: The campaign request, as a JSON document in a form field.
            files: The input coordinate files.

        Returns:
            The new campaign's summary.

        Raises:
            HTTPException: 422 for a bad request or filename, 413 for too much data, 409 if
                the campaign already exists.
        """
        request = _parse_upload_request(campaign)
        names = [upload.filename or "" for upload in files]
        _check_upload_names(names, {str(path) for path in request.input_paths()})

        directory = settings.campaign_dir(request.name).absolute()
        with _lock_for(request.name):
            if (directory / MANIFEST_NAME).exists():
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"campaign '{request.name}' already exists; an upload never overwrites one",
                )
            # Only what this request made is cleaned up again, so a directory that was
            # already there keeps whatever it held.
            discard = directory if not directory.exists() else directory / INPUTS_DIR
            try:
                inputs = directory / INPUTS_DIR
                inputs.mkdir(parents=True)
                _store_uploads(files, names, inputs, settings.max_upload_bytes)

                # Every protocol reports its paths the same way, so the request is repointed at
                # the campaign's own copies without knowing which protocol it is.
                for holder, attribute, _ in request.path_holders():
                    path: Path | None = getattr(holder, attribute)
                    if path is not None:
                        setattr(holder, attribute, inputs / path)
                request.resolve_paths(directory)
                created = Campaign.create(directory, request)
            except OpenFEAPIError as error:
                shutil.rmtree(discard, ignore_errors=True)
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
            except Exception:
                shutil.rmtree(discard, ignore_errors=True)
                raise

        return _summarize(created)

    @application.get("/campaigns/{name}")
    def get_campaign(name: str, settings: SettingsDep) -> CampaignSummary:
        """Report a campaign's state.

        Args:
            name: Campaign name.
            settings: Service settings.

        Returns:
            The campaign summary.
        """
        return _summarize(_open_campaign(name, settings))

    @application.post("/campaigns/{name}/prep", status_code=status.HTTP_202_ACCEPTED)
    def prep(name: str, settings: SettingsDep, background: BackgroundTasks) -> OperationRecord:
        """Start preparing a campaign's inputs.

        Args:
            name: Campaign name.
            settings: Service settings.
            background: FastAPI background task registry.

        Returns:
            A record saying the operation has started.
        """
        campaign = _open_campaign(name, settings)

        def work() -> str:
            outcome = prepare_campaign(campaign, check_parameters=settings.check_parameters)
            return f"prepared {len(outcome.reports)}, failed {len(outcome.failures)}"

        background.add_task(_run_in_background, name, campaign.directory, "prep", work)
        return OperationRecord(operation="prep", status="running")

    @application.post("/campaigns/{name}/plan", status_code=status.HTTP_202_ACCEPTED)
    def plan(name: str, settings: SettingsDep, background: BackgroundTasks) -> OperationRecord:
        """Start planning a campaign's transformations.

        Args:
            name: Campaign name.
            settings: Service settings.
            background: FastAPI background task registry.

        Returns:
            A record saying the operation has started.
        """
        campaign = _open_campaign(name, settings)

        def work() -> str:
            outcome = plan_campaign(campaign, processors=settings.processors)
            return (
                f"planned {len(outcome.planned)}, failed {len(outcome.failures)}, "
                f"{outcome.total_ns:,.0f} ns total"
            )

        background.add_task(_run_in_background, name, campaign.directory, "plan", work)
        return OperationRecord(operation="plan", status="running")

    @application.post("/campaigns/{name}/submit", status_code=status.HTTP_202_ACCEPTED)
    def submit(name: str, settings: SettingsDep, background: BackgroundTasks) -> OperationRecord:
        """Start running or queueing a campaign's repeats.

        Args:
            name: Campaign name.
            settings: Service settings.
            background: FastAPI background task registry.

        Returns:
            A record saying the operation has started.

        Raises:
            HTTPException: If the execution profile cannot be loaded.
        """
        campaign = _open_campaign(name, settings)
        try:
            profile = load_profile(
                campaign.manifest.request.execution.profile, settings.profiles_file
            )
        except OpenFEAPIError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

        def work() -> str:
            outcome = submit_campaign(campaign, profile)
            if outcome.backend == "slurm":
                return f"submitted {len(outcome.tasks)} repeat(s) as job {outcome.job_id}"
            return f"completed {len(outcome.completed)}, failed {len(outcome.failed)}"

        background.add_task(_run_in_background, name, campaign.directory, "submit", work)
        return OperationRecord(operation="submit", status="running")

    @application.get("/campaigns/{name}/results")
    def results(name: str, settings: SettingsDep) -> list[LigandResultSummary]:
        """Gather a campaign's finished results.

        Args:
            name: Campaign name.
            settings: Service settings.

        Returns:
            One entry per ligand with a usable result.
        """
        campaign = _open_campaign(name, settings)
        if isinstance(campaign.manifest.request, MdRequest):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"campaign '{name}' runs plain MD, which produces no free energy. Use "
                f"/campaigns/{name}/simulations instead.",
            )
        if not isinstance(campaign.manifest.request, AbfeRequest):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"campaign '{name}' runs the '{campaign.manifest.protocol}' protocol, whose "
                "results are relative rather than absolute binding free energies. Use "
                f"/campaigns/{name}/network instead.",
            )

        outcome = gather_campaign(campaign)
        return [
            LigandResultSummary(
                ligand=result.name,
                dg=result.dg,
                uncertainty=result.uncertainty,
                uncertainty_kind=result.uncertainty_kind,
                repeats=len(result.repeats),
                quality=outcome.reports[result.name].verdict,
                checks=[
                    {"name": check.name, "status": check.status, "detail": check.detail}
                    for check in outcome.reports[result.name].checks
                ],
            )
            for result in outcome.results
        ]

    @application.get("/campaigns/{name}/simulations")
    def simulations(name: str, settings: SettingsDep) -> list[SimulationSummary]:
        """Collect what a plain MD campaign produced.

        Args:
            name: Campaign name.
            settings: Service settings.

        Returns:
            One entry per repeat that wrote a result file.

        Raises:
            HTTPException: If the campaign does not run plain MD.
        """
        campaign = _open_campaign(name, settings)
        if not isinstance(campaign.manifest.request, MdRequest):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"campaign '{name}' runs the '{campaign.manifest.protocol}' protocol, not "
                f"plain MD. Use /campaigns/{name}/results or /campaigns/{name}/network.",
            )

        outcome = gather_campaign(campaign)
        return [
            SimulationSummary(
                system=run.name,
                repeat=entry.repeat,
                finished=entry.ok,
                artifacts={
                    label: _campaign_relative(campaign, path)
                    for label, path in entry.artifacts.items()
                },
                note=entry.failure,
            )
            for run in outcome.md
            for entry in run.repeats
        ]

    @application.get("/campaigns/{name}/files")
    def list_campaign_files(
        name: str,
        settings: SettingsDep,
        group: Annotated[list[str] | None, Query()] = None,
    ) -> list[CampaignFileSummary]:
        """List the files a campaign holds.

        Args:
            name: Campaign name.
            settings: Service settings.
            group: Groups to list. Every group when omitted. Repeat it for several.

        Returns:
            One entry per file, sorted by path.

        Raises:
            HTTPException: 404 if the campaign is unknown, 422 for an unknown group.
        """
        campaign = _open_campaign(name, settings)
        try:
            found = list_files(campaign, group)
        except OpenFEAPIError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        return [
            CampaignFileSummary(
                path=entry.path, group=entry.group, size=entry.size, modified=entry.modified
            )
            for entry in found
        ]

    @application.get("/campaigns/{name}/files/{path:path}")
    def download_campaign_file(name: str, path: str, settings: SettingsDep) -> FileResponse:
        """Download one file from a campaign.

        Ranged requests work, so a large trajectory resumes rather than starting over.

        Args:
            name: Campaign name.
            path: The file's path inside the campaign, as ``/files`` reports it.
            settings: Service settings.

        Returns:
            The file.

        Raises:
            HTTPException: 404 if the campaign is unknown, 422 if the path is not a file
                inside it.
        """
        campaign = _open_campaign(name, settings)
        try:
            resolved = resolve_file(campaign, path)
        except OpenFEAPIError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
        return FileResponse(resolved, filename=resolved.name)

    @application.get("/campaigns/{name}/archive")
    def download_campaign_archive(
        name: str,
        settings: SettingsDep,
        group: Annotated[list[str] | None, Query()] = None,
        compress: bool = False,
    ) -> StreamingResponse:
        """Download a campaign as one tar.

        Simulation output under ``runs`` is left out unless asked for: a production campaign
        keeps gigabytes there, and pulling those files one at a time resumes on a dropped
        connection while a single archive does not.

        Args:
            name: Campaign name.
            settings: Service settings.
            group: Groups to include. Everything but ``runs`` when omitted.
            compress: Whether to gzip the stream. Off by default, because simulation output
                is already dense binary and gzip costs far more time than it saves.

        Returns:
            The tar, streamed.

        Raises:
            HTTPException: 404 if the campaign is unknown, 422 for an unknown group.
        """
        campaign = _open_campaign(name, settings)
        try:
            found = list_files(campaign, group or list(DEFAULT_GROUPS))
        except OpenFEAPIError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

        suffix = "tar.gz" if compress else "tar"
        headers = {"content-disposition": f'attachment; filename="{name}.{suffix}"'}
        if not compress:
            # Known ahead of time for a plain tar, so a client can show progress and notice
            # a truncated download.
            headers["content-length"] = str(archive_size(found))

        return StreamingResponse(
            stream_archive(campaign, found, compress=compress),
            media_type="application/gzip" if compress else "application/x-tar",
            headers=headers,
        )

    @application.get("/campaigns/{name}/network")
    def network(name: str, settings: SettingsDep) -> NetworkResultSummary:
        """Gather an RBFE campaign's edges, fitted ligand values and cycle closures.

        Args:
            name: Campaign name.
            settings: Service settings.

        Returns:
            The gathered network.

        Raises:
            HTTPException: If the campaign does not run a network protocol, or has no plan.
        """
        campaign = _open_campaign(name, settings)
        if not isinstance(campaign.manifest.request, RbfeRequest | SepTopRequest):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"campaign '{name}' runs the '{campaign.manifest.protocol}' protocol, which "
                f"has no network. Use /campaigns/{name}/results instead.",
            )

        outcome = gather_campaign(campaign)
        results = outcome.network
        if results is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"campaign '{name}' has no planned network yet; run plan first",
            )

        return NetworkResultSummary(
            edges=[
                EdgeResultSummary(
                    edge=edge.name,
                    ligand_a=edge.ligand_a,
                    ligand_b=edge.ligand_b,
                    ddg=edge.ddg,
                    uncertainty=edge.uncertainty,
                    uncertainty_kind=edge.uncertainty_kind,
                    quality=(
                        outcome.reports[edge.name].verdict
                        if edge.name in outcome.reports
                        else "unknown"
                    ),
                    note=edge.failure,
                )
                for edge in results.edges
            ],
            ligands=[
                FittedLigandSummary(
                    ligand=ligand.name,
                    dg_relative=ligand.dg,
                    uncertainty=ligand.uncertainty,
                    edges=ligand.degree,
                )
                for ligand in results.ligands
            ],
            cycles=[
                CycleSummary(
                    ligands=list(cycle.ligands),
                    closure=cycle.closure,
                    per_edge=cycle.per_edge,
                )
                for cycle in results.cycles
            ],
            unreachable=list(results.unreachable),
            checks=[
                {"name": check.name, "status": check.status, "detail": check.detail}
                for check in outcome.reports.get("network", QualityReport(ligand="network")).checks
            ],
        )

    return application


app = create_app()
