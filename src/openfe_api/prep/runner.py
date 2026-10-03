"""Running preparation across a whole campaign."""

from __future__ import annotations

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.prep.complex import prepare_complex
from openfe_api.prep.relax import check_relaxation, relaxed_request
from openfe_api.prep.report import ComplexPrepReport, SeriesPrepReport, SystemPrepReport
from openfe_api.prep.series import prepare_series
from openfe_api.prep.system import prepare_system
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = ["PrepOutcome", "prepare_campaign"]

logger = get_logger(__name__)


class PrepOutcome:
    """The result of preparing a campaign.

    Attributes:
        reports: Successful preparation reports, keyed by complex name, for ABFE.
        failures: Error messages for runs that failed, keyed by name.
        series: The series report, for RBFE and SepTop.
        systems: Successful system reports, keyed by system name, for plain MD.
    """

    def __init__(self) -> None:
        """Initialize an empty outcome."""
        self.reports: dict[str, ComplexPrepReport] = {}
        self.failures: dict[str, str] = {}
        self.series: SeriesPrepReport | None = None
        self.systems: dict[str, SystemPrepReport] = {}

    @property
    def warnings(self) -> list[str]:
        """Every warning raised across all prepared complexes.

        Returns:
            Warnings in preparation order.
        """
        if self.series is not None:
            return list(self.series.warnings)
        return [
            warning
            for report in (*self.reports.values(), *self.systems.values())
            for warning in report.warnings
        ]


def prepare_campaign(
    campaign: Campaign,
    only: list[str] | None = None,
    check_parameters: bool = True,
) -> PrepOutcome:
    """Prepare a campaign's inputs, recording the outcome in the manifest.

    Args:
        campaign: The campaign to prepare.
        only: Names of runs to prepare, for the protocols that have named runs. If None,
            everything is prepared.
        check_parameters: Whether to check that each prepared protein parameterizes.

    Returns:
        The outcome, holding reports for successes and messages for failures.

    Raises:
        CampaignStateError: If a name in ``only`` is not part of the campaign.
        InputValidationError: If a series cannot be prepared at all, or a relaxation the
            campaign asked for has not finished.
        OpenFEAPIError: If the campaign asks for a relaxation it cannot have, if ``only`` is
            given for a protocol without per-complex runs, or the campaign's protocol has no
            preparation path.
    """
    request = campaign.manifest.request
    check_relaxation(request)
    # Swaps in the relaxed frame, so preparation rebuilds every molecule from its declared
    # SMILES against the relaxed coordinates.
    request = relaxed_request(campaign)
    match request:
        case AbfeRequest():
            return _prepare_complexes(campaign, request, only, check_parameters)
        case RbfeRequest() | SepTopRequest():
            return _prepare_series(campaign, request, only, check_parameters)
        case MdRequest():
            return _prepare_systems(campaign, request, only, check_parameters)
        case _:
            raise OpenFEAPIError(
                f"preparation does not handle the '{campaign.manifest.protocol}' protocol"
            )


def _prepare_series(
    campaign: Campaign,
    request: RbfeRequest | SepTopRequest,
    only: list[str] | None,
    check_parameters: bool,
) -> PrepOutcome:
    """Prepare a series campaign's ligands into one frame."""
    if only:
        raise OpenFEAPIError(
            f"'--only' selects complexes, which a '{request.protocol}' campaign does not "
            "have: its series is prepared as a whole, and its runs are the edges created "
            "by 'plan'"
        )

    outcome = PrepOutcome()
    outcome.series = prepare_series(
        request, campaign.directory / "prepared", check_parameters=check_parameters
    )
    campaign.save()
    return outcome


def _prepare_systems(
    campaign: Campaign,
    request: MdRequest,
    only: list[str] | None,
    check_parameters: bool,
) -> PrepOutcome:
    """Prepare every system of a plain MD campaign.

    A system that fails is recorded as failed and preparation continues with the rest, so one
    bad input does not hide problems in the others.
    """
    selected = request.systems
    if only is not None:
        for name in only:
            campaign.run(name)
        selected = [spec for spec in selected if spec.name in set(only)]

    outcome = PrepOutcome()
    for spec in selected:
        try:
            report = prepare_system(
                spec,
                campaign.directory / "prepared" / spec.name,
                check_parameters=check_parameters,
            )
        except InputValidationError as error:
            outcome.failures[spec.name] = str(error)
            campaign.set_state(spec.name, RunState.FAILED, message=str(error))
            logger.error("Preparation failed for '%s': %s", spec.name, error)
            continue

        outcome.systems[spec.name] = report
        campaign.set_state(spec.name, RunState.PREPARED)

    return outcome


def _prepare_complexes(
    campaign: Campaign,
    request: AbfeRequest,
    only: list[str] | None,
    check_parameters: bool,
) -> PrepOutcome:
    """Prepare every complex of an ABFE campaign.

    A complex that fails is recorded as failed and preparation continues with the rest, so
    one bad input does not hide problems in the others.
    """
    selected = request.complexes
    if only is not None:
        for name in only:
            campaign.run(name)
        selected = [spec for spec in selected if spec.name in set(only)]

    outcome = PrepOutcome()
    for spec in selected:
        output_dir = campaign.directory / "prepared" / spec.name
        try:
            report = prepare_complex(
                spec,
                output_dir,
                neutralize_ligands=request.neutralize_ligands,
                check_parameters=check_parameters,
            )
        except InputValidationError as error:
            outcome.failures[spec.name] = str(error)
            campaign.set_state(spec.name, RunState.FAILED, message=str(error))
            logger.error("Preparation failed for '%s': %s", spec.name, error)
            continue

        outcome.reports[spec.name] = report
        campaign.set_state(spec.name, RunState.PREPARED)

    return outcome
