"""Running planning across a whole campaign."""

from __future__ import annotations

import json

from gufe import SmallMoleculeComponent

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.prep.report import ComplexPrepReport, SeriesPrepReport, SystemPrepReport
from openfe_api.protocols.abfe import PlannedTransformation, build_settings, plan_abfe
from openfe_api.protocols.charges import ChargeCache
from openfe_api.protocols.md import PlannedMdRun, plan_md
from openfe_api.protocols.network import PlannedNetwork, plan_network
from openfe_api.protocols.rbfe import PlannedEdge, plan_rbfe
from openfe_api.protocols.septop import (
    PlannedSepTopEdge,
    SepTopNetwork,
    build_network,
    plan_septop,
)
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = ["PlanOutcome", "plan_campaign"]

logger = get_logger(__name__)


class PlanOutcome:
    """The result of planning a campaign.

    Attributes:
        planned: Planned transformations, keyed by complex name, for ABFE.
        failures: Error messages for runs that failed, keyed by name.
        edges: Planned edges, for RBFE.
        septop_edges: Planned edges, for SepTop.
        network: The planned network, for either network protocol.
        md_runs: Planned simulations, for plain MD.
    """

    def __init__(self) -> None:
        """Initialize an empty outcome."""
        self.planned: dict[str, PlannedTransformation] = {}
        self.failures: dict[str, str] = {}
        self.edges: list[PlannedEdge] = []
        self.septop_edges: list[PlannedSepTopEdge] = []
        self.network: PlannedNetwork | SepTopNetwork | None = None
        self.md_runs: list[PlannedMdRun] = []

    @property
    def total_ns(self) -> float:
        """Simulation time implied by everything planned.

        Returns:
            Total nanoseconds across every planned transformation, edge and repeat.
        """
        return (
            sum(item.cost.total_ns for item in self.planned.values())
            + sum(edge.cost.total_ns for edge in self.edges)
            + sum(edge.cost.total_ns for edge in self.septop_edges)
            + sum(run.cost.total_ns for run in self.md_runs)
        )


def plan_campaign(
    campaign: Campaign,
    only: list[str] | None = None,
    processors: int = 1,
) -> PlanOutcome:
    """Plan the transformations a campaign will run.

    Runs that have not been prepared are skipped with a recorded failure, so planning never
    silently produces less than the request asked for.

    Args:
        campaign: The campaign to plan.
        only: Names of complexes to plan, for ABFE. If None, everything is planned.
        processors: Number of processes to use for partial charge generation.

    Returns:
        The outcome, holding planned transformations or edges, and failures.

    Raises:
        CampaignStateError: If a name in ``only`` is not part of the campaign.
        InputValidationError: If the settings preset or overrides are invalid.
        OpenFEAPIError: If ``only`` is given for a protocol without per-complex runs, or the
            campaign's protocol has no planning path.
    """
    request = campaign.manifest.request
    if only and not isinstance(request, AbfeRequest | MdRequest):
        raise OpenFEAPIError(
            f"'--only' selects complexes, which a '{request.protocol}' campaign does not "
            "have; its edges are chosen by the network planner"
        )

    match request:
        case AbfeRequest():
            return _plan_complexes(campaign, request, only, processors)
        case RbfeRequest():
            return _plan_edges(campaign, request, processors)
        case SepTopRequest():
            return _plan_septop(campaign, request, processors)
        case MdRequest():
            return _plan_md(campaign, request, only, processors)
        case _:
            raise OpenFEAPIError(
                f"planning does not handle the '{campaign.manifest.protocol}' protocol"
            )


def _plan_md(
    campaign: Campaign,
    request: MdRequest,
    only: list[str] | None,
    processors: int,
) -> PlanOutcome:
    """Write one transformation per prepared system of a plain MD campaign."""
    outcome = PlanOutcome()
    selected = request.systems
    if only is not None:
        for name in only:
            campaign.run(name)
        selected = [spec for spec in selected if spec.name in set(only)]

    for spec in selected:
        prepared_dir = campaign.directory / "prepared" / spec.name
        report_path = prepared_dir / "prep_report.json"
        record = campaign.run(spec.name)
        if record.state is RunState.PENDING or not report_path.is_file():
            message = "not prepared yet; run 'openfe-api prep' first"
            outcome.failures[spec.name] = message
            logger.error("Cannot plan '%s': %s", spec.name, message)
            continue

        try:
            report = SystemPrepReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            planned = plan_md(
                report,
                request.settings,
                campaign.directory / "plans",
                repeats=request.execution.repeats,
                solvated=spec.solvent,
                charge_cache=ChargeCache(prepared_dir / "charges"),
                processors=processors,
            )
        except (InputValidationError, OpenFEAPIError) as error:
            outcome.failures[spec.name] = str(error)
            campaign.set_state(spec.name, RunState.FAILED, message=str(error))
            logger.error("Planning failed for '%s': %s", spec.name, error)
            continue

        outcome.md_runs.append(planned)
        campaign.set_state(spec.name, RunState.PLANNED)

    return outcome


def _series_inputs(campaign: Campaign) -> tuple[SeriesPrepReport, list[SmallMoleculeComponent]]:
    """Load a prepared series' report and its placed ligands."""
    report_path = campaign.directory / "prepared" / "prep_report.json"
    if not report_path.is_file():
        raise InputValidationError(
            f"{campaign.manifest.name}: the series is not prepared yet; run 'openfe-api prep' first"
        )

    report = SeriesPrepReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    ligands = [
        SmallMoleculeComponent(
            SmallMoleculeComponent.from_sdf_file(str(placement.path)).to_rdkit(),
            name=placement.name,
        )
        for placement in report.ligands
    ]
    return report, ligands


def _register_edges(campaign: Campaign, names: list[str], repeats: int) -> None:
    """Register planned edges as runs, which only the planner knows the names of."""
    campaign.add_runs(names, repeats=repeats, state=RunState.PREPARED)
    for name in names:
        campaign.set_state(name, RunState.PLANNED)


def _plan_septop(
    campaign: Campaign,
    request: SepTopRequest,
    processors: int,
) -> PlanOutcome:
    """Plan a prepared SepTop series and write one transformation per edge."""
    outcome = PlanOutcome()
    report, ligands = _series_inputs(campaign)

    network = build_network(ligands, request.hub(), request.network)
    outcome.network = network
    _write_septop_network(campaign, network)

    outcome.septop_edges = plan_septop(
        report,
        network,
        request.settings,
        campaign.directory / "plans",
        repeats=request.execution.repeats,
        charge_cache=ChargeCache(campaign.directory / "prepared" / "charges"),
        processors=processors,
    )
    _register_edges(
        campaign,
        [edge.name for edge in outcome.septop_edges],
        request.execution.repeats,
    )
    outcome.failures.update(network.dropped)
    return outcome


def _plan_edges(
    campaign: Campaign,
    request: RbfeRequest,
    processors: int,
) -> PlanOutcome:
    """Plan the network of a prepared RBFE series and write its transformations."""
    outcome = PlanOutcome()
    report, ligands = _series_inputs(campaign)

    planned = plan_network(ligands, request.network, request.charges, n_processes=processors)
    outcome.network = planned
    _write_network(campaign, planned)

    outcome.edges = plan_rbfe(
        report,
        planned,
        request.settings,
        campaign.directory / "plans",
        repeats=request.execution.repeats,
        charge_cache=ChargeCache(campaign.directory / "prepared" / "charges"),
        processors=processors,
    )

    _register_edges(
        campaign,
        [edge.name for edge in outcome.edges],
        request.execution.repeats,
    )
    outcome.failures.update(planned.dropped)
    return outcome


def _write_septop_network(campaign: Campaign, network: SepTopNetwork) -> None:
    """Persist the planned SepTop network.

    No graphml is written: a SepTop edge carries no atom mapping, so a ligand network file
    would hold empty mappings. Replanning exactly these edges is done with
    ``network.method: explicit``.
    """
    directory = campaign.directory / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "septop",
        "hub": network.hub,
        "edges": [
            {"name": edge.name, "ligand_a": edge.ligand_a, "ligand_b": edge.ligand_b}
            for edge in network.edges
        ],
        "dropped": network.dropped,
        "unreachable": network.unreachable,
        "warnings": network.warnings,
    }
    (directory / "network.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_network(campaign: Campaign, planned: PlannedNetwork) -> None:
    """Persist the planned network so a later plan can reproduce exactly these edges."""
    directory = campaign.directory / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "network.graphml").write_text(planned.network.to_graphml(), encoding="utf-8")
    summary = {
        "edges": [
            {
                "name": decision.name,
                "ligand_a": decision.ligand_a,
                "ligand_b": decision.ligand_b,
                "charge_difference": decision.charge_difference,
                "corrected": decision.corrected,
                "score": decision.score,
            }
            for decision in planned.decisions
        ],
        "dropped": planned.dropped,
        "unreachable": planned.unreachable,
        "warnings": planned.warnings,
    }
    (directory / "network.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plan_complexes(
    campaign: Campaign,
    request: AbfeRequest,
    only: list[str] | None,
    processors: int,
) -> PlanOutcome:
    """Plan an ABFE transformation for every prepared complex."""
    settings = build_settings(request.settings)

    names = [spec.name for spec in request.complexes]
    if only is not None:
        for name in only:
            campaign.run(name)
        names = [name for name in names if name in set(only)]

    outcome = PlanOutcome()
    for name in names:
        record = campaign.run(name)
        report_path = campaign.directory / "prepared" / name / "prep_report.json"
        if record.state is RunState.PENDING or not report_path.is_file():
            message = "not prepared yet; run 'openfe-api prep' first"
            outcome.failures[name] = message
            logger.error("Cannot plan '%s': %s", name, message)
            continue

        try:
            report = ComplexPrepReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            planned = plan_abfe(
                report,
                settings,
                campaign.directory / "plans",
                repeats=request.execution.repeats,
                charge_cache=ChargeCache(campaign.directory / "prepared" / name / "charges"),
                processors=processors,
            )
        except (InputValidationError, OpenFEAPIError) as error:
            outcome.failures[name] = str(error)
            campaign.set_state(name, RunState.FAILED, message=str(error))
            logger.error("Planning failed for '%s': %s", name, error)
            continue

        outcome.planned[name] = planned
        campaign.set_state(name, RunState.PLANNED)

    return outcome
