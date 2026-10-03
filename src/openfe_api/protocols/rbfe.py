"""Planning relative binding free energy transformations.

Each edge of the network becomes two transformations: one in solvent and one in the complex.
The binding free energy change is the difference between them, so both must run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gufe import (
    ChemicalSystem,
    Component,
    LigandAtomMapping,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.settings import Settings
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.prep.report import SeriesPrepReport
from openfe_api.protocols.charges import ChargeCache, assign_charges, load_molecule
from openfe_api.protocols.network import EdgeDecision, PlannedNetwork
from openfe_api.protocols.settings import (
    CostEstimate,
    apply_overrides,
    check_selection,
)
from openfe_api.protocols.settings import (
    nanoseconds as _nanoseconds,
)
from openfe_api.schema.common import SettingsSpec

__all__ = [
    "RBFE_PRESETS",
    "PlannedEdge",
    "build_settings",
    "edge_settings",
    "estimate_edge_cost",
    "plan_rbfe",
]

logger = get_logger(__name__)

PHASES = ("solvent", "complex")
"""The two legs of every edge. Their difference is the relative binding free energy."""

RBFE_PRESETS: dict[str, dict[str, object]] = {
    "default": {},
    "screening": {
        "simulation_settings.equilibration_length": "0.25 nanosecond",
        "simulation_settings.production_length": "1 nanosecond",
    },
}
"""Named settings presets.

``default`` keeps OpenFE's simulation defaults. ``screening`` shortens them for triage.
Neither changes the lambda schedule, because the window count must keep matching the replica
count, and neither sets the repeat count, which comes only from ``execution.repeats``.
"""


@dataclass(frozen=True)
class PlannedEdge:
    """One edge written out and ready to run.

    Attributes:
        name: Edge label, ``<ligand_a>_to_<ligand_b>``.
        decision: The charge decision this edge was planned under.
        paths: The transformation JSON for each phase, keyed by phase name.
        cost: Simulation time the settings imply, across both phases.
        n_replicas: Lambda windows each phase runs.
    """

    name: str
    decision: EdgeDecision
    paths: dict[str, Path]
    cost: CostEstimate
    n_replicas: int


def build_settings(spec: SettingsSpec) -> RelativeHybridTopologyProtocolSettings:
    """Build RBFE settings from a preset and the request's overrides.

    ``protocol_repeats`` is always 1: each repeat runs as its own ``openfe quickrun``
    process, so a higher value here would multiply every one of those processes.

    Args:
        spec: The settings selection from the request.

    Returns:
        The resulting settings.

    Raises:
        InputValidationError: If the preset is unknown, an override is invalid, or the
            overrides try to set ``protocol_repeats``.
    """
    check_selection(spec, RBFE_PRESETS)
    settings = cast(
        RelativeHybridTopologyProtocolSettings,
        RelativeHybridTopologyProtocol.default_settings(),
    )
    apply_overrides(settings, RBFE_PRESETS[spec.preset])
    apply_overrides(settings, spec.overrides)
    settings.protocol_repeats = 1
    return settings


def edge_settings(
    state_a: ChemicalSystem,
    state_b: ChemicalSystem,
    mapping: LigandAtomMapping,
    base: RelativeHybridTopologyProtocolSettings,
    spec: SettingsSpec,
    corrected: bool,
) -> RelativeHybridTopologyProtocolSettings:
    """Build one edge's settings, letting OpenFE recommend what a charge change needs.

    OpenFE owns the recommendation for a charge-changing edge, currently 22 lambda windows
    sampled for 20 ns each, so it is asked rather than reproduced here. The request's own
    overrides are applied afterwards and therefore win.

    Args:
        state_a: End state A of this phase.
        state_b: End state B of this phase.
        mapping: The atom mapping between the two alchemical ligands.
        base: Settings built from the preset.
        spec: The settings selection, reapplied on top of the recommendation.
        corrected: Whether this edge applies an explicit charge correction.

    Returns:
        The settings for this edge.

    Raises:
        InputValidationError: If an override is invalid.
    """
    settings = cast(
        RelativeHybridTopologyProtocolSettings,
        RelativeHybridTopologyProtocol._adaptive_settings(
            stateA=state_a, stateB=state_b, mapping=mapping, initial_settings=base
        ),
    )
    if not corrected:
        settings.alchemical_settings.explicit_charge_correction = False

    apply_overrides(settings, spec.overrides)
    settings.protocol_repeats = 1
    return settings


def estimate_edge_cost(
    settings: RelativeHybridTopologyProtocolSettings, repeats: int
) -> CostEstimate:
    """Estimate the simulation time one edge needs, across both phases.

    Both phases share one simulation settings block, so each costs the same.

    Args:
        settings: The edge's settings.
        repeats: Repeats that will be run per phase.

    Returns:
        The estimate.
    """
    simulation = settings.simulation_settings
    windows = settings.lambda_settings.lambda_windows
    per_phase = (
        _nanoseconds(simulation.equilibration_length) + _nanoseconds(simulation.production_length)
    ) * windows
    return CostEstimate(
        complex_ns=per_phase,
        solvent_ns=per_phase,
        repeats=repeats,
        complex_replicas=windows,
        solvent_replicas=windows,
    )


def _systems(
    ligand: SmallMoleculeComponent,
    protein: ProteinComponent,
    solvent: SolventComponent,
    cofactors: list[SmallMoleculeComponent],
) -> dict[str, ChemicalSystem]:
    """Build the solvent and complex systems holding one ligand."""
    solvent_system = ChemicalSystem(
        {"ligand": ligand, "solvent": solvent}, name=f"{ligand.name}_solvent"
    )
    components: dict[str, Component] = {
        "ligand": ligand,
        "protein": protein,
        "solvent": solvent,
    }
    for index, cofactor in enumerate(cofactors):
        components[f"cofactor_{index}"] = cofactor
    complex_system = ChemicalSystem(components, name=f"{ligand.name}_complex")
    return {"solvent": solvent_system, "complex": complex_system}


def plan_rbfe(
    report: SeriesPrepReport,
    planned: PlannedNetwork,
    spec: SettingsSpec,
    output_dir: Path,
    repeats: int,
    charge_cache: ChargeCache | None = None,
    processors: int = 1,
) -> list[PlannedEdge]:
    """Write one transformation per edge and phase for a prepared series.

    Args:
        report: The series preparation report.
        planned: The network, with a decision per edge.
        spec: The settings selection from the request.
        output_dir: Directory to write transformation JSON files into.
        repeats: Repeats that will be run per phase, used for the cost estimate.
        charge_cache: Cache for partial charges. If None, charges are recomputed.
        processors: Processes to use for charge generation.

    Returns:
        One entry per edge, in network order.

    Raises:
        InputValidationError: If the prepared files are unreadable or the protocol rejects
            a system.
        OpenFEAPIError: If a transformation cannot be written.
    """
    base = build_settings(spec)
    ligands = {
        placement.name: load_molecule(placement.path, placement.name)
        for placement in report.ligands
    }
    cofactors = [load_molecule(cofactor.path, cofactor.name) for cofactor in report.cofactors]

    charged = assign_charges(
        [*ligands.values(), *cofactors],
        base.partial_charge_settings,
        cache=charge_cache,
        processors=processors,
    )
    names = list(ligands)
    ligands = dict(zip(names, charged[: len(names)], strict=True))
    cofactors = charged[len(names) :]

    try:
        protein = ProteinComponent.from_pdb_file(report.protein.path)
    except Exception as error:
        raise InputValidationError(
            f"{report.name}: prepared protein {report.protein.path} could not be loaded "
            f"as a ProteinComponent: {error}"
        ) from error

    solvent = SolventComponent()
    output_dir.mkdir(parents=True, exist_ok=True)
    edges: list[PlannedEdge] = []

    for decision in planned.decisions:
        mapping = _remapped(planned.mapping(decision), ligands)
        systems_a = _systems(ligands[decision.ligand_a], protein, solvent, cofactors)
        systems_b = _systems(ligands[decision.ligand_b], protein, solvent, cofactors)

        paths: dict[str, Path] = {}
        settings = base
        for phase in PHASES:
            state_a, state_b = systems_a[phase], systems_b[phase]
            settings = edge_settings(state_a, state_b, mapping, base, spec, decision.corrected)
            protocol = RelativeHybridTopologyProtocol(cast(Settings, settings))
            try:
                protocol.validate(stateA=state_a, stateB=state_b, mapping=mapping)
            except Exception as error:
                raise InputValidationError(
                    f"{decision.name} ({phase}): the RBFE protocol rejected this system: {error}"
                ) from error

            name = f"rbfe_{decision.name}_{phase}"
            transformation = Transformation(
                stateA=state_a,
                stateB=state_b,
                protocol=protocol,
                mapping=mapping,
                name=name,
            )
            path = output_dir / f"{name}.json"
            try:
                transformation.to_json(path)
            except Exception as error:
                raise OpenFEAPIError(f"{name}: could not write {path}: {error}") from error
            paths[phase] = path

        cost = estimate_edge_cost(settings, repeats)
        edges.append(
            PlannedEdge(
                name=decision.name,
                decision=decision,
                paths=paths,
                cost=cost,
                n_replicas=settings.simulation_settings.n_replicas,
            )
        )
        logger.info("Planned edge '%s': %s", decision.name, cost.describe())

    return edges


def _remapped(
    mapping: LigandAtomMapping, ligands: dict[str, SmallMoleculeComponent]
) -> LigandAtomMapping:
    """Rebuild a mapping against the charged copies of its two ligands.

    The network was planned before charges were assigned, so its mappings point at uncharged
    components. The atom indices are unchanged, so only the components are replaced.
    """
    return LigandAtomMapping(
        componentA=ligands[mapping.componentA.name],
        componentB=ligands[mapping.componentB.name],
        componentA_to_componentB=dict(mapping.componentA_to_componentB),
        annotations=dict(mapping.annotations),
    )
