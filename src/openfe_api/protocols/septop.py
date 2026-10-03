"""Planning separated topologies transformations.

One edge is one transformation: the protocol builds both phases itself from a single pair of
end states, each holding the protein, the solvent and one of the two ligands.

Edge choice is graph work alone, since the protocol uses no atom mapping and every pair costs
the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.settings import Settings
from openfe.protocols.openmm_septop import SepTopProtocol
from openfe.protocols.openmm_septop.equil_septop_settings import SepTopSettings

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.graph import unreachable_names
from openfe_api.log import get_logger
from openfe_api.prep.report import SeriesPrepReport
from openfe_api.protocols.charges import ChargeCache, assign_charges, load_molecule
from openfe_api.protocols.restraints import check_restraint_search
from openfe_api.protocols.settings import (
    PRESETS,
    CostEstimate,
    apply_overrides,
    check_selection,
    estimate_cost,
)
from openfe_api.schema.common import SettingsSpec
from openfe_api.schema.septop import SepTopNetworkSpec

__all__ = [
    "PlannedSepTopEdge",
    "SepTopEdge",
    "SepTopNetwork",
    "build_network",
    "build_settings",
    "plan_septop",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class SepTopEdge:
    """One pair of ligands to run.

    Attributes:
        ligand_a: Name of the end state A ligand.
        ligand_b: Name of the end state B ligand.
    """

    ligand_a: str
    ligand_b: str

    @property
    def name(self) -> str:
        """Return the edge's label, used for directories and result files.

        Returns:
            ``<ligand_a>_to_<ligand_b>``.
        """
        return f"{self.ligand_a}_to_{self.ligand_b}"


@dataclass
class SepTopNetwork:
    """The network a SepTop campaign will run.

    Attributes:
        hub: Name of the ligand every other ligand connects to, for a star network.
        edges: The edges to run, in the order they were built.
        dropped: Why each excluded ligand was excluded, keyed by name.
        unreachable: Ligands with no path to the rest of the network.
        warnings: Concerns that do not exclude anything.
    """

    hub: str
    edges: list[SepTopEdge] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    unreachable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlannedSepTopEdge:
    """One edge written out and ready to run.

    Attributes:
        name: Edge label, ``<ligand_a>_to_<ligand_b>``.
        edge: The pair this transformation runs.
        path: The transformation JSON, covering both phases.
        cost: Simulation time the settings imply, across both phases.
        complex_replicas: Lambda windows the complex phase runs.
        solvent_replicas: Lambda windows the solvent phase runs.
    """

    name: str
    edge: SepTopEdge
    path: Path
    cost: CostEstimate
    complex_replicas: int
    solvent_replicas: int


def _star_edges(hub: str, spokes: list[str]) -> list[SepTopEdge]:
    """Return one edge from the hub to every other ligand."""
    return [SepTopEdge(ligand_a=hub, ligand_b=spoke) for spoke in spokes]


def _ring_edges(spokes: list[str]) -> list[SepTopEdge]:
    """Return edges joining consecutive spokes, closing a cycle through the hub.

    Every edge of a SepTop network costs the same and no per-pair score means anything here,
    so the pairs follow the order the ligands were requested in rather than a ranking that
    would imply one.
    """
    if len(spokes) < 2:
        return []
    if len(spokes) == 2:
        return [SepTopEdge(ligand_a=spokes[0], ligand_b=spokes[1])]
    return [
        SepTopEdge(ligand_a=spokes[index], ligand_b=spokes[(index + 1) % len(spokes)])
        for index in range(len(spokes))
    ]


def build_network(
    ligands: list[SmallMoleculeComponent],
    hub: str,
    spec: SepTopNetworkSpec,
) -> SepTopNetwork:
    """Choose the edges a SepTop campaign runs.

    Every ligand is checked against the automatic Boresch restraint search first, because the
    complex phase restrains both ligands of an edge and a ligand that cannot be restrained
    cannot appear in any edge.

    Args:
        ligands: The prepared ligands, in preparation order.
        hub: Name of the hub ligand, for a star network.
        spec: The network selection from the request.

    Returns:
        The network, with the reasons for any exclusions.

    Raises:
        InputValidationError: If the hub itself cannot be restrained, or the surviving
            ligands leave no edge to run.
    """
    dropped: dict[str, str] = {}
    usable: list[str] = []
    for ligand in ligands:
        try:
            check_restraint_search(ligand, "SepTop")
        except InputValidationError as error:
            dropped[ligand.name] = str(error)
            logger.warning("Dropped '%s': %s", ligand.name, error)
            continue
        usable.append(ligand.name)

    if spec.method == "explicit":
        requested = [
            SepTopEdge(ligand_a=first, ligand_b=second) for first, second in spec.edges or []
        ]
        edges = [edge for edge in requested if edge.ligand_a in usable and edge.ligand_b in usable]
    else:
        if hub in dropped:
            raise InputValidationError(
                f"the hub ligand '{hub}' was excluded, and every edge of a '{spec.method}' "
                f"network touches it, so there is nothing to run: {dropped[hub]}"
            )
        spokes = [name for name in usable if name != hub]
        edges = _star_edges(hub, spokes)
        if spec.method == "radial_redundant":
            edges += _ring_edges(spokes)

    if not edges:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in dropped.items())
        raise InputValidationError(
            f"no edge survived planning over {len(ligands)} ligand(s)"
            + (f"; {reasons}" if reasons else "")
        )

    network = SepTopNetwork(hub=hub, edges=edges, dropped=dropped)
    network.unreachable = unreachable_names(
        usable, [(edge.ligand_a, edge.ligand_b) for edge in edges]
    )
    if network.unreachable:
        network.warnings.append(
            f"no edge connects {', '.join(network.unreachable)} to the rest of the network, "
            "so their free energies cannot be compared with the others"
        )
    logger.info(
        "Planned a '%s' SepTop network: %d edge(s) over %d ligand(s)",
        spec.method,
        len(edges),
        len(usable),
    )
    return network


def build_settings(spec: SettingsSpec) -> SepTopSettings:
    """Build SepTop settings from a preset and the request's overrides.

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
    check_selection(spec, PRESETS)
    settings = cast(SepTopSettings, SepTopProtocol.default_settings())
    apply_overrides(settings, PRESETS[spec.preset])
    apply_overrides(settings, spec.overrides)
    settings.protocol_repeats = 1
    return settings


def _system(
    ligand: SmallMoleculeComponent,
    protein: ProteinComponent,
    solvent: SolventComponent,
    cofactors: list[SmallMoleculeComponent],
) -> ChemicalSystem:
    """Build the end state holding one ligand.

    Both phases come from this one system: the protocol strips the protein and the cofactors
    for the solvent phase itself.
    """
    components: dict[str, Component] = {"ligand": ligand, "protein": protein, "solvent": solvent}
    for index, cofactor in enumerate(cofactors):
        components[f"cofactor_{index}"] = cofactor
    return ChemicalSystem(components, name=f"{ligand.name}_complex")


def plan_septop(
    report: SeriesPrepReport,
    network: SepTopNetwork,
    spec: SettingsSpec,
    output_dir: Path,
    repeats: int,
    charge_cache: ChargeCache | None = None,
    processors: int = 1,
) -> list[PlannedSepTopEdge]:
    """Write one transformation per edge for a prepared series.

    Args:
        report: The series preparation report.
        network: The network to run.
        spec: The settings selection from the request.
        output_dir: Directory to write transformation JSON files into.
        repeats: Repeats that will be run per edge, used for the cost estimate.
        charge_cache: Cache for partial charges. If None, charges are recomputed.
        processors: Processes to use for charge generation.

    Returns:
        One entry per edge, in network order.

    Raises:
        InputValidationError: If the prepared files are unreadable or the protocol rejects
            a system.
        OpenFEAPIError: If a transformation cannot be written.
    """
    settings = build_settings(spec)
    ligands = {
        placement.name: load_molecule(placement.path, placement.name)
        for placement in report.ligands
    }
    cofactors = [load_molecule(cofactor.path, cofactor.name) for cofactor in report.cofactors]

    charged = assign_charges(
        [*ligands.values(), *cofactors],
        settings.partial_charge_settings,
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
    cost = estimate_cost(settings, repeats)
    planned: list[PlannedSepTopEdge] = []

    for edge in network.edges:
        state_a = _system(ligands[edge.ligand_a], protein, solvent, cofactors)
        state_b = _system(ligands[edge.ligand_b], protein, solvent, cofactors)
        protocol = SepTopProtocol(cast(Settings, settings))
        try:
            protocol.validate(stateA=state_a, stateB=state_b, mapping=None)
        except Exception as error:
            raise InputValidationError(
                f"{edge.name}: the SepTop protocol rejected this system: {error}"
            ) from error

        name = f"septop_{edge.name}"
        transformation = Transformation(
            stateA=state_a,
            stateB=state_b,
            protocol=protocol,
            mapping=None,
            name=name,
        )
        path = output_dir / f"{name}.json"
        try:
            transformation.to_json(path)
        except Exception as error:
            raise OpenFEAPIError(f"{name}: could not write {path}: {error}") from error

        planned.append(
            PlannedSepTopEdge(
                name=edge.name,
                edge=edge,
                path=path,
                cost=cost,
                complex_replicas=settings.complex_simulation_settings.n_replicas,
                solvent_replicas=settings.solvent_simulation_settings.n_replicas,
            )
        )
        logger.info("Planned edge '%s': %s", edge.name, cost.describe())

    return planned
