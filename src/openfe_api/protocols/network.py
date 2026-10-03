"""Choosing which ligand pairs to run, and how each one handles a charge change."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from gufe import AtomMapper, LigandAtomMapping, LigandNetwork, SmallMoleculeComponent
from openfe.setup import KartografAtomMapper, LomapAtomMapper
from openfe.setup.atom_mapping import lomap_scorers
from openfe.setup.ligand_network_planning import (
    generate_minimal_redundant_network,
    generate_minimal_spanning_network,
    generate_network_from_names,
    generate_radial_network,
)
from rdkit import Chem

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.graph import unreachable_names
from openfe_api.log import get_logger
from openfe_api.schema.rbfe import ChargeSpec, NetworkSpec

__all__ = ["EdgeDecision", "PlannedNetwork", "build_mappers", "plan_network", "read_network"]

logger = get_logger(__name__)

FORBIDDEN_SCORE = 0.0
"""Score given to a pair the charge policy forbids, so the planner reaches for it last."""


@dataclass(frozen=True)
class EdgeDecision:
    """One edge of the planned network and what its charge change costs.

    Attributes:
        ligand_a: Name of the end state A ligand.
        ligand_b: Name of the end state B ligand.
        charge_difference: Net formal charge of state A minus that of state B, as
            OpenFE defines it. A positive value means the ligand loses charge.
        corrected: Whether an explicit charge correction will be applied.
        score: The mapping's Lomap score, higher being a better mapping.
    """

    ligand_a: str
    ligand_b: str
    charge_difference: int
    corrected: bool
    score: float

    @property
    def name(self) -> str:
        """Return the edge's label, used for directories and result files.

        Returns:
            ``<ligand_a>_to_<ligand_b>``.
        """
        return f"{self.ligand_a}_to_{self.ligand_b}"


@dataclass
class PlannedNetwork:
    """The network a campaign will run.

    Attributes:
        network: The ligand network, after forbidden edges were removed.
        decisions: One decision per surviving edge, in network order.
        dropped: Why each excluded pair was excluded, keyed by ``a_to_b``.
        unreachable: Ligands with no path to the rest of the network.
        warnings: Concerns that do not exclude anything.
    """

    network: LigandNetwork
    decisions: list[EdgeDecision] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    unreachable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def mapping(self, decision: EdgeDecision) -> LigandAtomMapping:
        """Return the atom mapping for one decision.

        Args:
            decision: The edge to look up.

        Returns:
            Its mapping.

        Raises:
            OpenFEAPIError: If the network holds no such edge.
        """
        for edge in self.network.edges:
            if (edge.componentA.name, edge.componentB.name) == (
                decision.ligand_a,
                decision.ligand_b,
            ):
                return edge
        raise OpenFEAPIError(f"no mapping in the network for edge '{decision.name}'")


def _lomap() -> AtomMapper:
    """Build Lomap with the settings OpenFE's own network planner uses.

    ``max3d`` is a 3D criterion, so mapped atoms must already sit within 1 A of each other;
    this is what the common-core placement during preparation provides.
    """
    return LomapAtomMapper(time=20, threed=True, max3d=1.0, element_change=True, shift=False)


def build_mappers(name: str) -> list[AtomMapper]:
    """Build the atom mappers a request asks for.

    Args:
        name: ``lomap``, ``kartograf`` or ``both``.

    Returns:
        The mappers, in the order they are offered to the planner.

    Raises:
        InputValidationError: If the name is unknown.
    """
    if name == "lomap":
        return [_lomap()]
    if name == "kartograf":
        return [KartografAtomMapper()]
    if name == "both":
        return [_lomap(), KartografAtomMapper()]
    raise InputValidationError(f"unknown atom mapper '{name}'; choose lomap, kartograf or both")


def _forbidden_pairs(
    ligands: list[SmallMoleculeComponent], charges: ChargeSpec
) -> set[frozenset[str]]:
    """Return the ligand name pairs whose net charge difference the policy refuses."""
    if charges.allow_multi:
        return set()

    forbidden: set[frozenset[str]] = set()
    for index, first in enumerate(ligands):
        for second in ligands[index + 1 :]:
            if abs(int(second.total_charge) - int(first.total_charge)) > 1:
                forbidden.add(frozenset({first.name, second.name}))
    return forbidden


def _stereoisomer_pairs(ligands: list[SmallMoleculeComponent]) -> set[frozenset[str]]:
    """Return the ligand name pairs that differ only in stereochemistry.

    A hybrid topology maps such a pair atom for atom, so the perturbation is nothing at all
    and the edge returns a free energy near zero with convincing statistics. It is the
    quietest way to get a wrong answer, and a perfect mapping score means the planner prefers
    it over every real edge.
    """
    by_skeleton: dict[str, list[str]] = {}
    for ligand in ligands:
        molecule = Chem.Mol(ligand.to_rdkit())
        Chem.RemoveStereochemistry(molecule)
        by_skeleton.setdefault(Chem.MolToSmiles(molecule), []).append(ligand.name)

    return {
        frozenset(pair)
        for names in by_skeleton.values()
        if len(names) > 1
        for pair in combinations(names, 2)
    }


def _scorer(forbidden: set[frozenset[str]]):
    """Return a Lomap scorer that steers the planner away from forbidden pairs."""

    def score(mapping: LigandAtomMapping) -> float:
        pair = frozenset({mapping.componentA.name, mapping.componentB.name})
        if pair in forbidden:
            return FORBIDDEN_SCORE
        return float(lomap_scorers.default_lomap_score(mapping))

    return score


def _generate(
    ligands: list[SmallMoleculeComponent],
    spec: NetworkSpec,
    mappers: list[AtomMapper],
    scorer,
    n_processes: int,
) -> LigandNetwork:
    """Run the planner the request names."""
    try:
        if spec.method == "explicit":
            pairs = [(first, second) for first, second in spec.edges or []]
            return generate_network_from_names(ligands, mappers[0], pairs)
        if spec.method == "radial":
            if spec.central_ligand is None:
                raise InputValidationError(
                    "'network.central_ligand' is required for a radial network"
                )
            return generate_radial_network(
                ligands,
                central_ligand=spec.central_ligand,
                mappers=mappers,
                scorer=scorer,
                n_processes=n_processes,
            )
        if spec.method == "minimal_spanning":
            return generate_minimal_spanning_network(
                ligands, mappers, scorer, progress=False, n_processes=n_processes
            )
        return generate_minimal_redundant_network(
            ligands,
            mappers,
            scorer,
            progress=False,
            mst_num=spec.redundancy,
            n_processes=n_processes,
        )
    except (ValueError, KeyError, RuntimeError) as error:
        raise InputValidationError(
            f"could not plan a '{spec.method}' network over {len(ligands)} ligands: {error}. "
            "A ligand with no edges is too distant from the rest for the mapper to connect; "
            "tighten 'similarity.min_core_fraction' to exclude it during preparation, or try "
            "'network.mapper: both'."
        ) from error


def _charge_difference(mapping: LigandAtomMapping) -> int:
    """Return the net charge change across an edge, as state A minus state B."""
    return int(mapping.get_alchemical_charge_difference())


def plan_network(
    ligands: list[SmallMoleculeComponent],
    spec: NetworkSpec,
    charges: ChargeSpec,
    n_processes: int = 1,
) -> PlannedNetwork:
    """Plan the edges of an RBFE campaign and apply the charge policy to them.

    Pairs the protocol cannot handle, whose net charge differs by more than one or which
    differ only in stereochemistry, are scored last so the planner routes around them, and
    any that still appear are removed afterwards. Ligands left without a path to the rest of
    the network are reported rather than silently producing results that cannot be compared.

    Args:
        ligands: The prepared ligands, named and charged.
        spec: Which network to build and with which mapper.
        charges: How charge-changing edges are handled.
        n_processes: Processes to use while scoring candidate mappings.

    Returns:
        The planned network, its per-edge decisions, and what was excluded.

    Raises:
        InputValidationError: If the planner fails, or no edge survives the policy.
    """
    mappers = build_mappers(spec.mapper)
    forbidden = _forbidden_pairs(ligands, charges)
    if forbidden:
        listed = ", ".join(sorted(" to ".join(sorted(pair)) for pair in forbidden))
        logger.info("Routing around %d pair(s) past the charge policy: %s", len(forbidden), listed)

    stereoisomers = _stereoisomer_pairs(ligands)
    if stereoisomers:
        listed = ", ".join(sorted(" to ".join(sorted(pair)) for pair in stereoisomers))
        logger.info("Routing around %d stereoisomer pair(s): %s", len(stereoisomers), listed)

    network = _generate(ligands, spec, mappers, _scorer(forbidden | stereoisomers), n_processes)

    kept: list[LigandAtomMapping] = []
    decisions: list[EdgeDecision] = []
    dropped: dict[str, str] = {}
    warnings: list[str] = []

    for edge in sorted(network.edges, key=lambda e: (e.componentA.name, e.componentB.name)):
        name = f"{edge.componentA.name}_to_{edge.componentB.name}"
        difference = _charge_difference(edge)

        if frozenset({edge.componentA.name, edge.componentB.name}) in stereoisomers:
            dropped[name] = (
                "its two ligands differ only in stereochemistry, which a hybrid topology "
                "cannot change: it maps the atoms onto each other, so the edge would return "
                "a free energy near zero with convincing statistics rather than failing. The "
                "SepTop protocol uses no mapping and can run this pair."
            )
            continue

        if abs(difference) > 1 and not charges.allow_multi:
            dropped[name] = (
                f"its end states differ in net charge by {difference:+d} (state A minus "
                "state B), and OpenFE has no correction for a "
                "difference greater than one. Set 'charges.allow_multi' to run it anyway, "
                "accepting that the result is uncorrected."
            )
            continue

        corrected = abs(difference) == 1 and charges.correct_single
        if abs(difference) == 1 and not corrected:
            warnings.append(
                f"{name}: its end states differ in net charge by {difference:+d} (state A "
                "minus state B) and no correction will be applied, because "
                "'charges.correct_single' is off. The result is biased."
            )
        elif corrected:
            warnings.append(
                f"{name}: its end states differ in net charge by {difference:+d} (state A "
                "minus state B), so an explicit charge correction is applied. This edge needs "
                "22 lambda windows sampled for 20 ns each, roughly four times a neutral edge."
            )
        elif abs(difference) > 1:
            warnings.append(
                f"{name}: its end states differ in net charge by {difference:+d} (state A "
                "minus state B) and it is being run UNCORRECTED because 'charges.allow_multi' "
                "is on. Treat this result as unreliable."
            )

        kept.append(edge)
        decisions.append(
            EdgeDecision(
                ligand_a=edge.componentA.name,
                ligand_b=edge.componentB.name,
                charge_difference=difference,
                corrected=corrected,
                score=float(lomap_scorers.default_lomap_score(edge)),
            )
        )

    if not kept:
        raise InputValidationError(
            f"no edge survived planning out of {len(network.edges)} candidates; the reasons "
            f"were: {'; '.join(f'{name}: {reason}' for name, reason in dropped.items())}"
        )

    final = LigandNetwork(edges=kept, nodes=network.nodes)
    unreachable = unreachable_names(
        (node.name for node in final.nodes),
        ((edge.componentA.name, edge.componentB.name) for edge in final.edges),
    )
    if unreachable:
        warning = (
            f"{len(unreachable)} ligand(s) have no path to the rest of the network and will "
            f"get no comparable free energy: {', '.join(unreachable)}. This follows from the "
            "edges that were excluded; add an allowed pair connecting them, or run them "
            "separately."
        )
        warnings.append(warning)
        logger.warning(warning)

    for warning in warnings:
        logger.info(warning)

    return PlannedNetwork(
        network=final,
        decisions=decisions,
        dropped=dropped,
        unreachable=unreachable,
        warnings=warnings,
    )


def read_network(path: Path) -> LigandNetwork:
    """Load a ligand network written by a previous plan.

    Args:
        path: The graphml file to read.

    Returns:
        The network.

    Raises:
        InputValidationError: If the file cannot be read as a ligand network.
    """
    try:
        return LigandNetwork.from_graphml(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise InputValidationError(
            f"{path} could not be read as a ligand network: {error}"
        ) from error
