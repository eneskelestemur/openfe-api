"""Finding the substructure a series shares with its reference ligand."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

from openfe_api.exceptions import InputValidationError

__all__ = [
    "MAX_CORE_MAPPINGS",
    "CommonCore",
    "find_core",
    "match_core",
    "murcko_scaffold",
    "shares_chemotype",
]

MCS_TIMEOUT_SECONDS = 10
"""Cap on the maximum common substructure search, which is NP-hard in general."""

MAX_CORE_MAPPINGS = 64
"""Cap on the mappings one core may match a molecule by.

Matches are not uniquified, because a symmetric core's mappings differ by whole-molecule
flips and the caller picks between them. A large symmetric core multiplies those out, so the
count is capped: twelve atoms of biphenyl already match eight ways.
"""


@dataclass(frozen=True)
class CommonCore:
    """The substructure two molecules share, with the metrics the similarity gate needs.

    Attributes:
        pattern: The core as a query molecule, matched against whole molecules including
            their hydrogens.
        smarts: SMARTS of the core, recorded in the preparation report.
        size: Heavy atoms in the core.
        coverage: Core size as a fraction of the smaller molecule's heavy atoms.
        breaks_ring: Whether the core covers part of a ring but not all of it.
        perturbed_atoms: Heavy atoms outside the core, counted across both molecules.
    """

    pattern: Chem.Mol
    smarts: str
    size: int
    coverage: float
    breaks_ring: bool
    perturbed_atoms: int


def _heavy_atoms(molecule: Chem.Mol) -> int:
    """Count a molecule's non-hydrogen atoms."""
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def _breaks_a_ring(molecule: Chem.Mol, matched: tuple[int, ...]) -> bool:
    """Report whether the matched atoms cut through a ring.

    A core holding one atom of a ring attaches to that ring; a ring-size change looks like
    this and is a routine edge. Holding two or more atoms but not the whole ring cuts it.
    """
    covered = set(matched)
    for ring in molecule.GetRingInfo().AtomRings():
        overlap = len(covered.intersection(ring))
        if 1 < overlap < len(ring):
            return True
    return False


def match_core(molecule: Chem.Mol, core: CommonCore) -> tuple[tuple[int, ...], ...]:
    """Return every way the core maps onto a molecule.

    A symmetric core matches more than one way, and the mappings place the molecule in
    different orientations, so the caller chooses rather than taking the first.

    Args:
        molecule: The molecule to match against.
        core: The core to match.

    Returns:
        Atom index tuples, each in the core's atom order, at most
        :data:`MAX_CORE_MAPPINGS` of them.
    """
    return molecule.GetSubstructMatches(
        core.pattern, uniquify=False, useChirality=False, maxMatches=MAX_CORE_MAPPINGS
    )


def find_core(
    reference: Chem.Mol,
    probe: Chem.Mol,
    core_smarts: str | None = None,
    allow_ring_break: bool = False,
) -> CommonCore:
    """Find the substructure a ligand shares with the reference ligand.

    The search runs on hydrogen-free copies and returns a heavy-atom query, so the result
    matches both a molecule built from SMILES and one read from a prepared file.

    Args:
        reference: The reference ligand.
        probe: The ligand being compared to it.
        core_smarts: An explicit core to use instead of searching for one.
        allow_ring_break: Whether the search may return a core covering part of a ring.

    Returns:
        The common core and its metrics.

    Raises:
        InputValidationError: If ``core_smarts`` cannot be parsed or does not match both
            molecules, or if no common substructure exists.
    """
    if core_smarts is not None:
        pattern = Chem.MolFromSmarts(core_smarts)
        if pattern is None:
            raise InputValidationError(
                f"'similarity.core_smarts' is not a valid SMARTS pattern: {core_smarts!r}"
            )
        for name, molecule in (("the reference ligand", reference), ("this ligand", probe)):
            if not molecule.HasSubstructMatch(pattern):
                raise InputValidationError(
                    f"the common core {core_smarts!r} does not match {name}; correct "
                    "'similarity.core_smarts' or remove it to search for a core instead"
                )
        smarts = core_smarts
    else:
        result = rdFMCS.FindMCS(
            [Chem.RemoveHs(reference), Chem.RemoveHs(probe)],
            ringMatchesRingOnly=True,
            completeRingsOnly=not allow_ring_break,
            timeout=MCS_TIMEOUT_SECONDS,
        )
        if result.numAtoms == 0:
            raise InputValidationError("no common substructure with the reference ligand")
        pattern = result.queryMol
        smarts = result.smartsString

    reference_match = reference.GetSubstructMatch(pattern)
    probe_match = probe.GetSubstructMatch(pattern)
    size = sum(1 for index in probe_match if probe.GetAtomWithIdx(index).GetAtomicNum() > 1)
    reference_heavy = _heavy_atoms(reference)
    probe_heavy = _heavy_atoms(probe)

    return CommonCore(
        pattern=pattern,
        smarts=smarts,
        size=size,
        coverage=size / min(reference_heavy, probe_heavy),
        breaks_ring=(
            _breaks_a_ring(reference, reference_match) or _breaks_a_ring(probe, probe_match)
        ),
        perturbed_atoms=(reference_heavy - size) + (probe_heavy - size),
    )


def murcko_scaffold(molecule: Chem.Mol) -> str:
    """Return a molecule's Murcko scaffold as SMILES, for grouping a series by chemotype.

    Stereochemistry is removed: two stereoisomers are the same chemotype, and keeping it
    would count an enantiomer pair as two.

    Args:
        molecule: The molecule to reduce.

    Returns:
        Canonical SMILES of the scaffold, empty for an acyclic molecule.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(molecule))
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold)


def shares_chemotype(first: str, second: str) -> bool:
    """Report whether two Murcko scaffolds describe the same chemotype.

    Adding a ring substituent changes the scaffold but is a routine edge, so containment
    rather than equality decides: one scaffold being a substructure of the other means the
    series grew a ring, while neither containing the other means the chemotypes differ.

    Args:
        first: A scaffold SMILES, as :func:`murcko_scaffold` returns.
        second: The scaffold SMILES to compare against.

    Returns:
        True if either scaffold contains the other, or either is acyclic.
    """
    if not first or not second or first == second:
        return True
    left, right = Chem.MolFromSmiles(first), Chem.MolFromSmiles(second)
    if left is None or right is None:
        return False
    return left.HasSubstructMatch(right) or right.HasSubstructMatch(left)
