"""Reading PDB and mmCIF structures and selecting molecules inside them.

Structure files from AF3 and Boltz-2 hold protein, ligand and cofactors in one mmCIF, so
preparation starts by splitting them apart. OpenMM does the parsing, which keeps chain
identifiers and residue names as written by the structure model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from openmm.app import PDBFile, PDBxFile, Topology
from openmm.unit import Quantity, angstrom

from openfe_api.exceptions import InputValidationError
from openfe_api.schema.common import Selector

__all__ = [
    "AMINO_ACIDS",
    "IONS",
    "TERMINAL_CAPS",
    "WATERS",
    "ResidueRef",
    "Structure",
    "StructureCache",
]

AMINO_ACIDS = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "HID",
        "HIE",
        "HIP",
        "CYX",
        "ASH",
        "GLH",
        "LYN",
        "MSE",
        "SEC",
        "PYL",
    ]
)
TERMINAL_CAPS = frozenset({"ACE", "NME", "NMA"})
"""Terminal capping groups, part of the protein chain in prepared structures."""
WATERS = frozenset({"HOH", "WAT", "TIP3", "SOL"})
IONS = frozenset({"NA", "CL", "K", "MG", "ZN", "CA", "FE", "MN", "CU", "CO", "NI"})

_PDBX_SUFFIXES = {".cif", ".mmcif"}
_PDB_SUFFIXES = {".pdb", ".ent"}


@dataclass(frozen=True)
class ResidueRef:
    """A residue inside a parsed structure.

    Attributes:
        index: Topology residue index, unique within the structure.
        name: Residue name, such as ``LIG_E`` or ``ALA``.
        chain_id: Chain identifier as written in the file.
        seq_id: Residue sequence identifier as written in the file.
        n_atoms: Number of atoms in the residue.
    """

    index: int
    name: str
    chain_id: str
    seq_id: str
    n_atoms: int

    def describe(self) -> str:
        """Return a human-readable identifier for the residue.

        Returns:
            A string such as ``chain E residue LIG_E 1``.
        """
        return f"chain {self.chain_id} residue {self.name} {self.seq_id}"


class Structure:
    """A parsed structure file.

    Attributes:
        path: File the structure was read from.
        topology: OpenMM topology describing chains, residues and atoms.
        positions: Atomic positions, in the same order as the topology's atoms.
    """

    def __init__(self, path: Path, topology: Topology, positions: Quantity) -> None:
        """Initialize a structure from parsed OpenMM objects.

        Use :meth:`load` instead of calling this directly.

        Args:
            path: File the structure was read from.
            topology: OpenMM topology.
            positions: Atomic positions.
        """
        self.path = path
        self.topology = topology
        self.positions = positions

    @classmethod
    def load(cls, path: Path) -> Structure:
        """Parse a PDB or mmCIF file.

        Args:
            path: Path to a ``.pdb``, ``.ent``, ``.cif`` or ``.mmcif`` file.

        Returns:
            The parsed structure.

        Raises:
            InputValidationError: If the suffix is unsupported, or the file cannot be
                parsed as the format its suffix implies.
        """
        suffix = path.suffix.lower()
        if suffix in _PDBX_SUFFIXES:
            reader = PDBxFile
        elif suffix in _PDB_SUFFIXES:
            reader = PDBFile
        else:
            supported = ", ".join(sorted(_PDBX_SUFFIXES | _PDB_SUFFIXES))
            raise InputValidationError(
                f"{path}: unsupported structure format '{suffix}'; supported: {supported}"
            )

        try:
            parsed = reader(str(path))
        except Exception as error:
            raise InputValidationError(f"{path}: could not be parsed: {error}") from error

        return cls(path, parsed.topology, parsed.positions)

    def residues(self) -> list[ResidueRef]:
        """List every residue in the structure.

        Returns:
            Residue references in topology order.
        """
        return [
            ResidueRef(
                index=residue.index,
                name=residue.name,
                chain_id=residue.chain.id,
                seq_id=str(residue.id),
                n_atoms=len(list(residue.atoms())),
            )
            for residue in self.topology.residues()
        ]

    def residue(self, index: int) -> ResidueRef:
        """Return the residue with a given topology index.

        Args:
            index: Topology residue index.

        Returns:
            The matching residue.

        Raises:
            InputValidationError: If no residue has that index.
        """
        for residue in self.residues():
            if residue.index == index:
                return residue
        raise InputValidationError(f"{self.path}: no residue with topology index {index}")

    def select_residues(self, selector: Selector) -> list[ResidueRef]:
        """Find the residues matching a selector.

        Every field set on the selector must match.

        Args:
            selector: The selector to apply.

        Returns:
            Matching residues, in topology order.
        """
        matches = []
        for residue in self.residues():
            if selector.chain is not None and residue.chain_id != selector.chain:
                continue
            if selector.resname is not None and residue.name != selector.resname:
                continue
            if selector.resid is not None and residue.seq_id != str(selector.resid):
                continue
            matches.append(residue)
        return matches

    def select_one_residue(self, selector: Selector, role: str) -> ResidueRef:
        """Find the single residue matching a selector.

        Args:
            selector: The selector to apply.
            role: What is being selected, used in error messages, e.g. ``ligand``.

        Returns:
            The matching residue.

        Raises:
            InputValidationError: If the selector matches no residue or more than one.
        """
        matches = self.select_residues(selector)
        if not matches:
            available = ", ".join(
                sorted({f"{residue.chain_id}/{residue.name}" for residue in self.residues()})
            )
            raise InputValidationError(
                f"{self.path}: no residue matches the {role} selector "
                f"({selector.describe()}); available chain/residue pairs: {available}"
            )
        if len(matches) > 1:
            listed = "; ".join(residue.describe() for residue in matches)
            raise InputValidationError(
                f"{self.path}: the {role} selector ({selector.describe()}) matches "
                f"{len(matches)} residues, but must match exactly one: {listed}. "
                "Narrow the selector with 'chain', 'resname' or 'resid'."
            )
        return matches[0]

    def find_copies(self, residue: ResidueRef) -> list[ResidueRef]:
        """Find other residues that look like copies of the given one.

        Copies are matched on element composition rather than residue name, because a
        structure model may name each copy differently: AF3 writes the two ligand copies
        of a homodimer as ``LIG_E`` and ``LIG_F``. Polymer residues, waters and ions are
        never treated as copies.

        Args:
            residue: The reference residue.

        Returns:
            Matching residues, excluding the reference itself.
        """
        reference = self._composition(residue.index)
        return [
            other
            for other in self.residues()
            if other.index != residue.index
            and other.name not in AMINO_ACIDS
            and other.name not in TERMINAL_CAPS
            and other.name not in WATERS
            and other.name not in IONS
            and self._composition(other.index) == reference
        ]

    def _composition(self, residue_index: int) -> tuple[str, ...]:
        """Return the sorted element symbols of a residue's atoms."""
        return tuple(
            sorted(
                atom.element.symbol if atom.element is not None else "?"
                for atom in self.topology.atoms()
                if atom.residue.index == residue_index
            )
        )

    def residue_pdb_block(self, residue: ResidueRef) -> str:
        """Write one residue as a PDB block.

        Args:
            residue: The residue to write.

        Returns:
            The PDB text for that residue's atoms.
        """
        return self._write_pdb({residue.index})

    def chain_ids(self) -> list[str]:
        """List the chain identifiers in the structure.

        Returns:
            Chain identifiers in topology order.
        """
        return [chain.id for chain in self.topology.chains()]

    def write_pdb(self, residue_indices: set[int], path: Path) -> None:
        """Write a subset of residues to a PDB file.

        Args:
            residue_indices: Topology indices of the residues to keep.
            path: File to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._write_pdb(residue_indices), encoding="utf-8")

    def _write_pdb(self, residue_indices: set[int]) -> str:
        """Render a subset of residues as PDB text."""
        if not residue_indices:
            raise InputValidationError(f"{self.path}: no residues selected to write")

        keep = [atom for atom in self.topology.atoms() if atom.residue.index in residue_indices]
        subset = Topology()
        chain_map: dict[int, object] = {}
        residue_map: dict[int, object] = {}
        for atom in keep:
            chain = atom.residue.chain
            if chain.index not in chain_map:
                chain_map[chain.index] = subset.addChain(chain.id)
            if atom.residue.index not in residue_map:
                residue_map[atom.residue.index] = subset.addResidue(
                    atom.residue.name, chain_map[chain.index], atom.residue.id
                )
            subset.addAtom(atom.name, atom.element, residue_map[atom.residue.index])

        coordinates = Quantity(
            [self.positions[atom.index].value_in_unit(angstrom) for atom in keep], angstrom
        )
        handle = io.StringIO()
        PDBFile.writeFile(subset, coordinates, handle, keepIds=True)
        return handle.getvalue()


class StructureCache:
    """Parses each structure file at most once."""

    def __init__(self) -> None:
        """Create an empty cache."""
        self._structures: dict[Path, Structure] = {}

    def get(self, path: Path) -> Structure:
        """Return the parsed structure for a path.

        Args:
            path: Structure file to parse.

        Returns:
            The parsed structure.

        Raises:
            InputValidationError: If the file cannot be parsed.
        """
        if path not in self._structures:
            self._structures[path] = Structure.load(path)
        return self._structures[path]
