"""Protein preparation with PDBFixer.

Structure models emit proteins without hydrogens, and experimental structures often miss
side chain atoms. PDBFixer completes residues and adds hydrogens at a chosen pH. Changes
that would alter the biology rather than complete it -- building missing loops, swapping
non-standard residues -- are reported instead of applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openmm import Platform
from openmm.app import ForceField, PDBFile
from pdbfixer import PDBFixer

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep.report import ProteinReport
from openfe_api.prep.structure import AMINO_ACIDS, IONS, TERMINAL_CAPS, WATERS, Structure

__all__ = [
    "PreparedProtein",
    "check_parameterizable",
    "heavy_atom_positions",
    "prepare_protein",
]

logger = get_logger(__name__)

_FORCE_FIELDS = ("amber14-all.xml", "amber14/tip3p.xml")


@dataclass
class PreparedProtein:
    """A protein written out ready for simulation.

    Attributes:
        path: The prepared PDB file.
        chains: Chain identifiers kept.
        n_atoms: Number of atoms after preparation.
        n_residues: Number of residues after preparation.
        dropped_residues: Residues removed during preparation, as descriptions.
        warnings: Warnings raised while preparing the protein.
    """

    path: Path
    chains: list[str]
    n_atoms: int
    n_residues: int
    dropped_residues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def prepare_protein(
    structure: Structure,
    output_path: Path,
    chains: list[str] | None = None,
    ph: float = 7.4,
    keep_waters: bool = False,
    exclude_residues: set[int] | None = None,
) -> PreparedProtein:
    """Extract and prepare the protein from a structure.

    Args:
        structure: The parsed structure to take the protein from.
        output_path: PDB file to write the prepared protein to.
        chains: Chain identifiers to keep. If None, every chain is considered.
        ph: pH used when adding hydrogens.
        keep_waters: Whether to keep water molecules.
        exclude_residues: Topology indices to exclude, normally the ligand and cofactor
            residues that are handled separately.

    Returns:
        A description of the prepared protein.

    Raises:
        InputValidationError: If a requested chain is absent, no protein residues remain,
            or the structure contains non-standard residues.
    """
    excluded = exclude_residues or set()
    available_chains = structure.chain_ids()

    if chains is not None:
        missing = [chain for chain in chains if chain not in available_chains]
        if missing:
            raise InputValidationError(
                f"{structure.path}: chain(s) {', '.join(missing)} not found; the file has "
                f"chains {', '.join(available_chains)}"
            )

    keep: set[int] = set()
    dropped: list[str] = []
    warnings: list[str] = []
    kept_chains: list[str] = []

    for residue in structure.residues():
        if residue.index in excluded:
            continue
        if chains is not None and residue.chain_id not in chains:
            continue

        if residue.name in AMINO_ACIDS or residue.name in TERMINAL_CAPS:
            keep.add(residue.index)
            if residue.chain_id not in kept_chains:
                kept_chains.append(residue.chain_id)
        elif residue.name in WATERS:
            if keep_waters:
                keep.add(residue.index)
            else:
                dropped.append(residue.describe())
        elif residue.name in IONS:
            keep.add(residue.index)
        else:
            dropped.append(residue.describe())
            warnings.append(
                f"{structure.path}: dropped unrecognized residue {residue.describe()}; it "
                "is not a standard amino acid, terminal cap, water or supported ion."
            )

    if not keep:
        raise InputValidationError(
            f"{structure.path}: no protein residues remain after selection"
            + (f" of chain(s) {', '.join(chains)}" if chains else "")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted = output_path.with_suffix(".extracted.pdb")
    structure.write_pdb(keep, extracted)

    try:
        # CPU, so hydrogen placement is identical on every machine; some GPU drivers cannot
        # compile the kernels at all.
        fixer = PDBFixer(filename=str(extracted), platform=Platform.getPlatformByName("CPU"))
        fixer.findMissingResidues()
        gaps = dict(fixer.missingResidues)
        if gaps:
            warnings.append(
                f"{structure.path}: the protein has {len(gaps)} chain gap(s) with missing "
                "residues. They are NOT built, because modeled loops are unreliable; the "
                "gaps remain as chain breaks. Model them beforehand if they matter."
            )
            fixer.missingResidues = {}

        fixer.findNonstandardResidues()
        if fixer.nonstandardResidues:
            listed = ", ".join(
                f"{residue.name} {residue.id}" for residue, _ in fixer.nonstandardResidues
            )
            raise InputValidationError(
                f"{structure.path}: contains non-standard residue(s): {listed}. Replacing "
                "them would change the protein, so preparation stops here. Substitute them "
                "yourself, or remove them, before running."
            )

        fixer.findMissingAtoms()
        added_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
        if added_atoms:
            warnings.append(
                f"{structure.path}: added {added_atoms} missing heavy atom(s) to existing residues."
            )
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)

        with output_path.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    finally:
        extracted.unlink(missing_ok=True)

    prepared = PDBFile(str(output_path)).topology
    logger.info(
        "Prepared protein %s: %d atoms, %d residues, chains %s",
        output_path.name,
        prepared.getNumAtoms(),
        prepared.getNumResidues(),
        ", ".join(kept_chains),
    )

    return PreparedProtein(
        path=output_path,
        chains=kept_chains,
        n_atoms=prepared.getNumAtoms(),
        n_residues=prepared.getNumResidues(),
        dropped_residues=dropped,
        warnings=warnings,
    )


def check_parameterizable(path: Path) -> None:
    """Check that a prepared protein can be parameterized by the Amber force field.

    This runs on the CPU in seconds and catches missing templates, wrong protonation and
    broken residues before any GPU time is spent.

    Args:
        path: The prepared protein PDB file.

    Raises:
        InputValidationError: If the force field cannot parameterize the protein.
    """
    pdb = PDBFile(str(path))
    force_field = ForceField(*_FORCE_FIELDS)
    try:
        force_field.createSystem(pdb.topology)
    except Exception as error:
        raise InputValidationError(
            f"{path}: the Amber force field cannot parameterize this protein: {error}"
        ) from error


def heavy_atom_positions(path: Path) -> np.ndarray:
    """Read a PDB file's heavy atom coordinates.

    Args:
        path: The PDB file to read.

    Returns:
        Coordinates in angstrom, shape ``(n, 3)``.
    """
    pdb = PDBFile(str(path))
    positions = pdb.positions.value_in_unit(pdb.positions.unit)
    return np.array(
        [
            [value * 10 for value in positions[atom.index]]
            for atom in pdb.topology.atoms()
            if atom.element is not None and atom.element.symbol != "H"
        ]
    )


def protein_report(source: Path, prepared: PreparedProtein, parameterized: bool) -> ProteinReport:
    """Describe a prepared protein for a preparation report.

    Args:
        source: The file the protein was read from.
        prepared: What preparation produced.
        parameterized: Whether the force field check ran and passed.

    Returns:
        The record.
    """
    return ProteinReport(
        source=source,
        path=prepared.path,
        chains=prepared.chains,
        n_atoms=prepared.n_atoms,
        n_residues=prepared.n_residues,
        dropped_residues=prepared.dropped_residues,
        parameterized=parameterized,
    )
