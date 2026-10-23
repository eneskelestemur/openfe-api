"""Which pocket residues a ligand touches, and how often.

prolif's own runner pickles the trajectory to parallelize, which a multistate reader cannot
survive, so the frames are fingerprinted one at a time here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import MDAnalysis as mda
from prolif.fingerprint import Fingerprint
from prolif.molecule import Molecule

from openfe_api.analysis.trajectory import LIGAND_SELECTION, TrajectorySource, open_trajectory
from openfe_api.log import get_logger

__all__ = ["NOISY_INTERACTION", "POCKET_RADIUS", "ResidueContact", "measure_interactions"]

logger = get_logger(__name__)

POCKET_RADIUS = 8.0
"""How far from the ligand a residue may sit and still be fingerprinted, in angstrom."""

NOISY_INTERACTION = "VdWContact"
"""Fires on almost every nearby residue, so it is kept out of the figure."""


@dataclass(frozen=True)
class ResidueContact:
    """How often one residue made one kind of contact.

    Attributes:
        run: Run name.
        repeat: One-based repeat index.
        ligand: Which ligand was followed.
        residue: The pocket residue, as prolif names it.
        interaction: The kind of contact.
        occupancy: Fraction of frames the contact was present, from 0 to 1.
    """

    run: str
    repeat: int
    ligand: str
    residue: str
    interaction: str
    occupancy: float


def measure_interactions(source: TrajectorySource, run: str, repeat: int) -> list[ResidueContact]:
    """Fingerprint the contacts a ligand makes across one trajectory.

    Args:
        source: The trajectory and its topology.
        run: Run name, for the record.
        repeat: One-based repeat index, for the record.

    Returns:
        One entry per residue and interaction kind that occurred at least once. Empty when
        the system has no protein to contact.
    """
    universe = open_trajectory(source)
    try:
        return _fingerprint(universe, run, repeat, source.trajectory.name)
    finally:
        universe.trajectory.close()


def _conversion(atoms: mda.AtomGroup, source_name: str) -> dict[str, Any]:
    """Settle once how a selection converts for prolif, since every frame repeats it.

    A relative hybrid topology fuses both ligands of an edge onto shared atoms, so bond
    orders cannot be inferred from it. Skipping that inference keeps the contacts
    measurable, at the cost of the typing that distinguishes a donor from an acceptor.
    """
    try:
        Molecule.from_mda(atoms)
    except Exception:
        logger.warning(
            "Bond orders could not be inferred for %s; interaction typing on it is approximate",
            source_name,
        )
        return {"inferrer": None}
    return {}


def _fingerprint(
    universe: mda.Universe, run: str, repeat: int, source_name: str
) -> list[ResidueContact]:
    """Walk the frames of an open trajectory, counting contacts."""
    ligands = universe.select_atoms(LIGAND_SELECTION)
    if not len(ligands) or not len(universe.select_atoms("protein")):
        return []

    pocket = universe.select_atoms(
        f"protein and byres around {POCKET_RADIUS} group ligand", ligand=ligands
    )
    if not len(pocket):
        return []

    # OpenFE writes CONECT records for the ligand only, so the pocket arrives without bonds
    # and every interaction would come back empty.
    pocket.guess_bonds()
    pocket_options = _conversion(pocket, source_name)

    found: list[ResidueContact] = []
    for residue in ligands.residues:
        label = (
            f"{residue.resname}{residue.resid}" if len(ligands.residues) > 1 else residue.resname
        )
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        frames = 0
        ligand_options = _conversion(residue.atoms, source_name)
        fingerprint = Fingerprint()
        # The generated value is one bit per interaction, in the fingerprint's own order.
        names = list(fingerprint.interactions)
        for _ in universe.trajectory:
            frames += 1
            contacts = fingerprint.generate(
                Molecule.from_mda(residue.atoms, **ligand_options),
                Molecule.from_mda(pocket, **pocket_options),
                metadata=False,
            )
            for (_, pocket_residue), present in contacts.items():
                for interaction, seen in zip(names, present, strict=True):
                    if seen:
                        counts[(str(pocket_residue), interaction)] += 1

        found += [
            ResidueContact(
                run=run,
                repeat=repeat,
                ligand=label,
                residue=pocket_residue,
                interaction=interaction,
                occupancy=count / frames,
            )
            for (pocket_residue, interaction), count in sorted(counts.items())
        ]
        logger.info(
            "Fingerprinted '%s' repeat %d %s: %d contact kind(s) over %d frames",
            run,
            repeat,
            label,
            len(counts),
            frames,
        )

    return found
