"""Placing a ligand into the reference frame.

A ligand that shares a core with the reference is placed on that core. One that shares no
core, as a scaffold hop does, is placed by Open3DAlign onto the reference ligand's shape and
chemical features instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign
from rdkit.Geometry import Point3D

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger
from openfe_api.prep.mcs import CommonCore, match_core
from openfe_api.schema.series import AlignmentSpecBase

__all__ = [
    "GeneratedPose",
    "clash_score",
    "core_rmsd",
    "generate_pose",
    "shape_pose",
    "superpose_on_core",
    "transform_conformer",
]

logger = get_logger(__name__)

CLASH_DISTANCE = 2.0
"""Heavy-atom separation in angstrom below which a contact counts as a clash."""

CORE_FORCE_CONSTANT = 1000.0
"""Restraint strength holding core atoms on the reference while the rest relaxes."""


@dataclass(frozen=True)
class GeneratedPose:
    """A ligand placed in the reference frame, with how well it was placed.

    Attributes:
        molecule: The ligand, holding one conformer in the reference frame.
        core_rmsd: RMSD in angstrom of the core atoms to the reference ligand's, when the
            pose was built on a common core.
        clashes: Heavy atoms closer than :data:`CLASH_DISTANCE` to a pocket atom.
        worst_contact: Shortest heavy-atom distance to a pocket atom, in angstrom, or None
            when no pocket coordinates were given.
        conformers_tried: Conformers generated before this one was chosen.
        conformers_accepted: Conformers that were usable; for a core placement, those whose
            core RMSD was within the limit.
        shape_score: Open3DAlign score against the reference ligand, when the pose was built
            by shape alignment. Higher is a better overlap, and the scale grows with molecule
            size, so it compares conformers of one ligand rather than different ligands.
        shape_rmsd: RMSD in angstrom of the atoms Open3DAlign matched.
    """

    molecule: Chem.Mol
    clashes: int
    worst_contact: float | None
    conformers_tried: int
    conformers_accepted: int
    core_rmsd: float | None = None
    shape_score: float | None = None
    shape_rmsd: float | None = None


def _embed(molecule: Chem.Mol, conformers: int, seed: int) -> tuple[Chem.Mol, list[int]]:
    """Return a copy of a molecule holding freely embedded conformers."""
    copied = Chem.Mol(molecule)
    conformer_ids = list(
        rdDistGeom.EmbedMultipleConfs(copied, numConfs=conformers, randomSeed=seed)
    )
    if not conformer_ids:
        raise InputValidationError(
            "could not generate any conformer of this ligand; check that the SMILES "
            "describes a chemically reasonable molecule"
        )
    return copied, conformer_ids


def clash_score(molecule: Chem.Mol, pocket: np.ndarray | None) -> tuple[int, float | None]:
    """Count a conformer's close contacts with the pocket.

    Args:
        molecule: Molecule holding exactly one conformer.
        pocket: Pocket heavy-atom coordinates, or None to skip the check.

    Returns:
        The number of clashing heavy atoms and the shortest contact distance.
    """
    if pocket is None or len(pocket) == 0:
        return 0, None

    positions = molecule.GetConformer().GetPositions()
    heavy = [index for index, atom in enumerate(molecule.GetAtoms()) if atom.GetAtomicNum() > 1]
    distances = np.linalg.norm(positions[heavy][:, None, :] - pocket[None, :, :], axis=-1)
    closest = distances.min(axis=1)
    return int((closest < CLASH_DISTANCE).sum()), float(closest.min())


def superpose_on_core(
    molecule: Chem.Mol,
    reference: Chem.Mol,
    core: CommonCore,
    conformer_id: int = 0,
) -> tuple[float, list[tuple[int, int]]]:
    """Superpose a conformer onto the reference using the best core mapping.

    A symmetric core maps more than one way and the mappings differ by whole-molecule
    flips, so every mapping is tried and the closest kept.

    Args:
        molecule: The molecule to move, holding the conformer to superpose.
        reference: The reference ligand, left untouched.
        core: The common core between them.
        conformer_id: Which conformer of ``molecule`` to superpose.

    Returns:
        The core RMSD achieved and the atom mapping that achieved it.

    Raises:
        InputValidationError: If the core does not match both molecules.
    """
    probe_matches = match_core(molecule, core)
    reference_matches = match_core(reference, core)
    if not probe_matches or not reference_matches:
        raise InputValidationError(
            f"the common core {core.smarts!r} no longer matches after building the molecule"
        )

    reference_match = reference_matches[0]
    scored = [
        (
            rdMolAlign.AlignMol(
                molecule,
                reference,
                prbCid=conformer_id,
                atomMap=list(zip(probe_match, reference_match, strict=True)),
            ),
            list(zip(probe_match, reference_match, strict=True)),
        )
        for probe_match in probe_matches
    ]
    rmsd, atom_map = min(scored, key=lambda entry: entry[0])

    # AlignMol moved the conformer on each call, so restore the best mapping's placement.
    rdMolAlign.AlignMol(molecule, reference, prbCid=conformer_id, atomMap=atom_map)
    return rmsd, atom_map


def _snap_and_relax(
    molecule: Chem.Mol,
    conformer_id: int,
    atom_map: list[tuple[int, int]],
    reference_positions: np.ndarray,
    properties: object,
) -> float:
    """Move a conformer's core onto the reference's, then relax the rest around it.

    The reference core geometry came from a real pose, so it is a valid geometry for that
    substructure; the atoms outside the core are what have to give way.
    """
    conformer = molecule.GetConformer(conformer_id)
    for probe_index, reference_index in atom_map:
        x, y, z = reference_positions[reference_index]
        conformer.SetAtomPosition(probe_index, Point3D(float(x), float(y), float(z)))

    if properties is None:
        return 0.0

    field = rdForceFieldHelpers.MMFFGetMoleculeForceField(molecule, properties, confId=conformer_id)
    for probe_index, _ in atom_map:
        field.MMFFAddPositionConstraint(probe_index, 0.0, CORE_FORCE_CONSTANT)
    field.Minimize(maxIts=1000)
    return float(field.CalcEnergy())


def _measure_core(
    molecule: Chem.Mol,
    conformer_id: int,
    atom_map: list[tuple[int, int]],
    reference_positions: np.ndarray,
) -> float:
    """Return the core RMSD of one conformer against the reference, in angstrom."""
    positions = molecule.GetConformer(conformer_id).GetPositions()
    probe = [probe_index for probe_index, _ in atom_map]
    target = [reference_index for _, reference_index in atom_map]
    return float(
        np.sqrt(((positions[probe] - reference_positions[target]) ** 2).sum(axis=1).mean())
    )


def generate_pose(
    molecule: Chem.Mol,
    reference: Chem.Mol,
    core: CommonCore,
    spec: AlignmentSpecBase,
    pocket: np.ndarray | None = None,
    seed: int = 0xF00D,
) -> GeneratedPose:
    """Build a conformer of a ligand sitting on the reference ligand's core.

    Conformers are embedded freely, then each is superposed on the core, has its core atoms
    moved onto the reference's and the rest relaxed around them. Embedding with the core
    pinned instead is over-constrained for a drug-sized ligand: on a 59 atom ligand with a 28
    atom core it took 150 seconds per attempt and returned nothing.

    Args:
        molecule: The ligand, with explicit hydrogens and no conformers needed.
        reference: The reference ligand, holding the pose that defines the frame.
        core: The common core between them.
        spec: Conformer count, acceptance limit and ranking choice.
        pocket: Pocket heavy-atom coordinates used for clash ranking.
        seed: Random seed, so a campaign replans to the same coordinates.

    Returns:
        The chosen pose and its diagnostics.

    Raises:
        InputValidationError: If embedding fails, or no conformer meets ``max_core_rmsd``.
    """
    molecule, conformer_ids = _embed(molecule, spec.n_conformers, seed)

    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(molecule)
    reference_positions = reference.GetConformer().GetPositions()

    accepted: list[tuple[float, int, int, float | None, float]] = []
    for conformer_id in conformer_ids:
        _, atom_map = superpose_on_core(molecule, reference, core, conformer_id=conformer_id)
        energy = _snap_and_relax(molecule, conformer_id, atom_map, reference_positions, properties)
        rmsd = _measure_core(molecule, conformer_id, atom_map, reference_positions)
        if rmsd > spec.max_core_rmsd:
            continue
        single = _single_conformer(molecule, conformer_id)
        clashes, worst = clash_score(single, pocket)
        accepted.append((rmsd, conformer_id, clashes, worst, energy))

    if not accepted:
        raise InputValidationError(
            f"no conformer placed the common core within {spec.max_core_rmsd} A of the "
            f"reference ligand after {len(conformer_ids)} attempts; raise "
            "'alignment.max_core_rmsd' or 'alignment.n_conformers', or check that the core "
            "is not strained in this molecule"
        )

    if spec.rank_by == "clash":
        best = min(accepted, key=lambda entry: (entry[2], entry[0]))
    else:
        best = min(accepted, key=lambda entry: (entry[4], entry[0]))

    rmsd, conformer_id, clashes, worst, _ = best
    return GeneratedPose(
        molecule=_single_conformer(molecule, conformer_id),
        clashes=clashes,
        worst_contact=worst,
        conformers_tried=len(conformer_ids),
        conformers_accepted=len(accepted),
        core_rmsd=rmsd,
    )


def shape_pose(
    molecule: Chem.Mol,
    reference: Chem.Mol,
    spec: AlignmentSpecBase,
    pocket: np.ndarray | None = None,
    seed: int = 0xF00D,
) -> GeneratedPose:
    """Build a conformer of a ligand overlaid on the reference ligand's shape.

    Open3DAlign matches atoms by their chemical features rather than by a shared
    substructure, so this places a ligand that has no common core with the reference, which
    is the case SepTop exists for. The alignment is rigid, so the conformer's own internal
    geometry is whatever embedding produced.

    Args:
        molecule: The ligand, with explicit hydrogens and no conformers needed.
        reference: The reference ligand, holding the pose that defines the frame.
        spec: Conformer count and ranking choice. ``max_core_rmsd`` does not apply, since
            there is no core to measure.
        pocket: Pocket heavy-atom coordinates used for clash ranking.
        seed: Random seed, so a campaign replans to the same coordinates.

    Returns:
        The chosen pose and its diagnostics.

    Raises:
        InputValidationError: If embedding fails, or ranking by energy was asked for and the
            molecule has no MMFF parameters.
    """
    molecule, conformer_ids = _embed(molecule, spec.n_conformers, seed)

    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(molecule)
    reference_properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(reference)
    if spec.rank_by == "energy" and properties is None:
        raise InputValidationError(
            "ranking shape-aligned conformers by energy needs MMFF parameters, which this "
            "molecule has none for; set 'alignment.rank_by: clash'"
        )

    ranked: list[tuple[float, float, int, float, int, float | None]] = []
    for conformer_id in conformer_ids:
        alignment = (
            rdMolAlign.GetO3A(
                molecule, reference, properties, reference_properties, prbCid=conformer_id
            )
            if properties is not None and reference_properties is not None
            else rdMolAlign.GetCrippenO3A(molecule, reference, prbCid=conformer_id)
        )
        rmsd = float(alignment.Align())
        score = float(alignment.Score())
        energy = (
            float(
                rdForceFieldHelpers.MMFFGetMoleculeForceField(
                    molecule, properties, confId=conformer_id
                ).CalcEnergy()
            )
            if properties is not None
            else 0.0
        )
        clashes, worst = clash_score(_single_conformer(molecule, conformer_id), pocket)
        ranked.append((score, rmsd, conformer_id, energy, clashes, worst))

    if spec.rank_by == "clash":
        best = min(ranked, key=lambda entry: (entry[4], -entry[0]))
    else:
        best = min(ranked, key=lambda entry: (entry[3], -entry[0]))

    score, rmsd, conformer_id, _, clashes, worst = best
    return GeneratedPose(
        molecule=_single_conformer(molecule, conformer_id),
        clashes=clashes,
        worst_contact=worst,
        conformers_tried=len(conformer_ids),
        conformers_accepted=len(ranked),
        shape_score=score,
        shape_rmsd=rmsd,
    )


def _single_conformer(molecule: Chem.Mol, conformer_id: int) -> Chem.Mol:
    """Return a copy of a molecule holding only one of its conformers."""
    single = Chem.Mol(molecule)
    single.RemoveAllConformers()
    conformer = Chem.Conformer(molecule.GetConformer(conformer_id))
    conformer.SetId(0)
    single.AddConformer(conformer, assignId=True)
    return single


def core_rmsd(molecule: Chem.Mol, reference: Chem.Mol, core: CommonCore) -> float:
    """Measure how far a pose's core sits from the reference's, without moving it.

    Used for a pose that must be reported on rather than re-placed, such as one supplied by
    docking or by a structure model.

    Args:
        molecule: The molecule to measure, holding one conformer.
        reference: The reference ligand.
        core: The common core between them.

    Returns:
        The smallest core RMSD in angstrom across the core's mappings.

    Raises:
        InputValidationError: If the core does not match both molecules.
    """
    probe_matches = match_core(molecule, core)
    reference_matches = match_core(reference, core)
    if not probe_matches or not reference_matches:
        raise InputValidationError(
            f"the common core {core.smarts!r} does not match the molecule being measured"
        )

    probe_positions = molecule.GetConformer().GetPositions()
    reference_positions = reference.GetConformer().GetPositions()
    reference_match = reference_matches[0]
    return min(
        float(
            np.sqrt(
                (
                    (
                        probe_positions[list(probe_match)]
                        - reference_positions[list(reference_match)]
                    )
                    ** 2
                )
                .sum(axis=1)
                .mean()
            )
        )
        for probe_match in probe_matches
    )


def transform_conformer(
    molecule: Chem.Mol, rotation: np.ndarray, translation: np.ndarray
) -> Chem.Mol:
    """Return a copy of a molecule with its conformer moved by a rigid transform.

    Args:
        molecule: The molecule to move, holding one conformer.
        rotation: 3x3 rotation applied before the translation.
        translation: Translation in angstrom.

    Returns:
        The moved copy.
    """
    moved = Chem.Mol(molecule)
    conformer = moved.GetConformer()
    positions = conformer.GetPositions() @ rotation.T + translation
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, Point3D(*(float(value) for value in position)))
    return moved
