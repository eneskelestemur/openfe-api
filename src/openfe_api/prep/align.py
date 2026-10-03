"""Superposing a structure onto the reference frame on its binding site."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openmm.unit import angstrom

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.structure import Structure

__all__ = [
    "BACKBONE_ATOMS",
    "Superposition",
    "backbone_for_residues",
    "pocket_backbone",
    "pocket_heavy_atoms",
    "superpose",
]

BACKBONE_ATOMS = ("N", "CA", "C", "O")
"""Backbone atom names used to superpose, so side chain motion does not move the frame."""


@dataclass(frozen=True)
class Superposition:
    """A rigid transform taking one structure onto another.

    Attributes:
        rmsd: Backbone RMSD in angstrom after superposing.
        n_atoms: Backbone atoms the fit used.
        rotation: 3x3 rotation applied before the translation.
        translation: Translation in angstrom.
    """

    rmsd: float
    n_atoms: int
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, positions: np.ndarray) -> np.ndarray:
        """Move coordinates into the reference frame.

        Args:
            positions: Coordinates in angstrom, shaped ``(n, 3)``.

        Returns:
            The transformed coordinates.
        """
        return positions @ self.rotation.T + self.translation


def _coordinates(structure: Structure) -> np.ndarray:
    """Return every atom's position in angstrom."""
    return np.asarray(structure.positions.value_in_unit(angstrom), dtype=float)


def pocket_heavy_atoms(
    structure: Structure, center: np.ndarray, radius: float, exclude: set[int]
) -> np.ndarray:
    """Return heavy-atom coordinates near a point, for clash scoring.

    Args:
        structure: The structure to search.
        center: Coordinates in angstrom defining the pocket, usually a ligand's atoms.
        radius: Cutoff in angstrom from any of ``center``.
        exclude: Residue indices to leave out, such as the ligand itself.

    Returns:
        Coordinates in angstrom, shaped ``(n, 3)``.
    """
    positions = _coordinates(structure)
    selected: list[int] = []
    for residue in structure.topology.residues():
        if residue.index in exclude:
            continue
        for atom in residue.atoms():
            if atom.element is not None and atom.element.symbol != "H":
                selected.append(atom.index)

    if not selected:
        return np.empty((0, 3))

    candidates = positions[selected]
    distances = np.linalg.norm(candidates[:, None, :] - center[None, :, :], axis=-1)
    return candidates[distances.min(axis=1) <= radius]


def pocket_backbone(
    structure: Structure, center: np.ndarray, radius: float
) -> dict[tuple[str, str, str], np.ndarray]:
    """Return backbone coordinates of residues lining a site, keyed for matching.

    Keying on chain, residue id and atom name lets two structures of the same protein be
    matched without assuming their atoms arrive in the same order.

    Args:
        structure: The structure to search.
        center: Coordinates in angstrom defining the site.
        radius: Cutoff in angstrom from any of ``center``.

    Returns:
        One entry per backbone atom, keyed by chain, residue id and atom name.
    """
    positions = _coordinates(structure)
    selected: dict[tuple[str, str, str], np.ndarray] = {}

    for residue in structure.topology.residues():
        indices = [atom.index for atom in residue.atoms()]
        if not indices:
            continue
        distances = np.linalg.norm(positions[indices][:, None, :] - center[None, :, :], axis=-1)
        if distances.min() > radius:
            continue
        for atom in residue.atoms():
            if atom.name in BACKBONE_ATOMS:
                key = (residue.chain.id, str(residue.id), atom.name)
                selected[key] = positions[atom.index]

    return selected


def backbone_for_residues(
    structure: Structure, residues: set[tuple[str, str]]
) -> dict[tuple[str, str, str], np.ndarray]:
    """Return backbone coordinates of named residues, wherever they sit in space.

    The mobile structure is in its own frame, so its site residues are found by identity
    rather than by distance to the reference.

    Args:
        structure: The structure to read.
        residues: Chain and residue id pairs to collect.

    Returns:
        One entry per backbone atom, keyed by chain, residue id and atom name.
    """
    positions = _coordinates(structure)
    selected: dict[tuple[str, str, str], np.ndarray] = {}
    for residue in structure.topology.residues():
        if (residue.chain.id, str(residue.id)) not in residues:
            continue
        for atom in residue.atoms():
            if atom.name in BACKBONE_ATOMS:
                selected[(residue.chain.id, str(residue.id), atom.name)] = positions[atom.index]
    return selected


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation that best map mobile onto target."""
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    correlation = (mobile - mobile_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(correlation)
    sign = np.sign(np.linalg.det(right.T @ left.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = right.T @ correction @ left.T
    return rotation, target_center - rotation @ mobile_center


def superpose(
    mobile: Structure,
    reference: Structure,
    reference_center: np.ndarray,
    radius: float,
    minimum_atoms: int = 12,
) -> Superposition:
    """Fit a structure onto the reference over the backbone atoms lining its binding site.

    The site is defined on the reference, and the same residues are then located in the
    mobile structure by chain and residue id, since it arrives in its own frame.

    Args:
        mobile: The structure to move.
        reference: The structure defining the frame.
        reference_center: Coordinates in angstrom of the reference ligand, defining the site.
        radius: Cutoff in angstrom around ``reference_center``.
        minimum_atoms: Fewest shared backbone atoms accepted for a meaningful fit.

    Returns:
        The transform, with the RMSD it achieved.

    Raises:
        InputValidationError: If the two structures share too few site backbone atoms.
    """
    target_atoms = pocket_backbone(reference, reference_center, radius)
    site_residues = {(chain, residue_id) for chain, residue_id, _ in target_atoms}
    mobile_atoms = backbone_for_residues(mobile, site_residues)
    shared = sorted(set(target_atoms).intersection(mobile_atoms))

    if len(shared) < minimum_atoms:
        raise InputValidationError(
            f"{mobile.path.name}: only {len(shared)} binding site backbone atoms match the "
            f"reference, fewer than the {minimum_atoms} needed to superpose. The two "
            "structures must be the same protein with the same residue numbering."
        )

    mobile_points = np.array([mobile_atoms[key] for key in shared])
    target_points = np.array([target_atoms[key] for key in shared])
    rotation, translation = _kabsch(mobile_points, target_points)

    moved = mobile_points @ rotation.T + translation
    rmsd = float(np.sqrt(((moved - target_points) ** 2).sum(axis=1).mean()))

    return Superposition(rmsd=rmsd, n_atoms=len(shared), rotation=rotation, translation=translation)
