"""Tests for superposing a structure onto the reference binding site."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openmm.unit import angstrom

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.align import (
    BACKBONE_ATOMS,
    backbone_for_residues,
    pocket_backbone,
    pocket_heavy_atoms,
    superpose,
)
from openfe_api.prep.structure import Structure
from openfe_api.schema.common import Selector

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture
def complex_structure() -> Structure:
    """Load the trimmed complex fixture.

    Returns:
        The parsed structure.
    """
    return Structure.load(DATA / "mini_complex.cif")


def ligand_center(structure: Structure) -> tuple[np.ndarray, int]:
    """Return the ligand's coordinates and its residue index.

    Args:
        structure: The structure holding the ligand.

    Returns:
        Coordinates in angstrom and the ligand residue's topology index.
    """
    residue = structure.select_one_residue(Selector(chain="L"), "ligand")
    positions = np.asarray(structure.positions.value_in_unit(angstrom), dtype=float)
    indices = [
        atom.index
        for topology_residue in structure.topology.residues()
        if topology_residue.index == residue.index
        for atom in topology_residue.atoms()
    ]
    return positions[indices], residue.index


def rotated(structure: Structure, angle: float, shift: np.ndarray) -> Structure:
    """Return a copy of a structure rotated about z and translated.

    Args:
        structure: The structure to move.
        angle: Rotation in radians about the z axis.
        shift: Translation in angstrom.

    Returns:
        A structure with moved coordinates and the same topology.
    """
    positions = np.asarray(structure.positions.value_in_unit(angstrom), dtype=float)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = positions @ rotation.T + shift
    return Structure(structure.path, structure.topology, moved * angstrom)


def test_pocket_backbone_selects_only_backbone_atoms(complex_structure: Structure) -> None:
    center, _ = ligand_center(complex_structure)

    atoms = pocket_backbone(complex_structure, center, radius=10.0)

    assert atoms
    assert {key[2] for key in atoms} <= set(BACKBONE_ATOMS)


def test_pocket_backbone_respects_the_radius(complex_structure: Structure) -> None:
    center, _ = ligand_center(complex_structure)

    near = pocket_backbone(complex_structure, center, radius=4.0)
    far = pocket_backbone(complex_structure, center, radius=20.0)

    assert len(near) < len(far)


def test_superposing_a_structure_onto_itself_is_the_identity(
    complex_structure: Structure,
) -> None:
    center, _ = ligand_center(complex_structure)

    fit = superpose(complex_structure, complex_structure, center, radius=12.0)

    assert fit.rmsd == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(fit.rotation, np.eye(3), atol=1e-6)
    assert np.allclose(fit.translation, 0.0, atol=1e-6)


def test_a_rotated_copy_is_recovered_exactly(complex_structure: Structure) -> None:
    """A rigid transform must be undone to within numerical noise."""
    center, _ = ligand_center(complex_structure)
    moved = rotated(complex_structure, angle=0.7, shift=np.array([12.0, -4.0, 3.0]))

    fit = superpose(moved, complex_structure, center, radius=25.0)

    assert fit.rmsd == pytest.approx(0.0, abs=1e-4)
    assert fit.n_atoms > 12


def test_the_transform_moves_a_ligand_back_into_the_frame(
    complex_structure: Structure,
) -> None:
    center, _ = ligand_center(complex_structure)
    moved = rotated(complex_structure, angle=1.1, shift=np.array([-8.0, 5.0, 2.0]))
    moved_ligand, _ = ligand_center(moved)

    fit = superpose(moved, complex_structure, center, radius=25.0)

    assert fit.apply(moved_ligand) == pytest.approx(center, abs=1e-3)


def test_too_few_matching_atoms_is_rejected(complex_structure: Structure) -> None:
    center, _ = ligand_center(complex_structure)

    with pytest.raises(InputValidationError, match="binding site backbone atoms match"):
        superpose(complex_structure, complex_structure, center, radius=0.1)


def test_pocket_heavy_atoms_excludes_the_ligand(complex_structure: Structure) -> None:
    center, ligand_index = ligand_center(complex_structure)

    with_ligand = pocket_heavy_atoms(complex_structure, center, 10.0, exclude=set())
    without = pocket_heavy_atoms(complex_structure, center, 10.0, exclude={ligand_index})

    assert len(without) < len(with_ligand)
    assert len(without) > 0


def test_pocket_heavy_atoms_omits_hydrogens(complex_structure: Structure) -> None:
    center, _ = ligand_center(complex_structure)

    atoms = pocket_heavy_atoms(complex_structure, center, 25.0, exclude=set())

    heavy = sum(
        1
        for residue in complex_structure.topology.residues()
        for atom in residue.atoms()
        if atom.element is not None and atom.element.symbol != "H"
    )
    assert len(atoms) <= heavy


def test_a_distant_frame_is_still_matched(complex_structure: Structure) -> None:
    """The mobile structure arrives in its own frame, so its site is found by identity."""
    center, _ = ligand_center(complex_structure)
    moved = rotated(complex_structure, angle=2.4, shift=np.array([500.0, -300.0, 120.0]))

    fit = superpose(moved, complex_structure, center, radius=25.0)

    assert fit.rmsd == pytest.approx(0.0, abs=1e-3)
    assert fit.n_atoms > 12


def test_backbone_for_residues_ignores_distance(complex_structure: Structure) -> None:
    center, _ = ligand_center(complex_structure)
    site = pocket_backbone(complex_structure, center, radius=8.0)
    wanted = {(chain, residue_id) for chain, residue_id, _ in site}
    moved = rotated(complex_structure, angle=0.3, shift=np.array([250.0, 0.0, 0.0]))

    found = backbone_for_residues(moved, wanted)

    assert set(found) == set(site)
