"""Tests for the chemistry and geometry checks."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from openfe_api.exceptions import InputValidationError
from openfe_api.prep import checks


def _embedded(smiles: str) -> Chem.Mol:
    """Return a molecule with 3D coordinates.

    Args:
        smiles: SMILES to embed.

    Returns:
        The embedded molecule, with hydrogens.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    return molecule


def test_supported_elements_pass() -> None:
    checks.check_elements(_embedded("CC(=O)Nc1ccc(F)cc1S(=O)(=O)Cl"), "ligand")


def test_unsupported_element_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="contains element\\(s\\) B"):
        checks.check_elements(_embedded("BC"), "ligand")


def test_metal_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="Fe"):
        checks.check_elements(Chem.MolFromSmiles("[Fe]"), "cofactor")


def test_radicals_are_rejected() -> None:
    with pytest.raises(InputValidationError, match="unpaired electrons"):
        checks.check_radicals(Chem.MolFromSmiles("[CH3]"), "ligand")


def test_paired_electrons_pass() -> None:
    checks.check_radicals(_embedded("CCO"), "ligand")


def test_neutral_ligand_passes() -> None:
    checks.check_net_charge(0, "ligand")


def test_charged_ligand_is_rejected_with_the_neutralize_hint() -> None:
    with pytest.raises(InputValidationError, match=r"net charge of -1.*neutralize_ligands"):
        checks.check_net_charge(-1, "ligand")


def test_charged_ligand_message_without_the_hint() -> None:
    with pytest.raises(InputValidationError, match="Supply a neutral form"):
        checks.check_net_charge(2, "ligand", neutralize_available=False)


def test_clashes_are_reported() -> None:
    molecule = _embedded("CCO")
    position = molecule.GetConformer().GetAtomPosition(0)
    overlapping = np.array([[position.x, position.y, position.z]])

    warnings = checks.check_clashes(molecule, overlapping, "ligand", "the protein")

    assert len(warnings) == 1
    assert "within 1.5 A of the protein" in warnings[0]


def test_a_clear_pose_reports_nothing() -> None:
    molecule = _embedded("CCO")
    far_away = np.array([[100.0, 100.0, 100.0]])

    assert checks.check_clashes(molecule, far_away, "ligand", "the protein") == []


def test_no_reference_atoms_reports_nothing() -> None:
    assert checks.check_clashes(_embedded("CCO"), np.empty((0, 3)), "ligand", "x") == []


def test_clash_check_requires_coordinates() -> None:
    with pytest.raises(InputValidationError, match="no 3D coordinates"):
        checks.check_clashes(Chem.MolFromSmiles("CCO"), np.array([[0.0, 0.0, 0.0]]), "ligand", "x")
