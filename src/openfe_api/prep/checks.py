"""Chemistry and geometry checks applied to prepared molecules."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from openfe_api.exceptions import InputValidationError

__all__ = [
    "SUPPORTED_ELEMENTS",
    "check_clashes",
    "check_elements",
    "check_net_charge",
    "check_radicals",
]

SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

CLASH_DISTANCE = 1.5
"""Heavy atom separation, in angstrom, below which two atoms are considered clashing."""


def check_elements(molecule: Chem.Mol, name: str) -> None:
    """Check that every element is covered by the small molecule force field.

    Args:
        molecule: The molecule to check.
        name: Molecule name, used in error messages.

    Raises:
        InputValidationError: If the molecule contains an unsupported element.
    """
    found = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    unsupported = sorted(found - SUPPORTED_ELEMENTS)
    if unsupported:
        raise InputValidationError(
            f"{name}: contains element(s) {', '.join(unsupported)}, which the OpenFF small "
            f"molecule force field does not cover. Supported elements: "
            f"{', '.join(sorted(SUPPORTED_ELEMENTS))}."
        )


def check_radicals(molecule: Chem.Mol, name: str) -> None:
    """Check that the molecule has no unpaired electrons.

    Args:
        molecule: The molecule to check.
        name: Molecule name, used in error messages.

    Raises:
        InputValidationError: If any atom carries radical electrons.
    """
    radicals = [
        f"{atom.GetSymbol()}{atom.GetIdx()}"
        for atom in molecule.GetAtoms()
        if atom.GetNumRadicalElectrons() > 0
    ]
    if radicals:
        raise InputValidationError(
            f"{name}: has unpaired electrons on atom(s) {', '.join(radicals)}. Radicals "
            "cannot be parameterized; check the SMILES valences."
        )


def check_net_charge(net_charge: int, name: str, neutralize_available: bool = True) -> None:
    """Check that a ligand is neutral, as the ABFE protocol requires.

    Args:
        net_charge: The molecule's net formal charge.
        name: Molecule name, used in error messages.
        neutralize_available: Whether to mention the neutralization option in the error.

    Raises:
        InputValidationError: If the net charge is not zero.
    """
    if net_charge == 0:
        return

    remedy = (
        " Supply a neutral form, or set 'neutralize_ligands: true' to have the molecule "
        "neutralized automatically, which changes it chemically."
        if neutralize_available
        else " Supply a neutral form."
    )
    raise InputValidationError(
        f"{name}: has a net charge of {net_charge:+d}. The OpenFE absolute binding free "
        f"energy protocol only supports neutral ligands.{remedy}"
    )


def check_clashes(
    molecule: Chem.Mol,
    other_positions: np.ndarray,
    name: str,
    other_name: str,
    threshold: float = CLASH_DISTANCE,
) -> list[str]:
    """Check a molecule's heavy atoms against another set of coordinates.

    Args:
        molecule: The molecule to check; its first conformer is used.
        other_positions: Coordinates to check against, shape ``(n, 3)`` in angstrom.
        name: Molecule name, used in messages.
        other_name: Name of what the molecule is checked against, used in messages.
        threshold: Separation below which atoms are reported as clashing, in angstrom.

    Returns:
        Warnings describing any clashes found. Empty if the pose is clear.

    Raises:
        InputValidationError: If the molecule has no conformer.
    """
    if molecule.GetNumConformers() == 0:
        raise InputValidationError(f"{name}: has no 3D coordinates to check for clashes")
    if other_positions.size == 0:
        return []

    conformer = molecule.GetConformer()
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    coordinates = np.array([list(conformer.GetAtomPosition(index)) for index in heavy])

    distances = np.linalg.norm(
        coordinates[:, None, :] - other_positions[None, :, :],
        axis=-1,
    )
    closest = distances.min()
    if closest >= threshold:
        return []

    count = int((distances < threshold).any(axis=1).sum())
    return [
        (
            f"{name}: {count} heavy atom(s) lie within {threshold} A of {other_name} "
            f"(closest {closest:.2f} A). The pose may be distorted; check it before running."
        )
    ]
