"""Preflight for the automatic Boresch restraint search shared by ABFE and SepTop."""

from __future__ import annotations

from gufe import SmallMoleculeComponent
from openfe.protocols.restraint_utils.geometry.utils import get_aromatic_rings

from openfe_api.exceptions import InputValidationError

__all__ = ["check_restraint_search"]


def check_restraint_search(ligand: SmallMoleculeComponent, protocol: str) -> None:
    """Check that OpenFE's automatic Boresch restraint search can handle this ligand.

    In OpenFE 1.12 the search's ring grouping fails for any ligand with three or more fused
    aromatic rings, which otherwise only surfaces once the simulation is running on a GPU.
    OpenFE's own function is called, so this passes again as soon as OpenFE is fixed.

    Args:
        ligand: The ligand that will be restrained.
        protocol: Name of the protocol being planned, for the error message.

    Raises:
        InputValidationError: If the restraint search would fail on this ligand.
    """
    try:
        get_aromatic_rings(ligand.to_rdkit())
    except ValueError as error:
        raise InputValidationError(
            f"{ligand.name}: OpenFE's automatic Boresch restraint search cannot handle this "
            f"ligand ({error}). This is a known limitation of the installed OpenFE affecting "
            "ligands with three or more fused aromatic rings, and would make the complex "
            f"phase fail after the run had started. It cannot be run with the {protocol} "
            "protocol on this OpenFE version."
        ) from error
