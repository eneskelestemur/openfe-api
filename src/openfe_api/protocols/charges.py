"""Loading prepared molecules and assigning their partial charges once.

Charge assignment is the slow part of planning: AM1BCC takes minutes for a drug-sized
molecule. Charges are therefore computed once per molecule and cached on disk, so every
repeat of a run shares identical charges, as OpenFE recommends, and re-planning a campaign
does not pay the cost again.
"""

from __future__ import annotations

import json
from pathlib import Path

from gufe import SmallMoleculeComponent
from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger

__all__ = ["ChargeCache", "assign_charges", "load_molecule"]

logger = get_logger(__name__)


class ChargeCache:
    """A directory of charged molecules, keyed by molecule and charge method.

    Attributes:
        directory: Directory holding the cached SDF files and their metadata.
    """

    def __init__(self, directory: Path) -> None:
        """Open or create a charge cache.

        Args:
            directory: Directory to store cached molecules in.
        """
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, name: str, smiles: str, method: str) -> SmallMoleculeComponent | None:
        """Return a cached charged molecule, if one matches.

        Args:
            name: Molecule name.
            smiles: SMILES the cached entry must match.
            method: Charge method the cached entry must match.

        Returns:
            The cached molecule, or None if there is no usable entry.
        """
        sdf_path = self.directory / f"{name}.sdf"
        meta_path = self.directory / f"{name}.json"
        if not (sdf_path.is_file() and meta_path.is_file()):
            return None

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        if metadata.get("smiles") != smiles or metadata.get("method") != method:
            return None

        try:
            return SmallMoleculeComponent.from_sdf_file(str(sdf_path))
        except Exception:
            logger.debug("Ignoring unreadable cached charges for '%s'", name)
            return None

    def store(self, molecule: SmallMoleculeComponent, smiles: str, method: str) -> None:
        """Write a charged molecule to the cache.

        Args:
            molecule: The charged molecule.
            smiles: SMILES to record alongside it.
            method: Charge method to record alongside it.
        """
        (self.directory / f"{molecule.name}.sdf").write_text(molecule.to_sdf(), encoding="utf-8")
        (self.directory / f"{molecule.name}.json").write_text(
            json.dumps({"smiles": smiles, "method": method}, indent=2), encoding="utf-8"
        )


def assign_charges(
    molecules: list[SmallMoleculeComponent],
    settings: OpenFFPartialChargeSettings,
    cache: ChargeCache | None = None,
    processors: int = 1,
) -> list[SmallMoleculeComponent]:
    """Assign partial charges to molecules, reusing cached results where possible.

    Args:
        molecules: The molecules to charge.
        settings: Settings describing the charge method and toolkit backend.
        cache: Cache to read from and write to. If None, nothing is cached.
        processors: Number of processes to use for charge generation.

    Returns:
        The charged molecules, in the order they were given.

    Raises:
        OpenFEAPIError: If charge generation fails.
    """
    method = settings.partial_charge_method
    backend = settings.off_toolkit_backend

    charged: dict[str, SmallMoleculeComponent] = {}
    pending: list[SmallMoleculeComponent] = []
    for molecule in molecules:
        cached = cache.load(molecule.name, molecule.smiles, method) if cache else None
        if cached is not None:
            logger.debug("Reusing cached %s charges for '%s'", method, molecule.name)
            charged[molecule.name] = cached
        else:
            pending.append(molecule)

    if pending:
        names = ", ".join(molecule.name for molecule in pending)
        logger.info("Assigning %s charges to %d molecule(s): %s", method, len(pending), names)
        try:
            results = bulk_assign_partial_charges(
                molecules=pending,
                overwrite=True,
                method=method,
                toolkit_backend=backend,
                generate_n_conformers=settings.number_of_conformers,
                nagl_model=settings.nagl_model,
                processors=max(1, min(processors, len(pending))),
            )
        except Exception as error:
            raise OpenFEAPIError(
                f"partial charge assignment failed using '{method}' with the "
                f"'{backend}' backend: {error}"
            ) from error

        # By name, not position: the backend returns results in completion order, which once
        # gave two stereoisomers each other's SMILES.
        by_name = {molecule.name: molecule for molecule in pending}
        if {result.name for result in results} != set(by_name):
            raise OpenFEAPIError(
                f"partial charge assignment returned molecules that do not match the ones it "
                f"was given: asked for {sorted(by_name)}, got "
                f"{sorted(result.name for result in results)}"
            )
        for result in results:
            charged[result.name] = result
            if cache is not None:
                cache.store(result, by_name[result.name].smiles, method)

    return [charged[molecule.name] for molecule in molecules]


def load_molecule(path: Path, name: str) -> SmallMoleculeComponent:
    """Load a prepared molecule from its SDF file.

    Args:
        path: The SDF file written by preparation.
        name: Name to give the component.

    Returns:
        The component.

    Raises:
        InputValidationError: If the file cannot be read.
    """
    try:
        molecule = SmallMoleculeComponent.from_sdf_file(str(path))
    except Exception as error:
        raise InputValidationError(
            f"{name}: prepared molecule {path} is unreadable: {error}"
        ) from error
    return SmallMoleculeComponent(molecule.to_rdkit(), name=name)
