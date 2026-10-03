"""Planning absolute binding free energy transformations.

The protocol takes a single pair of chemical systems: state A holds the protein, the
ligand, any cofactors and the solvent, and state B is the same system with the ligand
removed. OpenFE builds both legs of the thermodynamic cycle from that pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.settings import Settings
from openfe.protocols.openmm_afe import AbsoluteBindingProtocol
from openfe.protocols.openmm_afe.equil_afe_settings import AbsoluteBindingSettings

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.prep.report import ComplexPrepReport
from openfe_api.protocols.charges import ChargeCache, assign_charges, load_molecule
from openfe_api.protocols.restraints import check_restraint_search
from openfe_api.protocols.settings import (
    PRESETS,
    CostEstimate,
    apply_overrides,
    check_selection,
    estimate_cost,
)
from openfe_api.schema.common import SettingsSpec

__all__ = ["PlannedTransformation", "build_settings", "plan_abfe"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlannedTransformation:
    """A transformation written out and ready to run.

    Attributes:
        name: Transformation name.
        path: The transformation JSON file.
        cost: Simulation time the settings imply.
        ligand_name: Name of the alchemical ligand.
        n_components: Number of components in state A.
    """

    name: str
    path: Path
    cost: CostEstimate
    ligand_name: str
    n_components: int


def build_settings(spec: SettingsSpec) -> AbsoluteBindingSettings:
    """Build ABFE settings from a preset and the request's overrides.

    ``protocol_repeats`` is always 1. Each repeat runs as its own ``openfe quickrun``
    process, so a higher value here would silently multiply every one of those processes.

    Args:
        spec: The settings selection from the request.

    Returns:
        The resulting ``AbsoluteBindingSettings``.

    Raises:
        InputValidationError: If the preset is unknown, an override is invalid, or the
            overrides try to set ``protocol_repeats``.
    """
    check_selection(spec, PRESETS)
    settings = cast(AbsoluteBindingSettings, AbsoluteBindingProtocol.default_settings())
    apply_overrides(settings, PRESETS[spec.preset])
    apply_overrides(settings, spec.overrides)
    settings.protocol_repeats = 1
    return settings


def plan_abfe(
    report: ComplexPrepReport,
    settings: AbsoluteBindingSettings,
    output_dir: Path,
    repeats: int,
    charge_cache: ChargeCache | None = None,
    processors: int = 1,
) -> PlannedTransformation:
    """Plan the ABFE transformation for one prepared complex.

    Args:
        report: The preparation report describing the prepared files.
        settings: ABFE settings, as built by :func:`build_settings`.
        output_dir: Directory to write the transformation JSON into.
        repeats: Number of repeats that will be run, used for the cost estimate.
        charge_cache: Cache for partial charges. If None, charges are recomputed.
        processors: Number of processes to use for charge generation.

    Returns:
        A description of the planned transformation.

    Raises:
        InputValidationError: If the prepared files are unreadable, the restraint search
            would fail on the ligand, or the protocol rejects the chemical systems.
        OpenFEAPIError: If charge assignment fails.
    """
    ligand = load_molecule(report.ligand.path, report.ligand.name)
    check_restraint_search(ligand, "ABFE")
    others = [
        load_molecule(entry.path, entry.name)
        for entry in [*report.cofactors, *report.ligand_copies]
    ]

    charged = assign_charges(
        [ligand, *others],
        settings.partial_charge_settings,
        cache=charge_cache,
        processors=processors,
    )
    ligand, others = charged[0], charged[1:]

    try:
        protein = ProteinComponent.from_pdb_file(report.protein.path)
    except Exception as error:
        raise InputValidationError(
            f"{report.name}: prepared protein {report.protein.path} could not be loaded "
            f"as a ProteinComponent: {error}"
        ) from error

    solvent = SolventComponent()
    shared: dict[str, ProteinComponent | SolventComponent | SmallMoleculeComponent] = {
        "protein": protein,
        "solvent": solvent,
    }
    for molecule in others:
        shared[f"cofactor_{molecule.name}"] = molecule

    state_a = ChemicalSystem({**shared, "ligand": ligand}, name=f"{report.name}_bound")
    state_b = ChemicalSystem(dict(shared), name=f"{report.name}_unbound")

    protocol = AbsoluteBindingProtocol(cast(Settings, settings))
    try:
        protocol.validate(stateA=state_a, stateB=state_b, mapping=None)
    except Exception as error:
        raise InputValidationError(
            f"{report.name}: the ABFE protocol rejected this system: {error}"
        ) from error

    name = f"abfe_{report.name}"
    transformation = Transformation(
        stateA=state_a,
        stateB=state_b,
        protocol=protocol,
        mapping=None,
        name=name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    try:
        transformation.to_json(path)
    except Exception as error:
        raise OpenFEAPIError(f"{report.name}: could not write {path}: {error}") from error

    cost = estimate_cost(settings, repeats)
    logger.info("Planned '%s': %s", name, cost.describe())

    return PlannedTransformation(
        name=name,
        path=path,
        cost=cost,
        ligand_name=ligand.name,
        n_components=len(state_a.components),
    )
