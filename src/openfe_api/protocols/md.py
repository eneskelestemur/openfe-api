"""Planning plain molecular dynamics simulations.

One system is one transformation. The protocol compares its two end states by identity, so
both are the same object; gufe's token registry keeps that true across a JSON round trip, so
the transformation runs through ``openfe quickrun`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.settings import Settings
from openfe.protocols.openmm_md import PlainMDProtocol
from openfe.protocols.openmm_md.plain_md_settings import PlainMDProtocolSettings

from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.prep.report import SystemPrepReport
from openfe_api.protocols.charges import ChargeCache, assign_charges, load_molecule
from openfe_api.protocols.settings import apply_overrides, check_selection, nanoseconds
from openfe_api.schema.common import SettingsSpec

__all__ = [
    "MD_PRESETS",
    "MdCost",
    "PlannedMdRun",
    "build_settings",
    "estimate_md_cost",
    "plan_md",
]

logger = get_logger(__name__)

MD_PRESETS: dict[str, dict[str, object]] = {
    "default": {},
    "screening": {
        "simulation_settings.equilibration_length": "0.2 nanosecond",
        "simulation_settings.production_length": "0.5 nanosecond",
    },
}
"""Named settings presets.

``default`` keeps OpenFE's own lengths, 0.1 ns NVT then 1 ns NPT equilibration and 5 ns of
production. ``screening`` shortens the equilibration and production for a quick look. Neither
sets the repeat count, which comes only from ``execution.repeats``.
"""

VACUUM_NONBONDED = "nocutoff"
"""Nonbonded method a vacuum simulation needs, since PME has no box to work in."""


@dataclass(frozen=True)
class MdCost:
    """Simulation time one MD run implies.

    Attributes:
        equilibration_ns: Nanoseconds of equilibration per repeat, NVT and NPT together.
        production_ns: Nanoseconds of production per repeat.
        repeats: Number of independent runs.
    """

    equilibration_ns: float
    production_ns: float
    repeats: int

    @property
    def per_repeat_ns(self) -> float:
        """Total nanoseconds simulated for one repeat.

        Returns:
            Equilibration plus production.
        """
        return self.equilibration_ns + self.production_ns

    @property
    def total_ns(self) -> float:
        """Total nanoseconds simulated across every repeat.

        Returns:
            Per-repeat time multiplied by the number of repeats.
        """
        return self.per_repeat_ns * self.repeats

    def describe(self) -> str:
        """Summarize the cost in one line.

        Returns:
            A human-readable description of the simulation time.
        """
        return (
            f"{self.total_ns:,.2f} ns total "
            f"({self.repeats} repeat(s) x {self.per_repeat_ns:,.2f} ns: "
            f"{self.equilibration_ns:,.2f} ns equilibration, "
            f"{self.production_ns:,.2f} ns production)"
        )


@dataclass(frozen=True)
class PlannedMdRun:
    """One system written out and ready to simulate.

    Attributes:
        name: System name.
        path: The transformation JSON.
        cost: Simulation time the settings imply.
        solvated: Whether the system is solvated.
        n_components: Components in the simulated system.
    """

    name: str
    path: Path
    cost: MdCost
    solvated: bool
    n_components: int


def build_settings(spec: SettingsSpec, solvated: bool = True) -> PlainMDProtocolSettings:
    """Build plain MD settings from a preset and the request's overrides.

    ``protocol_repeats`` is always 1: each repeat runs as its own ``openfe quickrun``
    process, so a higher value here would multiply every one of those processes.

    Args:
        spec: The settings selection from the request.
        solvated: Whether the system will be solvated. A vacuum simulation cannot use PME,
            so the nonbonded method is changed for it and the change is logged.

    Returns:
        The resulting settings.

    Raises:
        InputValidationError: If the preset is unknown, an override is invalid, or the
            overrides try to set ``protocol_repeats``.
    """
    check_selection(spec, MD_PRESETS)
    settings = cast(PlainMDProtocolSettings, PlainMDProtocol.default_settings())
    apply_overrides(settings, MD_PRESETS[spec.preset])

    if not solvated and "forcefield_settings.nonbonded_method" not in spec.overrides:
        settings.forcefield_settings.nonbonded_method = VACUUM_NONBONDED
        logger.warning(
            "This system runs in vacuum, so the nonbonded method is set to '%s'. PME needs a "
            "periodic box and the protocol would refuse the system with it.",
            VACUUM_NONBONDED,
        )

    apply_overrides(settings, spec.overrides)
    settings.protocol_repeats = 1
    return settings


def estimate_md_cost(settings: PlainMDProtocolSettings, repeats: int) -> MdCost:
    """Work out how much simulation a set of MD settings implies.

    Args:
        settings: The settings to measure.
        repeats: Number of independent runs that will be performed.

    Returns:
        The cost estimate.
    """
    simulation = settings.simulation_settings
    nvt = (
        nanoseconds(simulation.equilibration_length_nvt)
        if simulation.equilibration_length_nvt is not None
        else 0.0
    )
    return MdCost(
        equilibration_ns=nvt + nanoseconds(simulation.equilibration_length),
        production_ns=nanoseconds(simulation.production_length),
        repeats=repeats,
    )


def _system(
    report: SystemPrepReport,
    molecules: list[SmallMoleculeComponent],
    solvated: bool,
) -> ChemicalSystem:
    """Build the system to simulate from a prepared report."""
    components: dict[str, Component] = {}
    if report.protein is not None:
        try:
            components["protein"] = ProteinComponent.from_pdb_file(report.protein.path)
        except Exception as error:
            raise InputValidationError(
                f"{report.name}: prepared protein {report.protein.path} could not be loaded "
                f"as a ProteinComponent: {error}"
            ) from error

    for index, molecule in enumerate(molecules):
        components[f"molecule_{index}"] = molecule

    if solvated:
        components["solvent"] = SolventComponent()

    return ChemicalSystem(components, name=report.name)


def plan_md(
    report: SystemPrepReport,
    spec: SettingsSpec,
    output_dir: Path,
    repeats: int,
    solvated: bool = True,
    charge_cache: ChargeCache | None = None,
    processors: int = 1,
) -> PlannedMdRun:
    """Write the transformation for one prepared system.

    Args:
        report: The system's preparation report.
        spec: The settings selection from the request.
        output_dir: Directory to write the transformation JSON into.
        repeats: Repeats that will be run, used for the cost estimate.
        solvated: Whether to solvate the system.
        charge_cache: Cache for partial charges. If None, charges are recomputed.
        processors: Processes to use for charge generation.

    Returns:
        A description of the planned run.

    Raises:
        InputValidationError: If the prepared files are unreadable or the protocol rejects
            the system.
        OpenFEAPIError: If the transformation cannot be written.
    """
    settings = build_settings(spec, solvated=solvated)
    molecules = [load_molecule(entry.path, entry.name) for entry in report.molecules]
    if molecules:
        molecules = assign_charges(
            molecules,
            settings.partial_charge_settings,
            cache=charge_cache,
            processors=processors,
        )

    system = _system(report, molecules, solvated)
    protocol = PlainMDProtocol(cast(Settings, settings))
    try:
        # The protocol compares its end states by identity, so the same object goes in twice.
        protocol.validate(stateA=system, stateB=system, mapping=None)
    except Exception as error:
        raise InputValidationError(
            f"{report.name}: the MD protocol rejected this system: {error}"
        ) from error

    name = f"md_{report.name}"
    transformation = Transformation(
        stateA=system,
        stateB=system,
        protocol=protocol,
        mapping=None,
        name=name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    try:
        transformation.to_json(path)
    except Exception as error:
        raise OpenFEAPIError(f"{name}: could not write {path}: {error}") from error

    cost = estimate_md_cost(settings, repeats)
    logger.info("Planned MD run '%s': %s", report.name, cost.describe())
    return PlannedMdRun(
        name=report.name,
        path=path,
        cost=cost,
        solvated=solvated,
        n_components=len(system.components),
    )
