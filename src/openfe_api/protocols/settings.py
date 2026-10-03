"""Protocol settings: named presets and dotted-path overrides.

A preset is a starting point taken from OpenFE's own defaults; overrides are applied on
top of it. Every override is checked against the settings model, so a misspelled path
fails at planning time rather than silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openff.units import unit
from pydantic import BaseModel

from openfe_api.exceptions import InputValidationError

__all__ = [
    "PRESETS",
    "CostEstimate",
    "apply_overrides",
    "check_selection",
    "estimate_cost",
    "nanoseconds",
    "parse_value",
]

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "screening": {
        "complex_equil_simulation_settings.production_length": "1 nanosecond",
        "complex_simulation_settings.production_length": "2 nanosecond",
        "solvent_equil_simulation_settings.production_length": "0.25 nanosecond",
        "solvent_simulation_settings.production_length": "2 nanosecond",
    },
}
"""Named settings presets.

``default`` keeps OpenFE's simulation defaults unchanged. ``screening`` shortens the
simulations, for triage rather than for numbers you would publish. Neither preset changes
the lambda schedules, because the number of windows must keep matching the number of
replicas, and neither sets the number of repeats, which comes only from the request's
``execution.repeats``.
"""


def check_selection(spec: Any, presets: dict[str, dict[str, Any]]) -> None:
    """Check a settings selection before it is applied to a protocol's defaults.

    Every protocol refuses the same two things, so they are refused in one place.

    Args:
        spec: The settings selection from the request.
        presets: The presets the protocol offers.

    Raises:
        InputValidationError: If the preset is unknown, or the overrides try to set
            ``protocol_repeats``.
    """
    if spec.preset not in presets:
        raise InputValidationError(
            f"unknown settings preset '{spec.preset}'; available: {', '.join(sorted(presets))}"
        )
    if "protocol_repeats" in spec.overrides:
        raise InputValidationError(
            "settings override 'protocol_repeats' is not allowed: set the number of repeats "
            "with 'execution.repeats' instead. Each repeat runs as its own process, so "
            "OpenFE's protocol_repeats is fixed at 1."
        )


def parse_value(value: Any) -> Any:
    """Convert a value from a request into what the settings model expects.

    Strings holding a number and a unit, such as ``"310 kelvin"``, become unit-aware
    quantities. Everything else is passed through unchanged.

    Args:
        value: The raw value from the request.

    Returns:
        The converted value.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    first = text.split(" ", 1)[0]
    try:
        float(first)
    except ValueError:
        return value

    try:
        return unit.Quantity(text)
    except Exception:
        return value


def apply_overrides[SettingsT: BaseModel](
    settings: SettingsT, overrides: dict[str, Any]
) -> SettingsT:
    """Apply dotted-path overrides to a settings object, in place.

    Args:
        settings: The settings object to modify.
        overrides: Mapping of dotted paths to values, for example
            ``{"thermo_settings.temperature": "310 kelvin"}``.

    Returns:
        The same settings object, modified.

    Raises:
        InputValidationError: If a path does not exist in the settings model, or the
            value is rejected by it.
    """
    for path, raw in overrides.items():
        parts = path.split(".")
        target: Any = settings
        for index, part in enumerate(parts[:-1]):
            if not hasattr(target, part):
                raise InputValidationError(
                    f"settings override '{path}': '{'.'.join(parts[: index + 1])}' is not a "
                    f"settings group; available: {', '.join(sorted(type(target).model_fields))}"
                )
            target = getattr(target, part)

        leaf = parts[-1]
        if not isinstance(target, BaseModel) or leaf not in type(target).model_fields:
            available = (
                ", ".join(sorted(type(target).model_fields))
                if isinstance(target, BaseModel)
                else "none"
            )
            raise InputValidationError(
                f"settings override '{path}': '{leaf}' is not a setting; available: {available}"
            )

        try:
            setattr(target, leaf, parse_value(raw))
        except Exception as error:
            raise InputValidationError(
                f"settings override '{path}': value {raw!r} was rejected: {error}"
            ) from error

    return settings


@dataclass(frozen=True)
class CostEstimate:
    """Simulation time implied by a set of settings.

    Attributes:
        complex_ns: Nanoseconds of the complex leg, per repeat.
        solvent_ns: Nanoseconds of the solvent leg, per repeat.
        repeats: Number of independent repeats.
        complex_replicas: Lambda windows in the complex leg.
        solvent_replicas: Lambda windows in the solvent leg.
    """

    complex_ns: float
    solvent_ns: float
    repeats: int
    complex_replicas: int
    solvent_replicas: int

    @property
    def per_repeat_ns(self) -> float:
        """Total nanoseconds simulated for one repeat.

        Returns:
            The sum of both legs.
        """
        return self.complex_ns + self.solvent_ns

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
            f"{self.total_ns:,.0f} ns total "
            f"({self.repeats} repeat(s) x {self.per_repeat_ns:,.0f} ns: "
            f"complex {self.complex_replicas} windows / {self.complex_ns:,.0f} ns, "
            f"solvent {self.solvent_replicas} windows / {self.solvent_ns:,.0f} ns)"
        )


def nanoseconds(quantity: Any) -> float:
    """Convert a time quantity to nanoseconds.

    Args:
        quantity: A unit-aware time quantity.

    Returns:
        The value in nanoseconds.
    """
    return float(quantity.to(unit.nanosecond).m)


def estimate_cost(settings: Any, repeats: int) -> CostEstimate:
    """Work out how much simulation a set of ABFE settings implies.

    Each leg runs its pre-equilibration MD once, then every lambda window runs the
    alchemical equilibration and production. That whole cycle is one repeat.

    Args:
        settings: The ABFE settings to measure.
        repeats: Number of independent repeats that will be run.

    Returns:
        The cost estimate.
    """

    def leg(equil: Any, alchemical: Any) -> tuple[float, int]:
        replicas = alchemical.n_replicas
        pre = (
            nanoseconds(equil.equilibration_length_nvt)
            + nanoseconds(equil.equilibration_length)
            + nanoseconds(equil.production_length)
        )
        per_window = nanoseconds(alchemical.equilibration_length) + nanoseconds(
            alchemical.production_length
        )
        return pre + replicas * per_window, replicas

    complex_ns, complex_replicas = leg(
        settings.complex_equil_simulation_settings, settings.complex_simulation_settings
    )
    solvent_ns, solvent_replicas = leg(
        settings.solvent_equil_simulation_settings, settings.solvent_simulation_settings
    )

    return CostEstimate(
        complex_ns=complex_ns,
        solvent_ns=solvent_ns,
        repeats=repeats,
        complex_replicas=complex_replicas,
        solvent_replicas=solvent_replicas,
    )
