"""Tests for settings presets, overrides and cost estimation."""

from __future__ import annotations

import pytest
from openff.units import unit
from pydantic import ValidationError

from openfe_api.exceptions import InputValidationError
from openfe_api.protocols.abfe import build_settings
from openfe_api.protocols.settings import (
    PRESETS,
    apply_overrides,
    estimate_cost,
    parse_value,
)
from openfe_api.schema.common import SettingsSpec


def test_parse_value_reads_a_quantity() -> None:
    assert parse_value("310 kelvin") == 310 * unit.kelvin


def test_parse_value_reads_a_decimal_quantity() -> None:
    assert parse_value("0.5 nanosecond") == 0.5 * unit.nanosecond


def test_parse_value_leaves_plain_strings_alone() -> None:
    assert parse_value("repex") == "repex"


def test_parse_value_leaves_numbers_alone() -> None:
    assert parse_value(3) == 3
    assert parse_value(True) is True


def test_default_preset_matches_openfe_defaults() -> None:
    settings = build_settings(SettingsSpec(preset="default"))

    assert settings.complex_simulation_settings.n_replicas == 30
    assert settings.solvent_simulation_settings.n_replicas == 14
    assert settings.complex_simulation_settings.production_length == 10 * unit.nanosecond


def test_screening_preset_shortens_the_run() -> None:
    settings = build_settings(SettingsSpec(preset="screening"))

    assert settings.complex_simulation_settings.production_length == 2 * unit.nanosecond


def test_screening_preset_keeps_the_lambda_windows() -> None:
    """Window counts must keep matching the lambda schedules, so presets never touch them."""
    settings = build_settings(SettingsSpec(preset="screening"))

    assert settings.complex_simulation_settings.n_replicas == 30
    assert len(settings.complex_lambda_settings.lambda_vdw) == 30


def test_overrides_are_applied_on_top_of_the_preset() -> None:
    settings = build_settings(
        SettingsSpec(
            preset="screening",
            overrides={
                "thermo_settings.temperature": "310 kelvin",
                "complex_simulation_settings.production_length": "5 nanosecond",
            },
        )
    )

    assert settings.thermo_settings.temperature == 310 * unit.kelvin
    assert settings.complex_simulation_settings.production_length == 5 * unit.nanosecond


def test_unknown_preset_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError, match="preset"):
        SettingsSpec(preset="turbo")  # type: ignore[arg-type]


def test_unknown_preset_is_rejected_by_the_planner() -> None:
    """The planner guards the preset too, for callers that bypass schema validation."""
    spec = SettingsSpec.model_construct(preset="turbo", overrides={})

    with pytest.raises(InputValidationError, match="unknown settings preset"):
        build_settings(spec)


def test_unknown_override_leaf_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="'temprature' is not a setting"):
        build_settings(SettingsSpec(overrides={"thermo_settings.temprature": "310 kelvin"}))


def test_unknown_override_group_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="is not a settings group"):
        build_settings(SettingsSpec(overrides={"thermodynamics.temperature": "310 kelvin"}))


def test_top_level_unknown_override_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="'repeats' is not a setting"):
        build_settings(SettingsSpec(overrides={"repeats": 2}))


def test_invalid_override_value_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="was rejected"):
        build_settings(SettingsSpec(overrides={"thermo_settings.temperature": "hot"}))


def test_apply_overrides_returns_the_same_object() -> None:
    settings = build_settings(SettingsSpec())

    assert apply_overrides(settings, {"thermo_settings.temperature": "310 kelvin"}) is settings


def test_cost_estimate_counts_both_legs() -> None:
    settings = build_settings(SettingsSpec(preset="default"))

    cost = estimate_cost(settings, repeats=3)

    assert cost.complex_replicas == 30
    assert cost.solvent_replicas == 14
    # complex: 0.25 + 0.5 + 5 pre-equilibration, then 30 windows x 11 ns
    assert cost.complex_ns == pytest.approx(335.75)
    # solvent: 0.1 + 0.2 + 0.5 pre-equilibration, then 14 windows x 11 ns
    assert cost.solvent_ns == pytest.approx(154.8)
    assert cost.total_ns == pytest.approx(3 * (335.75 + 154.8))


def test_cost_estimate_scales_with_repeats() -> None:
    settings = build_settings(SettingsSpec())
    single = estimate_cost(settings, repeats=1)
    double = estimate_cost(settings, repeats=2)

    assert double.total_ns == pytest.approx(2 * single.total_ns)


def test_cost_description_mentions_both_legs() -> None:
    description = estimate_cost(build_settings(SettingsSpec()), repeats=3).describe()

    assert "complex 30 windows" in description
    assert "solvent 14 windows" in description


def test_every_preset_builds() -> None:
    for name in PRESETS:
        build_settings(SettingsSpec.model_construct(preset=name, overrides={}))


def test_protocol_repeats_is_always_one() -> None:
    """Each repeat is its own process, so OpenFE must run exactly one per process."""
    for preset in PRESETS:
        settings = build_settings(SettingsSpec.model_construct(preset=preset, overrides={}))

        assert settings.protocol_repeats == 1


def test_protocol_repeats_override_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="set the number of repeats"):
        build_settings(SettingsSpec(overrides={"protocol_repeats": 3}))
