"""Tests for planning RBFE transformations from a prepared series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from gufe import Transformation
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)

from helpers import mini_series_data
from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.rbfe import (
    RBFE_PRESETS,
    build_settings,
    estimate_edge_cost,
)
from openfe_api.protocols.runner import plan_campaign
from openfe_api.schema.common import SettingsSpec

FAST_CHARGES = {"partial_charge_settings.partial_charge_method": "nagl"}


@pytest.fixture
def series_data() -> dict[str, Any]:
    """Return a two-ligand RBFE request on the trimmed complex fixture.

    Returns:
        Request data for the smallest network that has an edge.
    """
    return mini_series_data(
        "rbfe",
        "mini_series",
        protein={"chains": ["A"]},
        settings={"preset": "screening", "overrides": dict(FAST_CHARGES)},
        execution={"repeats": 1},
    )


@pytest.fixture
def planned_campaign(series_data: dict[str, Any], tmp_path: Path) -> Campaign:
    """Create, prepare and plan a small RBFE campaign.

    Args:
        series_data: The request data.
        tmp_path: Pytest temporary directory.

    Returns:
        The planned campaign.
    """
    from openfe_api.schema.request import validate_request

    campaign = Campaign.create(tmp_path / "campaign", validate_request(series_data))
    prepare_campaign(campaign, check_parameters=False)
    plan_campaign(campaign)
    return campaign


def test_presets_do_not_set_repeats_or_windows() -> None:
    """Repeats come only from execution.repeats, and windows must match the replica count."""
    for overrides in RBFE_PRESETS.values():
        assert "protocol_repeats" not in overrides
        assert not any(key.startswith("lambda_settings") for key in overrides)


def test_build_settings_forces_one_repeat() -> None:
    settings = build_settings(SettingsSpec())

    assert settings.protocol_repeats == 1


def test_screening_shortens_the_simulation() -> None:
    default = build_settings(SettingsSpec())
    screening = build_settings(SettingsSpec(preset="screening"))

    assert screening.simulation_settings.production_length.m < (
        default.simulation_settings.production_length.m
    )


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="unknown settings preset"):
        build_settings(SettingsSpec.model_construct(preset="fast", overrides={}))


def test_a_repeats_override_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="'protocol_repeats' is not allowed"):
        build_settings(SettingsSpec(overrides={"protocol_repeats": 3}))


def test_overrides_are_applied() -> None:
    settings = build_settings(SettingsSpec(overrides={"thermo_settings.temperature": "310 kelvin"}))

    temperature = settings.thermo_settings.temperature
    assert temperature is not None
    assert temperature.m == pytest.approx(310.0)


def test_cost_counts_both_phases() -> None:
    settings = build_settings(SettingsSpec(preset="screening"))

    cost = estimate_edge_cost(settings, repeats=2)

    assert cost.complex_ns == cost.solvent_ns
    assert cost.repeats == 2
    assert cost.total_ns == pytest.approx(cost.per_repeat_ns * 2)


def test_each_edge_writes_both_phases(planned_campaign: Campaign) -> None:
    plans = planned_campaign.directory / "plans"

    names = sorted(path.name for path in plans.glob("rbfe_*.json"))

    assert names == [
        "rbfe_reference_to_fluoro_complex.json",
        "rbfe_reference_to_fluoro_solvent.json",
    ]


def test_the_transformations_load_back(planned_campaign: Campaign) -> None:
    for path in (planned_campaign.directory / "plans").glob("rbfe_*.json"):
        transformation = cast(Transformation, Transformation.from_json(path))
        settings = cast(RelativeHybridTopologyProtocolSettings, transformation.protocol.settings)
        assert transformation.mapping is not None
        assert settings.protocol_repeats == 1


def test_the_complex_phase_holds_the_protein(planned_campaign: Campaign) -> None:
    path = planned_campaign.directory / "plans" / "rbfe_reference_to_fluoro_complex.json"

    transformation = cast(Transformation, Transformation.from_json(path))

    assert "protein" in transformation.stateA.components
    assert "protein" in transformation.stateB.components


def test_the_solvent_phase_has_no_protein(planned_campaign: Campaign) -> None:
    path = planned_campaign.directory / "plans" / "rbfe_reference_to_fluoro_solvent.json"

    transformation = cast(Transformation, Transformation.from_json(path))

    assert "protein" not in transformation.stateA.components
    assert set(transformation.stateA.components) == {"ligand", "solvent"}


def test_the_network_is_persisted(planned_campaign: Campaign) -> None:
    plans = planned_campaign.directory / "plans"

    assert (plans / "network.graphml").is_file()
    summary = json.loads((plans / "network.json").read_text(encoding="utf-8"))
    assert [edge["name"] for edge in summary["edges"]] == ["reference_to_fluoro"]
    assert summary["unreachable"] == []


def test_edges_become_runs_in_the_manifest(planned_campaign: Campaign) -> None:
    """A campaign starts with no runs, and planning registers one per edge."""
    record = planned_campaign.run("reference_to_fluoro")

    assert record.state is RunState.PLANNED
    assert record.repeats == 1


def test_replanning_keeps_the_same_edges(planned_campaign: Campaign) -> None:
    before = json.loads(
        (planned_campaign.directory / "plans" / "network.json").read_text(encoding="utf-8")
    )

    plan_campaign(planned_campaign)

    after = json.loads(
        (planned_campaign.directory / "plans" / "network.json").read_text(encoding="utf-8")
    )
    assert before["edges"] == after["edges"]


def test_planning_before_preparing_is_rejected(series_data: dict[str, Any], tmp_path: Path) -> None:
    from openfe_api.schema.request import validate_request

    campaign = Campaign.create(tmp_path / "campaign", validate_request(series_data))

    with pytest.raises(InputValidationError, match="not prepared yet"):
        plan_campaign(campaign)


def test_only_is_rejected_for_a_series(planned_campaign: Campaign) -> None:
    from openfe_api.exceptions import OpenFEAPIError

    with pytest.raises(OpenFEAPIError, match="'--only' selects complexes"):
        plan_campaign(planned_campaign, only=["reference_to_fluoro"])
