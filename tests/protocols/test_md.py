"""Tests for planning plain MD simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from gufe import Transformation

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.md import MD_PRESETS, build_settings, estimate_md_cost
from openfe_api.protocols.runner import plan_campaign
from openfe_api.schema.common import SettingsSpec
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"

FAST_CHARGES = {"partial_charge_settings.partial_charge_method": "nagl"}


def md_data(**system_overrides: Any) -> dict[str, Any]:
    """Build an MD request on the trimmed complex fixture.

    Args:
        **system_overrides: Fields to replace on the system.

    Returns:
        Request data ready for validation.
    """
    system: dict[str, Any] = {
        "name": "mini",
        "structure": str(DATA / "mini_complex.cif"),
        "protein": {"chains": ["A"]},
        "ligands": [{"name": "lig", "selector": {"chain": "L"}, "smiles": "C[C@H](O)c1ccccc1"}],
    }
    system.update(system_overrides)
    return {
        "protocol": "md",
        "name": "md_mini",
        "systems": [system],
        "settings": {"preset": "screening", "overrides": dict(FAST_CHARGES)},
        "execution": {"repeats": 1},
    }


@pytest.fixture
def planned_campaign(tmp_path: Path) -> Campaign:
    """Create, prepare and plan a small MD campaign.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The planned campaign.
    """
    campaign = Campaign.create(tmp_path / "campaign", validate_request(md_data()))
    prepare_campaign(campaign, check_parameters=False)
    plan_campaign(campaign)
    return campaign


def test_presets_do_not_set_repeats() -> None:
    for overrides in MD_PRESETS.values():
        assert "protocol_repeats" not in overrides


def test_build_settings_forces_one_repeat() -> None:
    assert build_settings(SettingsSpec()).protocol_repeats == 1


def test_screening_shortens_the_simulation() -> None:
    default = build_settings(SettingsSpec())
    screening = build_settings(SettingsSpec(preset="screening"))

    assert (
        screening.simulation_settings.production_length.m
        < default.simulation_settings.production_length.m
    )


def test_the_defaults_come_from_openfe() -> None:
    settings = build_settings(SettingsSpec())

    nvt = settings.simulation_settings.equilibration_length_nvt
    assert nvt is not None
    assert nvt.m == pytest.approx(0.1)
    assert settings.simulation_settings.equilibration_length.m == pytest.approx(1.0)
    assert settings.simulation_settings.production_length.m == pytest.approx(5.0)


def test_a_solvated_system_keeps_pme() -> None:
    assert build_settings(SettingsSpec(), solvated=True).forcefield_settings.nonbonded_method == (
        "PME"
    )


def test_vacuum_switches_the_nonbonded_method() -> None:
    """PME needs a periodic box, so the protocol would refuse a vacuum system with it."""
    settings = build_settings(SettingsSpec(), solvated=False)

    assert settings.forcefield_settings.nonbonded_method == "nocutoff"


def test_an_explicit_nonbonded_override_wins() -> None:
    spec = SettingsSpec(overrides={"forcefield_settings.nonbonded_method": "cutoffnonperiodic"})

    settings = build_settings(spec, solvated=False)

    assert settings.forcefield_settings.nonbonded_method == "cutoffnonperiodic"


def test_cost_counts_equilibration_and_production() -> None:
    cost = estimate_md_cost(build_settings(SettingsSpec()), repeats=3)

    assert cost.equilibration_ns == pytest.approx(1.1)
    assert cost.production_ns == pytest.approx(5.0)
    assert cost.total_ns == pytest.approx(18.3)
    assert "3 repeat(s)" in cost.describe()


def test_cost_tolerates_skipped_nvt() -> None:
    settings = build_settings(
        SettingsSpec(overrides={"simulation_settings.equilibration_length_nvt": None})
    )

    assert estimate_md_cost(settings, repeats=1).equilibration_ns == pytest.approx(
        settings.simulation_settings.equilibration_length.m
    )


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="unknown settings preset"):
        build_settings(SettingsSpec.model_construct(preset="fast", overrides={}))


def test_a_repeats_override_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="'protocol_repeats' is not allowed"):
        build_settings(SettingsSpec(overrides={"protocol_repeats": 3}))


@pytest.mark.slow
def test_planning_writes_one_transformation_per_system(planned_campaign: Campaign) -> None:
    plans = sorted(path.name for path in (planned_campaign.directory / "plans").glob("*.json"))

    assert plans == ["md_mini.json"]


@pytest.mark.slow
def test_the_transformation_holds_one_state_twice(planned_campaign: Campaign) -> None:
    """The protocol compares its end states by identity, which must survive the round trip."""
    path = planned_campaign.directory / "plans" / "md_mini.json"

    transformation = cast(Transformation, Transformation.from_json(path))

    assert transformation.stateA is transformation.stateB
    assert transformation.create() is not None


@pytest.mark.slow
def test_the_system_holds_the_protein_the_ligand_and_the_solvent(
    planned_campaign: Campaign,
) -> None:
    path = planned_campaign.directory / "plans" / "md_mini.json"

    transformation = cast(Transformation, Transformation.from_json(path))

    assert "protein" in transformation.stateA.components
    assert "solvent" in transformation.stateA.components
    assert "molecule_0" in transformation.stateA.components


@pytest.mark.slow
def test_the_system_becomes_a_planned_run(planned_campaign: Campaign) -> None:
    assert planned_campaign.run("mini").state is RunState.PLANNED


@pytest.mark.slow
def test_a_vacuum_system_plans(tmp_path: Path) -> None:
    data = md_data(solvent=False, protein=None)
    campaign = Campaign.create(tmp_path / "campaign", validate_request(data))
    prepare_campaign(campaign, check_parameters=False)

    outcome = plan_campaign(campaign)

    assert outcome.failures == {}
    assert outcome.md_runs[0].solvated is False


@pytest.mark.slow
def test_planning_before_preparation_is_refused(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign", validate_request(md_data()))

    outcome = plan_campaign(campaign)

    assert "not prepared yet" in outcome.failures["mini"]
