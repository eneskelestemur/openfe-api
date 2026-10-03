"""Tests for planning SepTop transformations and the star networks they run in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from gufe import SmallMoleculeComponent, Transformation
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from helpers import mini_series_data
from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols.runner import plan_campaign
from openfe_api.protocols.septop import SepTopEdge, build_network, build_settings
from openfe_api.schema.common import SettingsSpec
from openfe_api.schema.request import validate_request
from openfe_api.schema.septop import SepTopNetworkSpec

FAST_CHARGES = {"partial_charge_settings.partial_charge_method": "nagl"}
FUSED_AROMATICS = "c1ccc2cc3ccccc3cc2c1"


def component(smiles: str, name: str) -> SmallMoleculeComponent:
    """Build a named component holding one conformer.

    Args:
        smiles: The molecule's SMILES.
        name: The component's name.

    Returns:
        The component.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=7)
    return SmallMoleculeComponent(molecule, name=name)


@pytest.fixture
def series() -> list[SmallMoleculeComponent]:
    """Return four runnable ligands.

    Returns:
        A hub and three other ligands.
    """
    return [
        component("Cc1ccc(cc1)C(=O)Nc1ccccc1", "hub"),
        component("CCc1ccc(cc1)C(=O)Nc1ccccc1", "a"),
        component("O=C(N)C1CCOCC1", "b"),
        component("Fc1ccc(cc1)C(=O)Nc1ccccc1", "c"),
    ]


@pytest.fixture
def septop_data() -> dict[str, Any]:
    """Return a two-ligand SepTop request on the trimmed complex fixture.

    Returns:
        Request data for the smallest network that has an edge.
    """
    return mini_series_data(
        "septop",
        "mini_septop",
        protein={"chains": ["A"]},
        settings={"preset": "screening", "overrides": dict(FAST_CHARGES)},
        execution={"repeats": 1},
    )


@pytest.fixture
def planned_campaign(septop_data: dict[str, Any], tmp_path: Path) -> Campaign:
    """Create, prepare and plan a small SepTop campaign.

    Args:
        septop_data: The request data.
        tmp_path: Pytest temporary directory.

    Returns:
        The planned campaign.
    """
    campaign = Campaign.create(tmp_path / "campaign", validate_request(septop_data))
    prepare_campaign(campaign, check_parameters=False)
    plan_campaign(campaign)
    return campaign


def test_a_star_connects_every_ligand_to_the_hub(series: list[SmallMoleculeComponent]) -> None:
    network = build_network(series, "hub", SepTopNetworkSpec())

    assert [(edge.ligand_a, edge.ligand_b) for edge in network.edges] == [
        ("hub", "a"),
        ("hub", "b"),
        ("hub", "c"),
    ]


def test_a_star_needs_one_edge_per_other_ligand(series: list[SmallMoleculeComponent]) -> None:
    """The cheapest connected network, and every edge holds the trusted pose."""
    network = build_network(series, "hub", SepTopNetworkSpec())

    assert len(network.edges) == len(series) - 1


def test_the_hub_may_be_any_ligand(series: list[SmallMoleculeComponent]) -> None:
    network = build_network(series, "b", SepTopNetworkSpec())

    assert all(edge.ligand_a == "b" for edge in network.edges)


def test_redundancy_closes_cycles_through_the_hub(
    series: list[SmallMoleculeComponent],
) -> None:
    network = build_network(series, "hub", SepTopNetworkSpec(method="radial_redundant"))

    assert [(edge.ligand_a, edge.ligand_b) for edge in network.edges] == [
        ("hub", "a"),
        ("hub", "b"),
        ("hub", "c"),
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
    ]


def test_every_ligand_has_two_paths_in_a_redundant_network(
    series: list[SmallMoleculeComponent],
) -> None:
    network = build_network(series, "hub", SepTopNetworkSpec(method="radial_redundant"))

    for ligand in series:
        touching = [edge for edge in network.edges if ligand.name in (edge.ligand_a, edge.ligand_b)]
        assert len(touching) >= 2


def test_two_spokes_are_joined_once(series: list[SmallMoleculeComponent]) -> None:
    """A ring of two spokes is one edge, not the same edge twice."""
    network = build_network(series[:3], "hub", SepTopNetworkSpec(method="radial_redundant"))

    assert [(edge.ligand_a, edge.ligand_b) for edge in network.edges] == [
        ("hub", "a"),
        ("hub", "b"),
        ("a", "b"),
    ]


def test_explicit_edges_are_run_as_given(series: list[SmallMoleculeComponent]) -> None:
    spec = SepTopNetworkSpec(method="explicit", edges=[("a", "c"), ("c", "b")])

    network = build_network(series, "hub", spec)

    assert network.edges == [SepTopEdge("a", "c"), SepTopEdge("c", "b")]


def test_an_unrestrainable_ligand_is_dropped(series: list[SmallMoleculeComponent]) -> None:
    """Three fused aromatic rings break OpenFE's restraint search, which runs per ligand."""
    series.append(component(FUSED_AROMATICS, "fused"))

    network = build_network(series, "hub", SepTopNetworkSpec())

    assert "Boresch restraint search" in network.dropped["fused"]
    assert all("fused" not in (edge.ligand_a, edge.ligand_b) for edge in network.edges)


def test_an_unrestrainable_hub_stops_planning(series: list[SmallMoleculeComponent]) -> None:
    """Every edge touches the hub, so a hub that cannot be restrained leaves nothing to run."""
    series.append(component(FUSED_AROMATICS, "fused"))

    with pytest.raises(InputValidationError, match="hub ligand 'fused' was excluded"):
        build_network(series, "fused", SepTopNetworkSpec())


def test_an_unreachable_ligand_is_reported(series: list[SmallMoleculeComponent]) -> None:
    spec = SepTopNetworkSpec(method="explicit", edges=[("hub", "a")])

    network = build_network(series, "hub", spec)

    assert sorted(network.unreachable) == ["b", "c"]
    assert any("no edge connects" in warning for warning in network.warnings)


def test_build_settings_forces_one_repeat() -> None:
    assert build_settings(SettingsSpec()).protocol_repeats == 1


def test_screening_shortens_both_phases() -> None:
    default = build_settings(SettingsSpec())
    screening = build_settings(SettingsSpec(preset="screening"))

    assert (
        screening.complex_simulation_settings.production_length.m
        < default.complex_simulation_settings.production_length.m
    )
    assert (
        screening.solvent_simulation_settings.production_length.m
        < default.solvent_simulation_settings.production_length.m
    )


def test_the_default_window_counts_come_from_openfe() -> None:
    """The solvent phase runs more windows than the complex phase, unlike ABFE."""
    settings = build_settings(SettingsSpec())

    assert settings.solvent_simulation_settings.n_replicas == 27
    assert settings.complex_simulation_settings.n_replicas == 19


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="unknown settings preset"):
        build_settings(SettingsSpec.model_construct(preset="fast", overrides={}))


def test_a_repeats_override_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="'protocol_repeats' is not allowed"):
        build_settings(SettingsSpec(overrides={"protocol_repeats": 3}))


@pytest.mark.slow
def test_planning_writes_one_transformation_per_edge(planned_campaign: Campaign) -> None:
    """Both phases live in one transformation: the protocol builds them from one state pair."""
    plans = sorted(path.name for path in (planned_campaign.directory / "plans").glob("*.json"))

    assert plans == ["network.json", "septop_reference_to_fluoro.json"]


@pytest.mark.slow
def test_the_transformation_holds_the_protein_in_both_states(
    planned_campaign: Campaign,
) -> None:
    """SepTop validates a protein in both end states and strips it for the solvent phase."""
    path = planned_campaign.directory / "plans" / "septop_reference_to_fluoro.json"

    transformation = cast(Transformation, Transformation.from_json(path))

    for state in (transformation.stateA, transformation.stateB):
        assert "protein" in state.components
        assert "solvent" in state.components


@pytest.mark.slow
def test_the_edge_becomes_a_planned_run(planned_campaign: Campaign) -> None:
    record = planned_campaign.run("reference_to_fluoro")

    assert record.state is RunState.PLANNED


@pytest.mark.slow
def test_the_network_summary_names_the_protocol(planned_campaign: Campaign) -> None:
    summary = json.loads(
        (planned_campaign.directory / "plans" / "network.json").read_text(encoding="utf-8")
    )

    assert summary["protocol"] == "septop"
    assert summary["hub"] == "reference"
    assert [edge["name"] for edge in summary["edges"]] == ["reference_to_fluoro"]


@pytest.mark.slow
def test_no_graphml_is_written(planned_campaign: Campaign) -> None:
    """A SepTop edge has no atom mapping, so a ligand network file would be empty of them."""
    assert not (planned_campaign.directory / "plans" / "network.graphml").exists()
