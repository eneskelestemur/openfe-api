"""Tests for choosing network edges and applying the charge policy."""

from __future__ import annotations

import pytest
from gufe import SmallMoleculeComponent
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.mcs import find_core
from openfe_api.prep.pose import generate_pose
from openfe_api.protocols.network import build_mappers, plan_network, read_network
from openfe_api.schema.rbfe import AlignmentSpec, ChargeSpec, NetworkSpec

REFERENCE = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
ETHYL = "CCc1ccc(cc1)C(=O)Nc1ccccc1"
FLUORO = "Fc1ccc(cc1)C(=O)Nc1ccccc1"
ANION = "[O-]C(=O)c1ccc(cc1)C(=O)Nc1ccccc1"
DIANION = "[O-]C(=O)c1ccc(cc1)C(=O)Nc1ccc([O-])cc1"


def reference_molecule() -> Chem.Mol:
    """Build the reference ligand with a minimized pose.

    Returns:
        The reference ligand holding one conformer.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(REFERENCE))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=3)
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule)
    return molecule


def aligned_series(smiles: dict[str, str]) -> list[SmallMoleculeComponent]:
    """Build a series placed on the reference ligand's common core.

    Lomap maps in 3D with a 1 A cutoff, so an unaligned series scores near zero and the
    planner cannot tell good edges from bad ones.

    Args:
        smiles: Ligand names mapped to their SMILES.

    Returns:
        The reference first, then each aligned ligand.
    """
    reference = reference_molecule()
    components = [SmallMoleculeComponent(reference, name="ref")]
    for name, pattern in smiles.items():
        probe = Chem.AddHs(Chem.MolFromSmiles(pattern))
        core = find_core(reference, probe)
        pose = generate_pose(probe, reference, core, AlignmentSpec(n_conformers=10))
        components.append(SmallMoleculeComponent(pose.molecule, name=name))
    return components


@pytest.fixture(scope="module")
def neutral_series() -> list[SmallMoleculeComponent]:
    """Return three neutral ligands sharing one core.

    Returns:
        The aligned series.
    """
    return aligned_series({"ethyl": ETHYL, "fluoro": FLUORO})


@pytest.fixture(scope="module")
def charged_series() -> list[SmallMoleculeComponent]:
    """Return a series spanning net charges 0, -1 and -2.

    Returns:
        The aligned series.
    """
    return aligned_series({"ethyl": ETHYL, "anion": ANION, "dianion": DIANION})


def test_lomap_is_the_default_mapper() -> None:
    assert len(build_mappers("lomap")) == 1
    assert len(build_mappers("kartograf")) == 1
    assert len(build_mappers("both")) == 2


def test_an_unknown_mapper_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="unknown atom mapper"):
        build_mappers("perses")


def test_a_spanning_network_connects_every_ligand(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(neutral_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())

    assert len(planned.decisions) == len(neutral_series) - 1
    assert planned.unreachable == []
    assert planned.dropped == {}


def test_a_redundant_network_adds_edges(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    spanning = plan_network(neutral_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())
    redundant = plan_network(neutral_series, NetworkSpec(method="minimal_redundant"), ChargeSpec())

    assert len(redundant.decisions) > len(spanning.decisions)


def test_alignment_makes_the_mapping_scores_usable(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    """Lomap maps in 3D, so the scores only mean anything once the series shares a frame."""
    planned = plan_network(neutral_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())

    assert all(decision.score > 0.5 for decision in planned.decisions)


def test_explicit_edges_are_run_exactly(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    spec = NetworkSpec(method="explicit", edges=[("ref", "fluoro")])

    planned = plan_network(neutral_series, spec, ChargeSpec())

    assert [decision.name for decision in planned.decisions] == ["ref_to_fluoro"]
    assert planned.unreachable == ["ethyl"]


def test_a_radial_network_uses_the_hub(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    spec = NetworkSpec(method="radial", central_ligand="ref")

    planned = plan_network(neutral_series, spec, ChargeSpec())

    assert all("ref" in decision.name for decision in planned.decisions)


def test_a_single_charge_change_is_corrected(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(charged_series, NetworkSpec(method="minimal_redundant"), ChargeSpec())

    changing = [decision for decision in planned.decisions if decision.charge_difference != 0]
    assert changing
    assert all(abs(decision.charge_difference) == 1 for decision in changing)
    assert all(decision.corrected for decision in changing)


def test_correction_can_be_turned_off_with_a_warning(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(
        charged_series,
        NetworkSpec(method="minimal_redundant"),
        ChargeSpec(correct_single=False),
    )

    changing = [decision for decision in planned.decisions if decision.charge_difference != 0]
    assert changing
    assert not any(decision.corrected for decision in changing)
    assert any("no correction will be applied" in w for w in planned.warnings)


def test_a_large_charge_change_is_routed_around(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    """A difference of two has no correction, so the planner must not use that pair."""
    planned = plan_network(charged_series, NetworkSpec(method="minimal_redundant"), ChargeSpec())

    assert all(abs(decision.charge_difference) <= 1 for decision in planned.decisions)
    assert "ref_to_dianion" not in [decision.name for decision in planned.decisions]


def test_the_network_stays_connected_around_a_forbidden_pair(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    """Routing around a pair must not orphan a ligand that another path can reach."""
    planned = plan_network(charged_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())

    assert planned.unreachable == []
    assert {decision.ligand_a for decision in planned.decisions} | {
        decision.ligand_b for decision in planned.decisions
    } == {component.name for component in charged_series}


def test_a_large_charge_change_can_be_forced(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    """OpenFE reports the difference as state A minus state B, so losing 2e reads as +2."""
    planned = plan_network(
        charged_series,
        NetworkSpec(method="explicit", edges=[("ref", "dianion")]),
        ChargeSpec(allow_multi=True),
    )

    assert [decision.charge_difference for decision in planned.decisions] == [2]
    assert any("UNCORRECTED" in warning for warning in planned.warnings)


def test_a_forbidden_explicit_edge_is_dropped(
    charged_series: list[SmallMoleculeComponent],
) -> None:
    with pytest.raises(InputValidationError, match="no edge survived planning"):
        plan_network(
            charged_series,
            NetworkSpec(method="explicit", edges=[("ref", "dianion")]),
            ChargeSpec(),
        )


def test_the_network_round_trips_through_graphml(
    neutral_series: list[SmallMoleculeComponent], tmp_path
) -> None:
    """A saved network must replan to exactly the same edges."""
    planned = plan_network(neutral_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())
    path = tmp_path / "network.graphml"
    path.write_text(planned.network.to_graphml(), encoding="utf-8")

    reloaded = read_network(path)

    assert {(edge.componentA.name, edge.componentB.name) for edge in reloaded.edges} == {
        (edge.componentA.name, edge.componentB.name) for edge in planned.network.edges
    }


def test_an_unreadable_network_is_rejected(tmp_path) -> None:
    path = tmp_path / "network.graphml"
    path.write_text("not a network", encoding="utf-8")

    with pytest.raises(InputValidationError, match="could not be read as a ligand network"):
        read_network(path)


def test_mapping_lookup_finds_the_edge(
    neutral_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(neutral_series, NetworkSpec(method="minimal_spanning"), ChargeSpec())

    mapping = planned.mapping(planned.decisions[0])

    assert mapping.componentA.name == planned.decisions[0].ligand_a
    assert mapping.componentB.name == planned.decisions[0].ligand_b


R_ISOMER = "C[C@H](O)c1ccc(cc1)C(=O)Nc1ccccc1"
S_ISOMER = "C[C@@H](O)c1ccc(cc1)C(=O)Nc1ccccc1"


@pytest.fixture(scope="module")
def stereo_series() -> list[SmallMoleculeComponent]:
    """Return a series holding both configurations of one stereocenter.

    Returns:
        The reference, an ethyl analogue, and the two stereoisomers.
    """
    return aligned_series({"ethyl": ETHYL, "r_form": R_ISOMER, "s_form": S_ISOMER})


def test_a_stereoisomer_pair_is_never_an_edge(
    stereo_series: list[SmallMoleculeComponent],
) -> None:
    """A perfect mapping makes this the planner's favorite edge, and it is unrunnable."""
    planned = plan_network(stereo_series, NetworkSpec(), ChargeSpec())

    pairs = {frozenset({decision.ligand_a, decision.ligand_b}) for decision in planned.decisions}
    assert frozenset({"r_form", "s_form"}) not in pairs


def test_the_stereoisomer_drop_names_septop(
    stereo_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(
        stereo_series,
        NetworkSpec(method="explicit", edges=[("r_form", "s_form"), ("ref", "ethyl")]),
        ChargeSpec(),
    )

    reason = planned.dropped["r_form_to_s_form"]
    assert "differ only in stereochemistry" in reason
    assert "SepTop" in reason


def test_the_rest_of_the_network_survives_a_stereoisomer_drop(
    stereo_series: list[SmallMoleculeComponent],
) -> None:
    planned = plan_network(
        stereo_series,
        NetworkSpec(method="explicit", edges=[("r_form", "s_form"), ("ref", "ethyl")]),
        ChargeSpec(),
    )

    assert [decision.name for decision in planned.decisions] == ["ref_to_ethyl"]
