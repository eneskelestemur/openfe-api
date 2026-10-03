"""Tests for the RBFE request schema: the series input contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from conftest import ANALOGUE_SMILES, REFERENCE_SMILES
from openfe_api.exceptions import InputValidationError
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import load_request, validate_request


def validated(data: dict[str, Any]) -> RbfeRequest:
    """Validate request data and narrow it to an RBFE request.

    Args:
        data: A request dictionary naming ``rbfe`` as its protocol.

    Returns:
        The validated request.
    """
    request = validate_request(data)
    assert isinstance(request, RbfeRequest)
    return request


def test_series_request_validates(series_request_data: dict[str, Any]) -> None:
    request = validated(series_request_data)

    assert request.protocol == "rbfe"
    assert [ligand.name for ligand in request.ligands] == ["lig_ref", "lig_b", "lig_c"]
    assert request.cofactors == []


def test_defaults_match_the_agreed_policy(series_request_data: dict[str, Any]) -> None:
    """The defaults encode the plan: MCS poses, a redundant network, corrected Dq of one."""
    request = validated(series_request_data)

    assert request.alignment.pose == "mcs"
    assert request.similarity.min_core_fraction == pytest.approx(0.6)
    assert request.similarity.allow_ring_break is False
    assert request.network.method == "minimal_redundant"
    assert request.network.redundancy == 2
    assert request.network.mapper == "lomap"
    assert request.charges.correct_single is True
    assert request.charges.allow_multi is False


def test_reference_defaults_to_the_first_ligand(series_request_data: dict[str, Any]) -> None:
    request = validated(series_request_data)

    assert request.reference is None
    assert request.reference_ligand().name == "lig_ref"


def test_reference_may_be_named_explicitly(input_dir: Path) -> None:
    data = {
        "protocol": "rbfe",
        "name": "named_reference",
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES},
            {
                "name": "lig_b",
                "smiles": ANALOGUE_SMILES,
                "structure": str(input_dir / "model.cif"),
                "selector": {"resname": "LIG"},
            },
        ],
        "reference": "lig_b",
    }

    request = validated(data)

    assert request.reference_ligand().name == "lig_b"


def test_unknown_reference_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["reference"] = "lig_absent"

    with pytest.raises(InputValidationError, match="not one of the ligands"):
        validate_request(series_request_data)


def test_reference_without_coordinates_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["reference"] = "lig_b"

    with pytest.raises(InputValidationError, match="must supply coordinates"):
        validate_request(series_request_data)


def test_reference_carrying_no_protein_is_rejected(input_dir: Path) -> None:
    """A bare SDF reference gives a pose but no frame, so the protein must come from a file."""
    data = {
        "protocol": "rbfe",
        "name": "no_protein",
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": str(input_dir / "ligand.sdf")},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
        ],
    }

    with pytest.raises(InputValidationError, match="no source for the protein"):
        validate_request(data)


def test_reference_with_a_protein_path_validates(input_dir: Path) -> None:
    data = {
        "protocol": "rbfe",
        "name": "protein_path",
        "protein": {"path": str(input_dir / "protein.pdb")},
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": str(input_dir / "ligand.sdf")},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
        ],
    }

    request = validated(data)

    assert request.reference_ligand().name == "lig_a"


def test_pose_on_the_reference_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligands"][0]["pose"] = "keep"

    with pytest.raises(InputValidationError, match="must not be set on the reference ligand"):
        validate_request(series_request_data)


def test_pose_defaults_to_mcs_and_keep_is_per_ligand(
    series_request_data: dict[str, Any], input_dir: Path
) -> None:
    """A docked ligand keeps its pose while the rest of the series is built on the core."""
    series_request_data["ligands"][1]["path"] = str(input_dir / "ligand.sdf")
    series_request_data["ligands"][1]["pose"] = "keep"

    request = validated(series_request_data)

    assert request.pose_source(request.ligands[0]) == "keep"
    assert request.pose_source(request.ligands[1]) == "keep"
    assert request.pose_source(request.ligands[2]) == "mcs"


def test_campaign_wide_keep_applies_to_every_ligand(input_dir: Path) -> None:
    """Docking into one fixed receptor means every pose is already in the right frame."""
    data = {
        "protocol": "rbfe",
        "name": "docked_series",
        "protein": {"path": str(input_dir / "protein.pdb")},
        "alignment": {"pose": "keep"},
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": str(input_dir / "ligand.sdf")},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES, "path": str(input_dir / "ligand.sdf")},
        ],
    }

    request = validated(data)

    assert request.pose_source(request.ligands[1]) == "keep"


def test_keep_without_coordinates_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligands"][1]["pose"] = "keep"

    with pytest.raises(InputValidationError, match="supplies no coordinates"):
        validate_request(series_request_data)


def test_campaign_wide_keep_still_needs_coordinates(series_request_data: dict[str, Any]) -> None:
    """A SMILES-only ligand cannot keep a pose it does not have."""
    series_request_data["alignment"] = {"pose": "keep"}

    request = validated(series_request_data)

    assert request.pose_source(request.ligands[1]) == "keep"
    assert request.ligands[1].has_coordinates() is False


def test_both_path_and_structure_are_rejected(
    series_request_data: dict[str, Any], input_dir: Path
) -> None:
    series_request_data["ligands"][1]["path"] = str(input_dir / "ligand.sdf")
    series_request_data["ligands"][1]["structure"] = str(input_dir / "model.cif")
    series_request_data["ligands"][1]["selector"] = {"resname": "LIG"}

    with pytest.raises(InputValidationError, match="sets both 'path' and 'structure'"):
        validate_request(series_request_data)


def test_structure_without_a_selector_is_rejected(
    series_request_data: dict[str, Any], input_dir: Path
) -> None:
    series_request_data["ligands"][1]["structure"] = str(input_dir / "model.cif")

    with pytest.raises(InputValidationError, match="'selector' is required for ligand 'lig_b'"):
        validate_request(series_request_data)


def test_unsupported_structure_suffix_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligands"][0]["structure"] = "model.xyz"

    with pytest.raises(InputValidationError, match="must be a PDB or mmCIF file"):
        validate_request(series_request_data)


def test_duplicate_ligand_names_are_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligands"].append(dict(series_request_data["ligands"][1]))

    with pytest.raises(InputValidationError, match="duplicate ligand names: lig_b"):
        validate_request(series_request_data)


def test_a_single_ligand_is_rejected(series_request_data: dict[str, Any]) -> None:
    """One ligand cannot form an edge, so there is nothing to compute."""
    series_request_data["ligands"] = series_request_data["ligands"][:1]

    with pytest.raises(InputValidationError, match="ligands"):
        validate_request(series_request_data)


def test_blank_smiles_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligands"][1]["smiles"] = "   "

    with pytest.raises(InputValidationError, match="must not be blank"):
        validate_request(series_request_data)


def test_unknown_field_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["ligand"] = []

    with pytest.raises(InputValidationError, match="ligand"):
        validate_request(series_request_data)


def test_cofactor_from_the_reference_structure_validates(
    series_request_data: dict[str, Any],
) -> None:
    series_request_data["cofactors"] = [
        {"name": "NAD", "smiles": "NC(=O)c1ccccc1", "selector": {"resname": "NAD"}}
    ]

    request = validated(series_request_data)

    assert request.cofactors[0].name == "NAD"


def test_cofactor_selector_is_required_from_a_structure(
    series_request_data: dict[str, Any],
) -> None:
    series_request_data["cofactors"] = [{"name": "NAD", "smiles": "NC(=O)c1ccccc1"}]

    with pytest.raises(InputValidationError, match="'selector' is required for cofactor 'NAD'"):
        validate_request(series_request_data)


def test_cofactor_without_a_source_is_rejected(input_dir: Path) -> None:
    data = {
        "protocol": "rbfe",
        "name": "cofactor_no_source",
        "protein": {"path": str(input_dir / "protein.pdb")},
        "cofactors": [{"name": "NAD", "smiles": "NC(=O)c1ccccc1"}],
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": str(input_dir / "ligand.sdf")},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
        ],
    }

    with pytest.raises(InputValidationError, match="no source for cofactor 'NAD'"):
        validate_request(data)


def test_duplicate_cofactor_names_are_rejected(series_request_data: dict[str, Any]) -> None:
    cofactor = {"name": "NAD", "smiles": "NC(=O)c1ccccc1", "selector": {"resname": "NAD"}}
    series_request_data["cofactors"] = [cofactor, dict(cofactor)]

    with pytest.raises(InputValidationError, match="duplicate cofactor names: NAD"):
        validate_request(series_request_data)


def test_explicit_edges_validate(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {
        "method": "explicit",
        "edges": [["lig_ref", "lig_b"], ["lig_b", "lig_c"]],
    }

    request = validated(series_request_data)

    assert request.network.edges == [("lig_ref", "lig_b"), ("lig_b", "lig_c")]


def test_explicit_without_edges_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"method": "explicit"}

    with pytest.raises(InputValidationError, match=r"'network\.edges' is required"):
        validate_request(series_request_data)


def test_edges_without_explicit_are_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"edges": [["lig_ref", "lig_b"]]}

    with pytest.raises(InputValidationError, match="only applies to method 'explicit'"):
        validate_request(series_request_data)


def test_edges_naming_unknown_ligands_are_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {
        "method": "explicit",
        "edges": [["lig_ref", "lig_absent"]],
    }

    with pytest.raises(InputValidationError, match="not in the series: lig_absent"):
        validate_request(series_request_data)


def test_a_self_edge_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"method": "explicit", "edges": [["lig_b", "lig_b"]]}

    with pytest.raises(InputValidationError, match="self-edge"):
        validate_request(series_request_data)


def test_radial_requires_a_central_ligand(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"method": "radial"}

    with pytest.raises(InputValidationError, match=r"'network\.central_ligand' is required"):
        validate_request(series_request_data)


def test_central_ligand_without_radial_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"central_ligand": "lig_ref"}

    with pytest.raises(InputValidationError, match="only applies to method 'radial'"):
        validate_request(series_request_data)


def test_unknown_central_ligand_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"method": "radial", "central_ligand": "lig_absent"}

    with pytest.raises(InputValidationError, match="not one of the ligands"):
        validate_request(series_request_data)


def test_redundancy_on_another_method_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["network"] = {"method": "minimal_spanning", "redundancy": 3}

    with pytest.raises(InputValidationError, match="only applies to method 'minimal_redundant'"):
        validate_request(series_request_data)


def test_core_fraction_above_one_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["similarity"] = {"min_core_fraction": 1.5}

    with pytest.raises(InputValidationError, match="min_core_fraction"):
        validate_request(series_request_data)


def test_runs_are_not_known_before_planning(series_request_data: dict[str, Any]) -> None:
    """Edges depend on the core gate, the mappings and the charge policy, so none exist yet."""
    request = validated(series_request_data)

    assert request.initial_run_names() == []


def test_input_paths_are_unique_and_ordered(series_request_data: dict[str, Any]) -> None:
    request = validated(series_request_data)

    paths = request.input_paths()

    assert [path.name for path in paths] == ["model.cif"]


def test_load_request_resolves_relative_paths(
    input_dir: Path, write_request: Callable[..., Path]
) -> None:
    data = {
        "protocol": "rbfe",
        "name": "relative_series",
        "protein": {"path": "inputs/protein.pdb"},
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": "inputs/ligand.sdf"},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
        ],
    }
    request_file = write_request(data)

    request = load_request(request_file)

    assert isinstance(request, RbfeRequest)
    ligand_path = request.ligands[0].path
    assert ligand_path is not None
    assert ligand_path == input_dir / "ligand.sdf"


def test_load_request_reports_missing_input_files(write_request: Callable[..., Path]) -> None:
    data = {
        "protocol": "rbfe",
        "name": "missing_series",
        "protein": {"path": "inputs/absent.pdb"},
        "ligands": [
            {"name": "lig_a", "smiles": REFERENCE_SMILES, "path": "inputs/absent.sdf"},
            {"name": "lig_b", "smiles": ANALOGUE_SMILES},
        ],
    }
    request_file = write_request(data)

    with pytest.raises(InputValidationError, match="input files not found"):
        load_request(request_file)


def test_an_unknown_protocol_is_rejected(series_request_data: dict[str, Any]) -> None:
    series_request_data["protocol"] = "fep"

    with pytest.raises(InputValidationError, match="does not match any of the expected tags"):
        validate_request(series_request_data)


def test_abfe_fields_are_rejected_in_an_rbfe_request(series_request_data: dict[str, Any]) -> None:
    """The discriminator picks one shape, so a field from the other protocol is an error."""
    series_request_data["complexes"] = []

    with pytest.raises(InputValidationError, match="complexes"):
        validate_request(series_request_data)


@pytest.mark.parametrize(
    "network",
    [
        {"method": "minimal_redundant", "redundancy": 3},
        {"method": "minimal_spanning"},
        {"method": "radial", "central_ligand": "lig_ref"},
        {"method": "explicit", "edges": [["lig_ref", "lig_b"]]},
    ],
    ids=["redundant", "spanning", "radial", "explicit"],
)
def test_every_network_method_survives_a_round_trip(
    series_request_data: dict[str, Any], network: dict[str, Any]
) -> None:
    """A manifest serializes every field, so a validator must not reject its own output."""
    series_request_data["network"] = network
    request = validated(series_request_data)

    reloaded = validate_request(request.model_dump(mode="json"))

    assert isinstance(reloaded, RbfeRequest)
    assert reloaded.network.method == network["method"]
