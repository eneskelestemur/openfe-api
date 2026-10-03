"""Tests for the SepTop request schema: screening, pose sources and the star network."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from conftest import HOP_SMILES, REFERENCE_SMILES
from openfe_api.exceptions import InputValidationError
from openfe_api.schema.request import validate_request
from openfe_api.schema.septop import SepTopRequest


def validated(data: dict[str, Any]) -> SepTopRequest:
    """Validate request data and narrow it to a SepTop request.

    Args:
        data: A request dictionary naming ``septop`` as its protocol.

    Returns:
        The validated request.
    """
    request = validate_request(data)
    assert isinstance(request, SepTopRequest)
    return request


def test_septop_request_validates(septop_request_data: dict[str, Any]) -> None:
    request = validated(septop_request_data)

    assert request.protocol == "septop"
    assert [ligand.name for ligand in request.ligands] == ["lig_ref", "lig_b", "lig_hop"]


def test_defaults_are_built_for_scaffold_hops(septop_request_data: dict[str, Any]) -> None:
    """A hop shares no core, so neither the gate nor the placement may insist on one."""
    request = validated(septop_request_data)

    assert request.alignment.pose == "auto"
    assert request.screening.min_core_fraction is None
    assert request.screening.policy == "warn"
    assert request.network.method == "radial"


def test_the_hub_defaults_to_the_reference(septop_request_data: dict[str, Any]) -> None:
    request = validated(septop_request_data)

    assert request.network.central_ligand is None
    assert request.hub() == "lig_ref"


def test_the_hub_may_be_named_explicitly(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"method": "radial", "central_ligand": "lig_b"}

    assert validated(septop_request_data).hub() == "lig_b"


def test_an_unknown_hub_is_rejected(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"central_ligand": "absent"}

    with pytest.raises(InputValidationError, match="not one of the ligands"):
        validate_request(septop_request_data)


def test_the_reference_keeps_its_pose(septop_request_data: dict[str, Any]) -> None:
    request = validated(septop_request_data)

    assert request.pose_source(request.reference_ligand()) == "keep"


def test_a_ligand_without_a_pose_of_its_own_follows_the_campaign_default(
    septop_request_data: dict[str, Any],
) -> None:
    request = validated(septop_request_data)

    assert request.pose_source(request.ligands[1]) == "auto"


def test_a_ligand_may_override_the_campaign_default(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["ligands"][2]["pose"] = "shape"
    request = validated(septop_request_data)

    assert request.pose_source(request.ligands[2]) == "shape"


def test_pose_keep_needs_coordinates(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["ligands"][1]["pose"] = "keep"

    with pytest.raises(InputValidationError, match="supplies no coordinates"):
        validate_request(septop_request_data)


def test_pose_must_not_be_set_on_the_reference(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["ligands"][0]["pose"] = "keep"

    with pytest.raises(InputValidationError, match="must not be set on the reference"):
        validate_request(septop_request_data)


def test_an_unknown_pose_source_is_rejected(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["alignment"] = {"pose": "dock"}

    with pytest.raises(InputValidationError, match=re.escape("alignment.pose")):
        validate_request(septop_request_data)


def test_shape_is_not_an_rbfe_pose_source(series_request_data: dict[str, Any]) -> None:
    """``shape`` belongs to SepTop; RBFE places on the common core or keeps the pose."""
    series_request_data["alignment"] = {"pose": "shape"}

    with pytest.raises(InputValidationError, match=re.escape("alignment.pose")):
        validate_request(series_request_data)


def test_explicit_needs_edges(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"method": "explicit"}

    with pytest.raises(InputValidationError, match=re.escape("'network.edges' is required")):
        validate_request(septop_request_data)


def test_edges_need_the_explicit_method(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"edges": [["lig_ref", "lig_b"]]}

    with pytest.raises(InputValidationError, match="only applies to method 'explicit'"):
        validate_request(septop_request_data)


def test_a_hub_makes_no_sense_for_explicit_edges(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {
        "method": "explicit",
        "edges": [["lig_ref", "lig_b"]],
        "central_ligand": "lig_ref",
    }

    with pytest.raises(InputValidationError, match="does not apply to method 'explicit'"):
        validate_request(septop_request_data)


def test_a_self_edge_is_rejected(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"method": "explicit", "edges": [["lig_b", "lig_b"]]}

    with pytest.raises(InputValidationError, match="self-edge"):
        validate_request(septop_request_data)


def test_an_edge_naming_an_absent_ligand_is_rejected(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"method": "explicit", "edges": [["lig_ref", "absent"]]}

    with pytest.raises(InputValidationError, match="not in the series"):
        validate_request(septop_request_data)


def test_redundancy_is_available_as_its_own_method(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"method": "radial_redundant"}

    assert validated(septop_request_data).network.method == "radial_redundant"


def test_a_size_ratio_must_exceed_one(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["screening"] = {"max_size_ratio": 1.0}

    with pytest.raises(InputValidationError, match=re.escape("screening.max_size_ratio")):
        validate_request(septop_request_data)


def test_the_screening_policy_is_limited_to_warn_and_drop(
    septop_request_data: dict[str, Any],
) -> None:
    septop_request_data["screening"] = {"policy": "ignore"}

    with pytest.raises(InputValidationError, match=re.escape("screening.policy")):
        validate_request(septop_request_data)


def test_rbfe_fields_are_rejected(septop_request_data: dict[str, Any]) -> None:
    """SepTop has no charge correction, so its request must not accept one."""
    septop_request_data["charges"] = {"correct_single": False}

    with pytest.raises(InputValidationError, match="charges"):
        validate_request(septop_request_data)


def test_a_lomap_mapper_is_not_a_septop_setting(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["network"] = {"mapper": "lomap"}

    with pytest.raises(InputValidationError, match="mapper"):
        validate_request(septop_request_data)


def test_two_ligands_are_required(input_dir: Path) -> None:
    data = {
        "protocol": "septop",
        "name": "one_ligand",
        "ligands": [
            {
                "name": "lig_ref",
                "smiles": REFERENCE_SMILES,
                "structure": str(input_dir / "model.cif"),
                "selector": {"chain": "B"},
            }
        ],
    }

    with pytest.raises(InputValidationError, match="ligands"):
        validate_request(data)


def test_the_reference_must_carry_coordinates(input_dir: Path) -> None:
    data = {
        "protocol": "septop",
        "name": "no_reference_pose",
        "protein": {"path": str(input_dir / "protein.pdb")},
        "ligands": [
            {"name": "lig_ref", "smiles": REFERENCE_SMILES},
            {"name": "lig_hop", "smiles": HOP_SMILES},
        ],
    }

    with pytest.raises(InputValidationError, match="must supply coordinates"):
        validate_request(data)


def test_duplicate_ligand_names_are_rejected(septop_request_data: dict[str, Any]) -> None:
    septop_request_data["ligands"][2]["name"] = "lig_b"

    with pytest.raises(InputValidationError, match="duplicate ligand names"):
        validate_request(septop_request_data)
