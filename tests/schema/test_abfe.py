"""Tests for the ABFE request contract: complexes, their ligand and cofactors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openfe_api.exceptions import InputValidationError
from openfe_api.schema.abfe import AbfeRequest, ComplexSpec
from openfe_api.schema.request import validate_request


def validated(data: dict[str, Any]) -> AbfeRequest:
    """Validate request data and narrow it to an ABFE request.

    Args:
        data: A request dictionary naming ``abfe`` as its protocol.

    Returns:
        The validated request.
    """
    request = validate_request(data)
    assert isinstance(request, AbfeRequest)
    return request


def test_combined_request_validates(combined_request_data: dict[str, Any]) -> None:
    request = validated(combined_request_data)

    assert request.protocol == "abfe"
    assert request.complexes[0].extra_ligand_copies == "drop"
    assert request.complexes[0].ligand_name() == "complex_one"
    assert request.neutralize_ligands is False


def test_split_request_validates(split_request_data: dict[str, Any]) -> None:
    request = validated(split_request_data)

    complex_spec = request.complexes[0]
    assert complex_spec.structure is None
    assert complex_spec.extra_ligand_copies is None
    assert complex_spec.cofactors == []


def test_ligand_name_falls_back_to_complex_name(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"][0]["ligand"]["name"] = "explicit_name"

    request = validated(split_request_data)

    assert request.complexes[0].ligand_name() == "explicit_name"


def test_extra_ligand_copies_is_required_for_combined_structure(
    combined_request_data: dict[str, Any],
) -> None:
    del combined_request_data["complexes"][0]["extra_ligand_copies"]

    with pytest.raises(InputValidationError, match="extra_ligand_copies' is required"):
        validate_request(combined_request_data)


def test_extra_ligand_copies_is_rejected_for_split_files(
    split_request_data: dict[str, Any],
) -> None:
    split_request_data["complexes"][0]["extra_ligand_copies"] = "drop"

    with pytest.raises(InputValidationError, match="only applies to a ligand read from a combined"):
        validate_request(split_request_data)


def test_ligand_selector_required_for_combined_structure(
    combined_request_data: dict[str, Any],
) -> None:
    del combined_request_data["complexes"][0]["ligand"]["selector"]

    with pytest.raises(InputValidationError, match=r"ligand\.selector' is required"):
        validate_request(combined_request_data)


def test_cofactor_selector_required_for_combined_structure(
    combined_request_data: dict[str, Any],
) -> None:
    del combined_request_data["complexes"][0]["cofactors"][0]["selector"]

    with pytest.raises(InputValidationError, match="required for cofactor 'NAD_C'"):
        validate_request(combined_request_data)


def test_missing_ligand_source_is_rejected(split_request_data: dict[str, Any]) -> None:
    del split_request_data["complexes"][0]["ligand"]["path"]

    with pytest.raises(InputValidationError, match="no source for the ligand"):
        validate_request(split_request_data)


def test_missing_protein_source_is_rejected(split_request_data: dict[str, Any]) -> None:
    del split_request_data["complexes"][0]["protein"]["path"]

    with pytest.raises(InputValidationError, match="no source for the protein"):
        validate_request(split_request_data)


def test_smiles_is_required(split_request_data: dict[str, Any]) -> None:
    del split_request_data["complexes"][0]["ligand"]["smiles"]

    with pytest.raises(InputValidationError, match="smiles"):
        validate_request(split_request_data)


def test_blank_smiles_is_rejected(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"][0]["ligand"]["smiles"] = "   "

    with pytest.raises(InputValidationError, match="must not be blank"):
        validate_request(split_request_data)


def test_unknown_field_is_rejected(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"][0]["ligands"] = []

    with pytest.raises(InputValidationError, match="ligands"):
        validate_request(split_request_data)


def test_duplicate_complex_names_are_rejected(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"].append(dict(split_request_data["complexes"][0]))

    with pytest.raises(InputValidationError, match="duplicate complex names: complex_one"):
        validate_request(split_request_data)


def test_duplicate_cofactor_names_are_rejected(combined_request_data: dict[str, Any]) -> None:
    cofactors = combined_request_data["complexes"][0]["cofactors"]
    cofactors.append(dict(cofactors[0]))

    with pytest.raises(InputValidationError, match="duplicate cofactor names: NAD_C"):
        validate_request(combined_request_data)


def test_empty_complex_list_is_rejected(split_request_data: dict[str, Any]) -> None:
    split_request_data["complexes"] = []

    with pytest.raises(InputValidationError, match="complexes"):
        validate_request(split_request_data)


def test_unsupported_structure_suffix_is_rejected(combined_request_data: dict[str, Any]) -> None:
    combined_request_data["complexes"][0]["structure"] = "model.xyz"

    with pytest.raises(InputValidationError, match="must be a PDB or mmCIF file"):
        validate_request(combined_request_data)


def test_complex_input_paths_include_every_source(
    input_dir: Path, combined_request_data: dict[str, Any]
) -> None:
    complex_data = combined_request_data["complexes"][0]
    complex_data["cofactors"][0] = {
        "name": "NAD_C",
        "path": str(input_dir / "cofactor.sdf"),
        "smiles": "NC(=O)c1ccccc1",
    }

    complex_spec = ComplexSpec.model_validate(complex_data)

    assert [path.name for path in complex_spec.input_paths()] == ["model.cif", "cofactor.sdf"]
