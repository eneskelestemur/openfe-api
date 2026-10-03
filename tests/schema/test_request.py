"""Tests for loading a request from YAML and resolving its input paths."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from openfe_api.exceptions import InputValidationError
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.request import load_request, validate_request


def test_from_yaml_resolves_relative_paths(
    input_dir: Path,
    write_request: Callable[..., Path],
) -> None:
    data = {
        "protocol": "abfe",
        "name": "relative_campaign",
        "complexes": [
            {
                "name": "complex_one",
                "protein": {"path": "inputs/protein.pdb"},
                "ligand": {"path": "inputs/ligand.sdf", "smiles": "CCO"},
            }
        ],
    }
    request_file = write_request(data)

    request = load_request(request_file)

    assert isinstance(request, AbfeRequest)
    ligand_path = request.complexes[0].ligand.path
    assert ligand_path is not None
    assert ligand_path.is_absolute()
    assert ligand_path == input_dir / "ligand.sdf"


def test_from_yaml_reports_missing_input_files(write_request: Callable[..., Path]) -> None:
    data = {
        "protocol": "abfe",
        "name": "missing_campaign",
        "complexes": [
            {
                "name": "complex_one",
                "protein": {"path": "inputs/absent.pdb"},
                "ligand": {"path": "inputs/absent.sdf", "smiles": "CCO"},
            }
        ],
    }
    request_file = write_request(data)

    with pytest.raises(InputValidationError, match="input files not found"):
        load_request(request_file)


def test_from_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="request file not found"):
        load_request(tmp_path / "absent.yaml")


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    request_file = tmp_path / "request.yaml"
    request_file.write_text("- not a mapping\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="expected a YAML mapping"):
        load_request(request_file)


def test_from_yaml_rejects_invalid_yaml(tmp_path: Path) -> None:
    request_file = tmp_path / "request.yaml"
    request_file.write_text("protocol: [unclosed\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="invalid YAML"):
        load_request(request_file)


def test_input_paths_are_unique_and_ordered(combined_request_data: dict[str, Any]) -> None:
    request = validate_request(combined_request_data)

    paths = request.input_paths()

    assert len(paths) == 1
    assert paths[0].name == "model.cif"
