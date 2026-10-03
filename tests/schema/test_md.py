"""Tests for the plain MD request schema: systems, their contents and the solvent switch."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from conftest import LIGAND_SMILES
from openfe_api.exceptions import InputValidationError
from openfe_api.schema.md import MdRequest
from openfe_api.schema.request import validate_request


def validated(data: dict[str, Any]) -> MdRequest:
    """Validate request data and narrow it to an MD request.

    Args:
        data: A request dictionary naming ``md`` as its protocol.

    Returns:
        The validated request.
    """
    request = validate_request(data)
    assert isinstance(request, MdRequest)
    return request


def test_a_complex_system_validates(md_request_data: dict[str, Any]) -> None:
    request = validated(md_request_data)

    assert request.protocol == "md"
    assert [system.name for system in request.systems] == ["complex_one"]
    assert request.initial_run_names() == ["complex_one"]


def test_solvent_is_on_by_default(md_request_data: dict[str, Any]) -> None:
    assert validated(md_request_data).systems[0].solvent is True


def test_several_ligands_share_one_box(md_request_data: dict[str, Any]) -> None:
    """A plain MD run of a complex holding more than one ligand is ordinary work."""
    md_request_data["systems"][0]["ligands"].append(
        {"name": "second", "selector": {"chain": "F"}, "smiles": LIGAND_SMILES}
    )

    request = validated(md_request_data)

    assert [ligand.name for ligand in request.systems[0].ligands] == ["lig", "second"]


def test_an_apo_protein_is_a_system(input_dir: Path) -> None:
    data = {
        "protocol": "md",
        "name": "apo",
        "systems": [{"name": "apo_protein", "protein": {"path": str(input_dir / "protein.pdb")}}],
    }

    request = validated(data)

    assert request.systems[0].ligands == []
    assert request.systems[0].protein is not None


def test_a_ligand_in_water_is_a_system(input_dir: Path) -> None:
    data = {
        "protocol": "md",
        "name": "solvated_ligand",
        "systems": [
            {
                "name": "lig_in_water",
                "ligands": [
                    {"name": "lig", "path": str(input_dir / "ligand.sdf"), "smiles": LIGAND_SMILES}
                ],
            }
        ],
    }

    request = validated(data)

    assert request.systems[0].protein is None


def test_an_empty_system_is_rejected() -> None:
    data = {"protocol": "md", "name": "empty", "systems": [{"name": "nothing"}]}

    with pytest.raises(InputValidationError, match="holds nothing to simulate"):
        validate_request(data)


def test_vacuum_is_allowed(md_request_data: dict[str, Any]) -> None:
    """Planning sets the nonbonded method for it; the request only has to ask."""
    md_request_data["systems"][0]["solvent"] = False

    assert validated(md_request_data).systems[0].solvent is False


def test_a_molecule_from_the_structure_needs_a_selector(md_request_data: dict[str, Any]) -> None:
    del md_request_data["systems"][0]["ligands"][0]["selector"]

    with pytest.raises(InputValidationError, match="'selector' is required"):
        validate_request(md_request_data)


def test_a_molecule_needs_a_source(input_dir: Path) -> None:
    data = {
        "protocol": "md",
        "name": "sourceless",
        "systems": [{"name": "s", "ligands": [{"name": "lig", "smiles": LIGAND_SMILES}]}],
    }

    with pytest.raises(InputValidationError, match="no source for 'lig'"):
        validate_request(data)


def test_the_protein_needs_a_source(input_dir: Path) -> None:
    data = {
        "protocol": "md",
        "name": "proteinless",
        "systems": [
            {
                "name": "s",
                "protein": {"chains": ["A"]},
                "ligands": [
                    {"name": "lig", "path": str(input_dir / "ligand.sdf"), "smiles": LIGAND_SMILES}
                ],
            }
        ],
    }

    with pytest.raises(InputValidationError, match="no source for the protein"):
        validate_request(data)


def test_duplicate_molecule_names_are_rejected(md_request_data: dict[str, Any]) -> None:
    md_request_data["systems"][0]["cofactors"] = [
        {"name": "lig", "selector": {"chain": "C"}, "smiles": LIGAND_SMILES}
    ]

    with pytest.raises(InputValidationError, match="duplicate molecule names"):
        validate_request(md_request_data)


def test_duplicate_system_names_are_rejected(md_request_data: dict[str, Any]) -> None:
    md_request_data["systems"].append(dict(md_request_data["systems"][0]))

    with pytest.raises(InputValidationError, match="duplicate system names"):
        validate_request(md_request_data)


def test_one_system_is_required(input_dir: Path) -> None:
    with pytest.raises(InputValidationError, match="systems"):
        validate_request({"protocol": "md", "name": "none", "systems": []})


def test_an_unsupported_structure_suffix_is_rejected(md_request_data: dict[str, Any]) -> None:
    md_request_data["systems"][0]["structure"] = "model.xyz"

    with pytest.raises(InputValidationError, match="must be a PDB or mmCIF"):
        validate_request(md_request_data)


def test_free_energy_fields_are_rejected(md_request_data: dict[str, Any]) -> None:
    """MD has no network, no charge policy and no alchemical ligand."""
    md_request_data["systems"][0]["extra_ligand_copies"] = "drop"

    with pytest.raises(InputValidationError, match="extra_ligand_copies"):
        validate_request(md_request_data)


def test_every_input_path_is_collected(md_request_data: dict[str, Any]) -> None:
    request = validated(md_request_data)

    assert [path.name for path in request.input_paths()] == ["model.cif"]


def test_relaxation_is_available_here_too(md_request_data: dict[str, Any]) -> None:
    md_request_data["relax"] = {"enabled": True, "length": "0.2 nanosecond"}

    assert validated(md_request_data).relax.length == "0.2 nanosecond"


def test_the_molecule_list_keeps_ligands_before_cofactors(
    md_request_data: dict[str, Any],
) -> None:
    md_request_data["systems"][0]["cofactors"] = [
        {"name": "nad", "selector": {"chain": "C"}, "smiles": LIGAND_SMILES}
    ]

    names = [molecule.name for molecule in validated(md_request_data).systems[0].molecules()]

    assert names == ["lig", "nad"]


def test_an_unknown_protocol_names_md(md_request_data: dict[str, Any]) -> None:
    md_request_data["protocol"] = "mdrun"

    with pytest.raises(InputValidationError, match=re.escape("'md'")):
        validate_request(md_request_data)
