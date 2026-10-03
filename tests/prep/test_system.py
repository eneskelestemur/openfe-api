"""Tests for preparing a plain MD system: complex, apo protein and ligand in water."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.report import SystemPrepReport
from openfe_api.prep.runner import prepare_campaign
from openfe_api.prep.system import prepare_system
from openfe_api.schema.md import MdRequest, MdSystemSpec
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"
LIGAND_SMILES = "C[C@H](O)c1ccccc1"
CHARGED_SMILES = "CC(=O)[O-]"


def system_data(**overrides: Any) -> dict[str, Any]:
    """Build request data for one system on the trimmed complex fixture.

    Args:
        **overrides: Fields to replace on the system.

    Returns:
        The system dictionary.
    """
    data: dict[str, Any] = {
        "name": "complex_one",
        "structure": str(DATA / "mini_complex.cif"),
        "protein": {"chains": ["A"]},
        "ligands": [{"name": "lig", "selector": {"chain": "L"}, "smiles": LIGAND_SMILES}],
    }
    data.update(overrides)
    return data


def prepare(data: dict[str, Any], tmp_path: Path) -> SystemPrepReport:
    """Validate one system and prepare it.

    Args:
        data: The system dictionary.
        tmp_path: Pytest temporary directory.

    Returns:
        The preparation report.
    """
    spec = MdSystemSpec.model_validate(data)
    return prepare_system(spec, tmp_path / "prepared", check_parameters=False)


def test_a_complex_prepares(tmp_path: Path) -> None:
    report = prepare(system_data(), tmp_path)

    assert report.protein is not None
    assert [molecule.name for molecule in report.molecules] == ["lig"]
    assert (tmp_path / "prepared" / "lig.sdf").is_file()
    assert (tmp_path / "prepared" / "protein.pdb").is_file()


def test_an_apo_protein_prepares(tmp_path: Path) -> None:
    report = prepare(system_data(ligands=[]), tmp_path)

    assert report.protein is not None
    assert report.molecules == []


def test_a_ligand_in_water_prepares_without_a_protein(tmp_path: Path) -> None:
    data = system_data(protein=None, structure=str(DATA / "mini_complex.cif"))

    report = prepare(data, tmp_path)

    assert report.protein is None
    assert [molecule.name for molecule in report.molecules] == ["lig"]


def test_several_molecules_prepare_together(tmp_path: Path) -> None:
    data = system_data(
        cofactors=[{"name": "nad", "selector": {"chain": "L"}, "smiles": LIGAND_SMILES}]
    )

    report = prepare(data, tmp_path)

    assert [molecule.name for molecule in report.molecules] == ["lig", "nad"]
    assert [molecule.role for molecule in report.molecules] == ["ligand", "cofactor"]


def test_a_charged_molecule_is_accepted(tmp_path: Path) -> None:
    """Nothing here is alchemical, so the ABFE neutrality rule does not apply."""
    data = system_data(
        protein=None,
        structure=None,
        ligands=[{"name": "acetate", "path": str(DATA / "cofactor.sdf"), "smiles": CHARGED_SMILES}],
    )

    report = prepare(data, tmp_path)

    assert report.molecules[0].name == "acetate"


def test_a_mismatched_smiles_still_fails(tmp_path: Path) -> None:
    data = system_data(ligands=[{"name": "lig", "selector": {"chain": "L"}, "smiles": "CCO"}])

    with pytest.raises(InputValidationError):
        prepare(data, tmp_path)


def test_the_report_is_written(tmp_path: Path) -> None:
    prepare(system_data(), tmp_path)

    written = SystemPrepReport.model_validate_json(
        (tmp_path / "prepared" / "prep_report.json").read_text(encoding="utf-8")
    )
    assert written.name == "complex_one"


def test_a_campaign_prepares_every_system(tmp_path: Path) -> None:
    request = validate_request(
        {
            "protocol": "md",
            "name": "md_campaign",
            "systems": [system_data(), system_data(name="complex_two")],
            "execution": {"repeats": 1},
        }
    )
    assert isinstance(request, MdRequest)
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert sorted(outcome.systems) == ["complex_one", "complex_two"]
    assert campaign.run("complex_one").state is RunState.PREPARED


def test_only_selects_systems(tmp_path: Path) -> None:
    request = validate_request(
        {
            "protocol": "md",
            "name": "md_campaign",
            "systems": [system_data(), system_data(name="complex_two")],
        }
    )
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = prepare_campaign(campaign, only=["complex_two"], check_parameters=False)

    assert list(outcome.systems) == ["complex_two"]
