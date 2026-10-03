"""Tests for preparing a whole complex, and for protein preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.complex import prepare_complex
from openfe_api.prep.protein import check_parameterizable, prepare_protein
from openfe_api.prep.runner import prepare_campaign
from openfe_api.prep.structure import Structure
from openfe_api.schema.abfe import ComplexSpec
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"

LIGAND_SMILES = "C[C@H](O)c1ccccc1"
COFACTOR_SMILES = "CC(=O)[O-]"


def _combined_spec(**overrides: object) -> ComplexSpec:
    """Build a complex spec reading everything from the mini complex file.

    Args:
        **overrides: Fields to override on the spec.

    Returns:
        The validated complex specification.
    """
    data: dict[str, object] = {
        "name": "mini",
        "structure": str(DATA / "mini_complex.cif"),
        "protein": {"chains": ["A"]},
        "ligand": {"selector": {"chain": "L"}, "smiles": LIGAND_SMILES},
        "extra_ligand_copies": "drop",
        "cofactors": [{"name": "acetate", "selector": {"chain": "C"}, "smiles": COFACTOR_SMILES}],
    }
    data.update(overrides)
    return ComplexSpec.model_validate(data)


def test_prepare_protein_adds_hydrogens(tmp_path: Path) -> None:
    structure = Structure.load(DATA / "mini_protein.pdb")

    protein = prepare_protein(structure, tmp_path / "protein.pdb", chains=["A"])

    assert protein.path.is_file()
    assert protein.chains == ["A"]
    assert protein.n_atoms > 92
    assert not (tmp_path / "protein.extracted.pdb").exists()


def test_prepare_protein_rejects_an_absent_chain(tmp_path: Path) -> None:
    structure = Structure.load(DATA / "mini_protein.pdb")

    with pytest.raises(InputValidationError, match="chain\\(s\\) Q not found"):
        prepare_protein(structure, tmp_path / "protein.pdb", chains=["Q"])


def test_prepare_protein_rejects_an_empty_selection(tmp_path: Path) -> None:
    structure = Structure.load(DATA / "mini_complex.cif")
    excluded = {residue.index for residue in structure.residues()}

    with pytest.raises(InputValidationError, match="no protein residues remain"):
        prepare_protein(structure, tmp_path / "protein.pdb", exclude_residues=excluded)


def test_prepared_protein_parameterizes(tmp_path: Path) -> None:
    structure = Structure.load(DATA / "mini_protein.pdb")
    protein = prepare_protein(structure, tmp_path / "protein.pdb")

    check_parameterizable(protein.path)


def test_unparameterizable_protein_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "protein.pdb"
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="cannot parameterize"):
        check_parameterizable(path)


def test_prepare_complex_writes_every_artifact(tmp_path: Path) -> None:
    report = prepare_complex(_combined_spec(), tmp_path / "mini")

    assert (tmp_path / "mini" / "protein.pdb").is_file()
    assert (tmp_path / "mini" / "ligand.sdf").is_file()
    assert (tmp_path / "mini" / "cofactor_acetate.sdf").is_file()
    assert (tmp_path / "mini" / "prep_report.json").is_file()
    assert report.ligand.net_charge == 0
    assert report.cofactors[0].net_charge == -1
    assert report.provenance["python"]


def test_prepare_complex_drops_extra_ligand_copies(tmp_path: Path) -> None:
    report = prepare_complex(_combined_spec(), tmp_path / "mini")

    assert report.ligand_copies == []
    assert any("dropped an additional copy" in warning for warning in report.warnings)


def test_prepare_complex_keeps_extra_ligand_copies(tmp_path: Path) -> None:
    report = prepare_complex(_combined_spec(extra_ligand_copies="keep"), tmp_path / "mini")

    assert [copy.name for copy in report.ligand_copies] == ["mini_copy_M"]
    assert (tmp_path / "mini" / "mini_copy_M.sdf").is_file()
    assert any("kept an additional copy" in warning for warning in report.warnings)


def test_prepare_complex_rejects_a_charged_ligand(tmp_path: Path) -> None:
    spec = _combined_spec(
        ligand={"selector": {"chain": "C"}, "smiles": COFACTOR_SMILES},
        cofactors=[],
    )

    with pytest.raises(InputValidationError, match="net charge of -1"):
        prepare_complex(spec, tmp_path / "mini")


def test_prepare_complex_neutralizes_when_asked(tmp_path: Path) -> None:
    spec = _combined_spec(
        ligand={"selector": {"chain": "C"}, "smiles": COFACTOR_SMILES},
        cofactors=[],
    )

    report = prepare_complex(spec, tmp_path / "mini", neutralize_ligands=True)

    assert report.ligand.net_charge == 0
    assert report.ligand.neutralized is True
    assert any("NEUTRALIZED" in warning for warning in report.warnings)


def test_prepare_complex_reads_split_files(tmp_path: Path) -> None:
    spec = ComplexSpec.model_validate(
        {
            "name": "split",
            "protein": {"path": str(DATA / "mini_protein.pdb")},
            "ligand": {"path": str(DATA / "ligand.sdf"), "smiles": LIGAND_SMILES},
            "cofactors": [
                {
                    "name": "acetate",
                    "path": str(DATA / "cofactor.sdf"),
                    "smiles": COFACTOR_SMILES,
                }
            ],
        }
    )

    report = prepare_complex(spec, tmp_path / "split")

    assert report.ligand.selector is None
    assert report.cofactors[0].net_charge == -1


def test_prep_report_is_valid_json(tmp_path: Path) -> None:
    prepare_complex(_combined_spec(), tmp_path / "mini")

    payload = json.loads((tmp_path / "mini" / "prep_report.json").read_text(encoding="utf-8"))

    assert payload["name"] == "mini"
    assert payload["ligand"]["smiles"]
    assert payload["protein"]["parameterized"] is True


def _campaign(tmp_path: Path, **overrides: object) -> Campaign:
    """Create a campaign holding one mini complex.

    Args:
        tmp_path: Pytest temporary directory.
        **overrides: Fields to override on the complex spec.

    Returns:
        The created campaign.
    """
    spec = _combined_spec(**overrides)
    request = validate_request(
        {
            "protocol": "abfe",
            "name": "mini_campaign",
            "complexes": [spec.model_dump()],
            "execution": {"repeats": 1},
        }
    )
    return Campaign.create(tmp_path / "campaign", request)


def test_prepare_campaign_marks_runs_prepared(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = prepare_campaign(campaign)

    assert set(outcome.reports) == {"mini"}
    assert outcome.failures == {}
    assert campaign.run("mini").state is RunState.PREPARED


def test_prepare_campaign_records_failures_without_raising(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, ligand={"selector": {"chain": "L"}, "smiles": "CCO"})

    outcome = prepare_campaign(campaign)

    record = campaign.run("mini")
    assert "mini" in outcome.failures
    assert record.state is RunState.FAILED
    assert record.message is not None and "heavy atoms" in record.message


def test_prepare_campaign_collects_warnings(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = prepare_campaign(campaign)

    assert any("dropped an additional copy" in warning for warning in outcome.warnings)


def test_prepare_campaign_can_skip_the_parameter_check(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert outcome.reports["mini"].protein.parameterized is False


def test_terminal_caps_are_kept(tmp_path: Path) -> None:
    """Regression: ACE and NME caps were dropped as unrecognized residues.

    Dropping them silently turns deliberately capped termini into charged ones. The
    fixture is cut from the public TYK2 benchmark structure used in the OpenFE tutorials.
    """
    structure = Structure.load(DATA / "capped_fragment.pdb")

    protein = prepare_protein(structure, tmp_path / "protein.pdb")

    residue_names = {residue.name for residue in Structure.load(protein.path).topology.residues()}
    assert {"ACE", "NME"} <= residue_names
    assert protein.dropped_residues == []
    assert not any("terminal cap" in warning for warning in protein.warnings)
    check_parameterizable(protein.path)
