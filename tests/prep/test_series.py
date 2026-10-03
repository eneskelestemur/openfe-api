"""Tests for preparing a series into the reference ligand's frame, for both protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

from openfe_api.campaign import Campaign
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.runner import prepare_campaign
from openfe_api.prep.series import prepare_series
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.request import validate_request
from openfe_api.schema.septop import SepTopRequest

DATA = Path(__file__).parents[1] / "data"
REFERENCE_SMILES = "C[C@H](O)c1ccccc1"
METHYL = "C[C@H](O)c1ccc(C)cc1"
FLUORO = "C[C@H](O)c1ccc(F)cc1"


@pytest.fixture
def series_data() -> dict[str, Any]:
    """Return an RBFE request built on the trimmed complex fixture.

    Returns:
        Request data with a posed reference and two SMILES-only analogues.
    """
    return {
        "protocol": "rbfe",
        "name": "mini_series",
        "protein": {"chains": ["A"]},
        "ligands": [
            {
                "name": "reference",
                "structure": str(DATA / "mini_complex.cif"),
                "selector": {"chain": "L"},
                "smiles": REFERENCE_SMILES,
            },
            {"name": "methyl", "smiles": METHYL},
            {"name": "fluoro", "smiles": FLUORO},
        ],
    }


def _solvated_series_data() -> dict[str, Any]:
    """Return a series whose reference comes from a structure holding a solvent box."""
    return {
        "protocol": "rbfe",
        "name": "solvated_series",
        "protein": {"chains": ["A"], "keep_waters": False},
        "ligands": [
            {
                "name": "reference",
                "structure": str(DATA / "mini_complex_solvated.pdb"),
                "selector": {"chain": "L"},
                "smiles": REFERENCE_SMILES,
            },
            {"name": "methyl", "smiles": METHYL},
        ],
    }


def prepare(data: dict[str, Any], output_dir: Path) -> Any:
    """Validate a request and prepare its series.

    Args:
        data: Request data.
        output_dir: Directory to prepare into.

    Returns:
        The preparation report.
    """
    request = validate_request(data)
    assert isinstance(request, RbfeRequest)
    return prepare_series(request, output_dir, check_parameters=False)


def test_every_ligand_is_placed(series_data: dict[str, Any], tmp_path: Path) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    assert [placement.name for placement in report.ligands] == [
        "reference",
        "methyl",
        "fluoro",
    ]
    assert report.dropped == {}
    assert report.reference == "reference"


def test_the_reference_pose_is_kept_as_given(series_data: dict[str, Any], tmp_path: Path) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    reference = report.ligands[0]
    assert reference.pose == "keep"
    assert reference.core_rmsd == 0.0


def test_generated_poses_sit_on_the_reference_core(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    for placement in report.ligands[1:]:
        assert placement.pose == "mcs"
        assert placement.core_rmsd is not None
        assert placement.core_rmsd < 0.1


def test_prepared_files_are_written(series_data: dict[str, Any], tmp_path: Path) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    assert (tmp_path / "prepared" / "protein.pdb").is_file()
    assert (tmp_path / "prepared" / "prep_report.json").is_file()
    for placement in report.ligands:
        assert placement.path.is_file()
        molecule = Chem.SDMolSupplier(str(placement.path), removeHs=False)[0]
        assert molecule is not None
        assert molecule.GetNumConformers() == 1


def test_the_report_round_trips_as_json(series_data: dict[str, Any], tmp_path: Path) -> None:
    prepare(series_data, tmp_path / "prepared")

    written = json.loads((tmp_path / "prepared" / "prep_report.json").read_text())

    assert written["reference"] == "reference"
    assert len(written["ligands"]) == 3
    assert written["provenance"]["openfe"]


def test_core_metrics_are_recorded(series_data: dict[str, Any], tmp_path: Path) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    methyl = report.ligands[1]
    assert methyl.core_size > 0
    assert 0.0 < methyl.core_fraction <= 1.0
    assert methyl.core_smarts
    assert methyl.scaffold == "c1ccccc1"


def test_a_distant_ligand_is_dropped_with_a_reason(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    series_data["ligands"].append({"name": "unrelated", "smiles": "NC(=O)c1ccncc1C(F)(F)F"})

    report = prepare(series_data, tmp_path / "prepared")

    assert "unrelated" in report.dropped
    assert {placement.name for placement in report.ligands} == {
        "reference",
        "methyl",
        "fluoro",
    }


def test_an_undefined_stereocenter_excludes_only_that_ligand(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    """One unusable ligand must not stop the rest of the series preparing."""
    series_data["ligands"].append({"name": "racemic", "smiles": "CC(O)c1ccc(Br)cc1"})

    report = prepare(series_data, tmp_path / "prepared")

    assert "racemic" in report.dropped
    assert any("undefined stereocenter" in warning for warning in report.warnings)
    assert len(report.ligands) == 3


def test_a_series_with_nothing_to_run_is_rejected(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    series_data["ligands"] = [
        series_data["ligands"][0],
        {"name": "unrelated", "smiles": "NC(=O)c1ccncc1C(F)(F)F"},
    ]

    with pytest.raises(InputValidationError, match="no edge to run"):
        prepare(series_data, tmp_path / "prepared")


def test_clashes_against_the_protein_are_reported(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    report = prepare(series_data, tmp_path / "prepared")

    for placement in report.ligands:
        assert placement.worst_contact is not None


def test_a_cofactor_is_prepared_from_the_reference_frame(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    series_data["cofactors"] = [
        {"name": "acetate", "selector": {"chain": "C"}, "smiles": "CC(=O)[O-]"}
    ]

    report = prepare(series_data, tmp_path / "prepared")

    assert [cofactor.name for cofactor in report.cofactors] == ["acetate"]
    assert report.cofactors[0].path.is_file()
    assert report.cofactors[0].net_charge == -1


def test_an_explicit_core_is_honoured(series_data: dict[str, Any], tmp_path: Path) -> None:
    series_data["similarity"] = {"core_smarts": "C(O)c1ccccc1"}

    report = prepare(series_data, tmp_path / "prepared")

    assert all(placement.core_smarts == "C(O)c1ccccc1" for placement in report.ligands[1:])


def test_preparation_is_reproducible(series_data: dict[str, Any], tmp_path: Path) -> None:
    """Preparing twice must give the same coordinates, so a replan does not move ligands."""
    first = prepare(series_data, tmp_path / "one")
    second = prepare(series_data, tmp_path / "two")

    for one, two in zip(first.ligands, second.ligands, strict=True):
        left = Chem.SDMolSupplier(str(one.path), removeHs=False)[0]
        right = Chem.SDMolSupplier(str(two.path), removeHs=False)[0]
        assert left is not None and right is not None
        assert left.GetConformer().GetPositions() == pytest.approx(
            right.GetConformer().GetPositions()
        )


def test_prepare_campaign_dispatches_to_the_series(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    request = validate_request(series_data)
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert outcome.series is not None
    assert outcome.reports == {}
    assert len(outcome.series.ligands) == 3
    assert (campaign.directory / "prepared" / "prep_report.json").is_file()


def test_prepare_campaign_surfaces_series_warnings(
    series_data: dict[str, Any], tmp_path: Path
) -> None:
    request = validate_request(series_data)
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert outcome.warnings == (outcome.series.warnings if outcome.series else [])


def test_clashes_are_scored_against_the_prepared_protein(tmp_path: Path) -> None:
    """A solvated input would otherwise score every pose against waters it then drops."""
    data = _solvated_series_data()

    report = prepare(data, tmp_path / "prepared")

    dry = prepare(
        {
            **data,
            "ligands": [
                {**data["ligands"][0], "structure": str(DATA / "mini_complex.cif")},
                data["ligands"][1],
            ],
        },
        tmp_path / "dry",
    )

    solvated_clashes = {entry.name: entry.clashes for entry in report.ligands}
    dry_clashes = {entry.name: entry.clashes for entry in dry.ligands}
    assert solvated_clashes == dry_clashes


def test_waters_are_not_counted_as_pocket_contacts(tmp_path: Path) -> None:
    """Waters sit 2.5 A from the ligand center, so they would dominate the contact distance."""
    data = _solvated_series_data()

    report = prepare(data, tmp_path / "prepared")

    for placement in report.ligands:
        assert placement.worst_contact is not None
        assert placement.worst_contact > 1.0


SEPTOP_HOP = "O=C(N)C1CCOCC1"
SEPTOP_ANION = "C[C@H](O)c1ccc(cc1)C(=O)[O-]"


@pytest.fixture
def septop_data() -> dict[str, Any]:
    """Return a SepTop request holding one close analogue and one scaffold hop.

    Returns:
        Request data with a posed reference, a fluoro analogue and a hop sharing no ring
        system with the reference.
    """
    return {
        "protocol": "septop",
        "name": "mini_septop",
        "protein": {"chains": ["A"]},
        "ligands": [
            {
                "name": "reference",
                "structure": str(DATA / "mini_complex.cif"),
                "selector": {"chain": "L"},
                "smiles": REFERENCE_SMILES,
            },
            {"name": "fluoro", "smiles": FLUORO},
            {"name": "hop", "smiles": SEPTOP_HOP},
        ],
    }


def prepare_septop(data: dict[str, Any], output_dir: Path) -> Any:
    """Validate a SepTop request and prepare its series.

    Args:
        data: Request data.
        output_dir: Directory to prepare into.

    Returns:
        The preparation report.
    """
    request = validate_request(data)
    assert isinstance(request, SepTopRequest)
    return prepare_series(request, output_dir, check_parameters=False)


def placement(report: Any, name: str) -> Any:
    """Return one ligand's placement record.

    Args:
        report: A series preparation report.
        name: The ligand's name.

    Returns:
        The placement.
    """
    return next(entry for entry in report.ligands if entry.name == name)


def test_a_septop_series_places_an_analogue_and_a_hop(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert sorted(entry.name for entry in report.ligands) == ["fluoro", "hop", "reference"]
    assert report.dropped == {}


def test_auto_places_a_close_analogue_on_the_common_core(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    report = prepare_septop(septop_data, tmp_path / "prepared")

    entry = placement(report, "fluoro")
    assert entry.pose == "mcs"
    assert entry.core_rmsd is not None
    assert entry.shape_score is None


def test_auto_places_a_hop_by_shape(septop_data: dict[str, Any], tmp_path: Path) -> None:
    """A hop shares too little core to place on, so the fallback must be shape alignment."""
    report = prepare_septop(septop_data, tmp_path / "prepared")

    entry = placement(report, "hop")
    assert entry.pose == "shape"
    assert entry.shape_score is not None
    assert entry.shape_rmsd is not None
    assert entry.core_rmsd is None


def test_auto_keeps_a_supplied_pose(septop_data: dict[str, Any], tmp_path: Path) -> None:
    prepare_septop(septop_data, tmp_path / "prepared")
    ligand_path = tmp_path / "prepared" / "ligands" / "fluoro.sdf"
    septop_data["ligands"][1] = {"name": "fluoro", "smiles": FLUORO, "path": str(ligand_path)}

    second = prepare_septop(septop_data, tmp_path / "again")

    assert placement(second, "fluoro").pose == "keep"


def test_a_ligand_may_force_shape_placement(septop_data: dict[str, Any], tmp_path: Path) -> None:
    septop_data["ligands"][1]["pose"] = "shape"

    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert placement(report, "fluoro").pose == "shape"


def test_the_campaign_may_force_mcs_placement(septop_data: dict[str, Any], tmp_path: Path) -> None:
    """Forcing MCS on a hop must fail by name rather than placing it on a token core."""
    septop_data["alignment"] = {"pose": "mcs"}

    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert placement(report, "fluoro").pose == "mcs"
    assert "below the 3 needed to fix an orientation" in report.dropped["hop"]


def test_a_charge_change_is_dropped_from_a_septop_series(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    septop_data["ligands"][2] = {"name": "anion", "smiles": SEPTOP_ANION}

    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert "net charge" in report.dropped["anion"]


def test_the_resolved_path_is_recorded_for_every_ligand(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    """``auto`` must never reach the report: the user has to see what actually happened."""
    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert {entry.pose for entry in report.ligands} == {"keep", "mcs", "shape"}


def test_a_septop_campaign_prepares_through_the_runner(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    request = validate_request(septop_data)
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = prepare_campaign(campaign, check_parameters=False)

    assert outcome.series is not None
    assert len(outcome.series.ligands) == 3


def test_the_drop_policy_excludes_a_clashing_pose(
    septop_data: dict[str, Any], tmp_path: Path
) -> None:
    """Clash count is only known after placement, so the policy is applied there."""
    septop_data["screening"] = {"policy": "drop", "max_clashes": 0}

    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert "max_clashes" in report.dropped["fluoro"]
    assert "'screening.policy' is 'drop'" in report.dropped["fluoro"]
    assert [entry.name for entry in report.ligands] == ["reference", "hop"]


def test_the_warn_policy_keeps_a_clashing_pose(septop_data: dict[str, Any], tmp_path: Path) -> None:
    septop_data["screening"] = {"policy": "warn", "max_clashes": 0}

    report = prepare_septop(septop_data, tmp_path / "prepared")

    assert report.dropped == {}
    assert any("clash cutoff" in warning for warning in report.warnings)
