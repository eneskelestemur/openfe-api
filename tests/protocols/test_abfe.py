"""Tests for planning ABFE transformations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from gufe import SmallMoleculeComponent, Transformation
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from openfe_api.campaign import Campaign, RunState
from openfe_api.exceptions import InputValidationError
from openfe_api.prep.complex import prepare_complex
from openfe_api.prep.report import ComplexPrepReport
from openfe_api.prep.runner import prepare_campaign
from openfe_api.protocols import charges
from openfe_api.protocols.abfe import build_settings, plan_abfe
from openfe_api.protocols.charges import ChargeCache, assign_charges
from openfe_api.protocols.restraints import check_restraint_search
from openfe_api.protocols.runner import plan_campaign
from openfe_api.schema.abfe import ComplexSpec
from openfe_api.schema.common import SettingsSpec
from openfe_api.schema.request import validate_request

DATA = Path(__file__).parents[1] / "data"

LIGAND_SMILES = "C[C@H](O)c1ccccc1"
COFACTOR_SMILES = "CC(=O)[O-]"

FAST_CHARGES = {"partial_charge_settings.partial_charge_method": "nagl"}


def _complex_data(**overrides: Any) -> dict[str, Any]:
    """Build request data for the mini complex.

    Args:
        **overrides: Fields to override on the complex spec.

    Returns:
        The complex specification data.
    """
    data: dict[str, Any] = {
        "name": "mini",
        "structure": str(DATA / "mini_complex.cif"),
        "protein": {"chains": ["A"]},
        "ligand": {"selector": {"chain": "L"}, "smiles": LIGAND_SMILES},
        "extra_ligand_copies": "drop",
        "cofactors": [{"name": "acetate", "selector": {"chain": "C"}, "smiles": COFACTOR_SMILES}],
    }
    data.update(overrides)
    return data


@pytest.fixture(scope="session")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> ComplexPrepReport:
    """Prepare the mini complex once for the whole session.

    Tests only read this report, or copy it before changing it.

    Args:
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        The preparation report.
    """
    spec = ComplexSpec.model_validate(_complex_data())
    return prepare_complex(spec, tmp_path_factory.mktemp("prepared_mini"), check_parameters=False)


@pytest.fixture(scope="session")
def fast_settings():
    """Build screening settings that use the fast NAGL charge method.

    Returns:
        The ABFE settings.
    """
    return build_settings(SettingsSpec(preset="screening", overrides=dict(FAST_CHARGES)))


@pytest.fixture(scope="session")
def planned_transformation(
    prepared: ComplexPrepReport,
    fast_settings: Any,
    tmp_path_factory: pytest.TempPathFactory,
) -> Transformation:
    """Plan the mini complex once and return the transformation it wrote.

    Args:
        prepared: The prepared mini complex.
        fast_settings: Screening settings using NAGL charges.
        tmp_path_factory: Pytest temporary directory factory.

    Returns:
        The transformation read back from disk.
    """
    planned = plan_abfe(prepared, fast_settings, tmp_path_factory.mktemp("plans"), repeats=1)
    return cast(Transformation, Transformation.from_json(planned.path))


def test_plan_writes_a_transformation(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    planned = plan_abfe(prepared, fast_settings, tmp_path / "plans", repeats=1)

    assert planned.path.is_file()
    assert planned.name == "abfe_mini"
    assert planned.ligand_name == "mini"


def test_planned_transformation_round_trips(planned_transformation: Transformation) -> None:
    transformation = planned_transformation

    assert transformation.name == "abfe_mini"
    assert transformation.mapping is None


def test_state_b_is_state_a_without_the_ligand(
    planned_transformation: Transformation,
) -> None:
    transformation = planned_transformation

    state_a = set(transformation.stateA.components)
    state_b = set(transformation.stateB.components)

    assert state_a - state_b == {"ligand"}
    assert state_b == {"protein", "solvent", "cofactor_acetate"}


def test_cofactors_appear_in_both_states(planned_transformation: Transformation) -> None:
    transformation = planned_transformation

    assert "cofactor_acetate" in transformation.stateA.components
    assert "cofactor_acetate" in transformation.stateB.components


def test_planned_ligand_carries_partial_charges(
    planned_transformation: Transformation,
) -> None:
    ligand = cast(SmallMoleculeComponent, planned_transformation.stateA.components["ligand"])
    assert ligand.to_openff().partial_charges is not None


def test_plan_reports_the_cost(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    planned = plan_abfe(prepared, fast_settings, tmp_path / "plans", repeats=1)

    assert planned.cost.complex_replicas == 30
    assert planned.cost.repeats == 1
    assert planned.cost.total_ns > 0


def test_plan_rejects_a_charged_ligand(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    """The protocol itself refuses a charged alchemical species."""
    charged = prepared.model_copy(deep=True)
    charged.ligand.path = prepared.cofactors[0].path
    charged.cofactors = []

    with pytest.raises(InputValidationError, match="rejected this system"):
        plan_abfe(charged, fast_settings, tmp_path / "plans", repeats=1)


def test_plan_reports_an_unreadable_molecule(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    broken = prepared.model_copy(deep=True)
    broken.ligand.path = tmp_path / "absent.sdf"

    with pytest.raises(InputValidationError, match="unreadable"):
        plan_abfe(broken, fast_settings, tmp_path / "plans", repeats=1)


def test_plan_reports_an_unreadable_protein(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    broken = prepared.model_copy(deep=True)
    broken.protein.path = tmp_path / "absent.pdb"

    with pytest.raises(InputValidationError, match="could not be loaded"):
        plan_abfe(broken, fast_settings, tmp_path / "plans", repeats=1)


def test_charges_are_cached_and_reused(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    cache = ChargeCache(tmp_path / "charges")
    plan_abfe(prepared, fast_settings, tmp_path / "plans", repeats=1, charge_cache=cache)

    assert (cache.directory / "mini.sdf").is_file()
    metadata = json.loads((cache.directory / "mini.json").read_text(encoding="utf-8"))
    assert metadata["method"] == "nagl"


def test_cache_is_ignored_when_the_method_changes(
    prepared: ComplexPrepReport, fast_settings: Any, tmp_path: Path
) -> None:
    cache = ChargeCache(tmp_path / "charges")
    plan_abfe(prepared, fast_settings, tmp_path / "plans", repeats=1, charge_cache=cache)

    metadata = json.loads((cache.directory / "mini.json").read_text(encoding="utf-8"))
    assert cache.load("mini", metadata["smiles"], "nagl") is not None
    assert cache.load("mini", metadata["smiles"], "am1bcc") is None
    assert cache.load("mini", "wrong-smiles", "nagl") is None


def test_cache_miss_on_unknown_molecule(tmp_path: Path) -> None:
    cache = ChargeCache(tmp_path / "charges")

    assert cache.load("absent", "CCO", "nagl") is None


def test_assign_charges_preserves_order(tmp_path: Path) -> None:
    spec = ComplexSpec.model_validate(_complex_data())
    report = prepare_complex(spec, tmp_path / "prepared", check_parameters=False)
    settings = build_settings(SettingsSpec(overrides=dict(FAST_CHARGES)))
    molecules = [
        SmallMoleculeComponent.from_sdf_file(str(report.ligand.path)),
        SmallMoleculeComponent.from_sdf_file(str(report.cofactors[0].path)),
    ]

    charged = assign_charges(molecules, settings.partial_charge_settings)

    assert [molecule.name for molecule in charged] == [m.name for m in molecules]


def test_cached_metadata_follows_the_molecule_not_the_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The backend may return results in the order it finished them.

    Pairing them positionally recorded one molecule's SMILES against another's charges, which
    two stereoisomers cannot survive: they differ only there, so the cache missed every time
    and silently recomputed.
    """
    first = _embedded_component("C[C@H](O)c1ccccc1", "r_form")
    second = _embedded_component("C[C@@H](O)c1ccccc1", "s_form")
    monkeypatch.setattr(
        charges,
        "bulk_assign_partial_charges",
        lambda molecules, **kwargs: list(reversed(molecules)),
    )
    settings = build_settings(SettingsSpec(overrides=dict(FAST_CHARGES))).partial_charge_settings
    cache = ChargeCache(tmp_path / "charges")

    charged = assign_charges([first, second], settings, cache=cache)

    assert [molecule.name for molecule in charged] == ["r_form", "s_form"]
    for molecule in (first, second):
        metadata = json.loads(
            (cache.directory / f"{molecule.name}.json").read_text(encoding="utf-8")
        )
        assert metadata["smiles"] == molecule.smiles
    assert cache.load("r_form", first.smiles, "nagl") is not None
    assert cache.load("s_form", second.smiles, "nagl") is not None


def _campaign(tmp_path: Path) -> Campaign:
    """Create and prepare a campaign holding the mini complex.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The prepared campaign.
    """
    request = validate_request(
        {
            "protocol": "abfe",
            "name": "mini_campaign",
            "complexes": [_complex_data()],
            "settings": {"preset": "screening", "overrides": dict(FAST_CHARGES)},
            "execution": {"repeats": 1},
        }
    )
    campaign = Campaign.create(tmp_path / "campaign", request)
    prepare_campaign(campaign, check_parameters=False)
    return campaign


def test_plan_campaign_marks_runs_planned(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = plan_campaign(campaign)

    assert set(outcome.planned) == {"mini"}
    assert campaign.run("mini").state is RunState.PLANNED
    assert (campaign.directory / "plans" / "abfe_mini.json").is_file()


def test_plan_campaign_totals_the_cost(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    outcome = plan_campaign(campaign)

    assert outcome.total_ns == pytest.approx(outcome.planned["mini"].cost.total_ns)


def test_plan_campaign_skips_unprepared_runs(tmp_path: Path) -> None:
    request = validate_request(
        {
            "protocol": "abfe",
            "name": "mini_campaign",
            "complexes": [_complex_data()],
            "settings": {"preset": "screening", "overrides": dict(FAST_CHARGES)},
        }
    )
    campaign = Campaign.create(tmp_path / "campaign", request)

    outcome = plan_campaign(campaign)

    assert "not prepared yet" in outcome.failures["mini"]
    assert campaign.run("mini").state is RunState.PENDING


@pytest.mark.slow
def test_plan_with_default_am1bcc_charges(tmp_path: Path) -> None:
    """The default charge method is AM1BCC, which is accurate but slow."""
    spec = ComplexSpec.model_validate(_complex_data(cofactors=[]))
    report = prepare_complex(spec, tmp_path / "prepared", check_parameters=False)
    settings = build_settings(SettingsSpec(preset="screening"))

    planned = plan_abfe(report, settings, tmp_path / "plans", repeats=1)

    assert planned.path.is_file()


def test_planned_transformation_runs_exactly_one_repeat(
    planned_transformation: Transformation,
) -> None:
    """Regression: OpenFE's default of 3 made every quickrun process run 3 repeats."""
    protocol = planned_transformation.protocol

    assert protocol.settings.protocol_repeats == 1  # type: ignore[attr-defined]


def _embedded_component(smiles: str, name: str) -> SmallMoleculeComponent:
    """Build a small molecule component with 3D coordinates and hydrogens.

    Args:
        smiles: SMILES to embed.
        name: Component name.

    Returns:
        The component.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    return SmallMoleculeComponent(molecule, name=name)


@pytest.mark.parametrize(
    "smiles",
    ["c1ccccc1", "c1ccc2ccccc2c1", "C[C@H](O)c1ccccc1"],
    ids=["benzene", "naphthalene", "phenylethanol"],
)
def test_restraint_search_accepts_up_to_two_fused_rings(smiles: str) -> None:
    check_restraint_search(_embedded_component(smiles, "ligand"), "ABFE")


@pytest.mark.parametrize(
    "smiles",
    ["c1ccc2cc3ccccc3cc2c1", "c1ccc2c(c1)ccc1ccccc12"],
    ids=["anthracene", "phenanthrene"],
)
def test_restraint_search_rejects_three_fused_rings(smiles: str) -> None:
    """Tripwire for an OpenFE bug: when this starts failing, OpenFE has been fixed.

    OpenFE 1.12's ``get_aromatic_rings`` crashes on three or more fused aromatic rings,
    which would otherwise only surface once the complex leg starts on a GPU.
    """
    with pytest.raises(InputValidationError, match="Boresch restraint search"):
        check_restraint_search(_embedded_component(smiles, "ligand"), "ABFE")
