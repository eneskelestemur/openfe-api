"""Tests for the gates applied to a series: RBFE's common core and SepTop's screen."""

from __future__ import annotations

from rdkit import Chem

from openfe_api.prep.gate import clash_verdict, filter_series, screen_series
from openfe_api.schema.rbfe import SimilaritySpec
from openfe_api.schema.septop import ScreeningSpec

REFERENCE = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
CLOSE = "CCc1ccc(cc1)C(=O)Nc1ccccc1"
ALSO_CLOSE = "Clc1ccc(cc1)C(=O)Nc1ccccc1"
PROPYL = "CCCc1ccc(cc1)C(=O)Nc1ccccc1"
DISTANT = "CCO"
OTHER_CHEMOTYPE = "c1ccc2[nH]ccc2c1"
HOP = "O=C(Nc1ccncc1)C1CCOCC1"
ANION = "Cc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)[O-]"
# Hydrogen sulfide shares no element with the reference, so their MCS is empty.
NO_SHARED_ATOMS = "S"


def mol(smiles: str) -> Chem.Mol:
    """Build a molecule from SMILES.

    Args:
        smiles: The molecule's SMILES.

    Returns:
        The molecule.
    """
    return Chem.MolFromSmiles(smiles)


def test_close_analogues_are_kept() -> None:
    outcome = filter_series(
        mol(REFERENCE), "ref", {"a": mol(CLOSE), "b": mol(ALSO_CLOSE)}, SimilaritySpec()
    )

    assert [member.name for member in outcome.kept] == ["a", "b"]
    assert outcome.dropped == {}
    assert outcome.warnings == []


def test_a_distant_ligand_is_dropped_with_the_numbers() -> None:
    outcome = filter_series(mol(REFERENCE), "ref", {"far": mol(DISTANT)}, SimilaritySpec())

    assert outcome.kept == []
    assert "below the 60% required" in outcome.dropped["far"]


def test_lowering_the_threshold_keeps_it() -> None:
    outcome = filter_series(
        mol(REFERENCE), "ref", {"far": mol(DISTANT)}, SimilaritySpec(min_core_fraction=0.1)
    )

    assert [member.name for member in outcome.kept] == ["far"]


def test_a_ring_cutting_core_is_dropped_by_default() -> None:
    outcome = filter_series(
        mol(REFERENCE), "ref", {"other": mol(OTHER_CHEMOTYPE)}, SimilaritySpec()
    )

    assert "cuts through a ring" in outcome.dropped["other"]


def test_perturbed_atom_cap_is_enforced() -> None:
    spec = SimilaritySpec(max_perturbed_atoms=1)

    outcome = filter_series(
        mol(REFERENCE), "ref", {"ethyl": mol(CLOSE), "propyl": mol(PROPYL)}, spec
    )

    assert [member.name for member in outcome.kept] == ["ethyl"]
    assert "heavy atoms differ from the reference" in outcome.dropped["propyl"]


def test_an_explicit_core_is_applied_to_every_ligand() -> None:
    spec = SimilaritySpec(core_smarts="O=C(Nc1ccccc1)c1ccccc1")

    outcome = filter_series(mol(REFERENCE), "ref", {"a": mol(CLOSE), "far": mol(DISTANT)}, spec)

    assert [member.name for member in outcome.kept] == ["a"]
    assert "does not match" in outcome.dropped["far"]
    kept = outcome.kept[0].core
    assert kept is not None
    assert kept.smarts == spec.core_smarts


def test_a_ligand_with_nothing_in_common_is_dropped_not_raised() -> None:
    """One unusable ligand must not stop the rest of the series being prepared."""
    outcome = filter_series(
        mol(REFERENCE), "ref", {"salt": mol("[Na+].[Cl-]"), "a": mol(CLOSE)}, SimilaritySpec()
    )

    assert [member.name for member in outcome.kept] == ["a"]
    assert "salt" in outcome.dropped


def test_a_ring_addition_does_not_warn() -> None:
    """A ring substituent changes the scaffold but is a routine edge, so this must stay quiet."""
    outcome = filter_series(
        mol("COc1cc2ncnc(Nc3ccccc3)c2cc1OC"),
        "ref",
        {"grown": mol("COc1cc2ncnc(Nc3ccccc3)c2cc1OC1CCNC1")},
        SimilaritySpec(),
    )

    assert [member.name for member in outcome.kept] == ["grown"]
    assert outcome.warnings == []


def test_stereoisomers_do_not_warn() -> None:
    outcome = filter_series(
        mol("COc1cc2ncnc(Nc3ccccc3)c2cc1O[C@H]1CCNC1"),
        "ref",
        {"enantiomer": mol("COc1cc2ncnc(Nc3ccccc3)c2cc1O[C@@H]1CCNC1")},
        SimilaritySpec(),
    )

    assert outcome.warnings == []


def test_a_genuinely_different_chemotype_warns() -> None:
    outcome = filter_series(
        mol(REFERENCE),
        "ref",
        {"a": mol(CLOSE), "other": mol(OTHER_CHEMOTYPE)},
        SimilaritySpec(allow_ring_break=True, min_core_fraction=0.1),
    )

    assert any("more than one chemotype" in warning for warning in outcome.warnings)


def test_an_empty_series_warns() -> None:
    outcome = filter_series(mol(REFERENCE), "ref", {"far": mol(DISTANT)}, SimilaritySpec())

    assert any("no edge to run" in warning for warning in outcome.warnings)


def test_scaffold_is_recorded_per_member() -> None:
    outcome = filter_series(mol(REFERENCE), "ref", {"a": mol(CLOSE)}, SimilaritySpec())

    assert outcome.kept[0].scaffold == "O=C(Nc1ccccc1)c1ccccc1"


def test_a_hop_passes_the_septop_screen() -> None:
    """Sharing no core is the case SepTop exists for, so it must not be a rejection."""
    outcome = screen_series(mol(REFERENCE), "ref", {"hop": mol(HOP)}, ScreeningSpec())

    assert [member.name for member in outcome.kept] == ["hop"]
    assert outcome.dropped == {}


def test_a_hop_is_reported_as_one() -> None:
    outcome = screen_series(mol(REFERENCE), "ref", {"hop": mol(HOP)}, ScreeningSpec())

    assert any("share no ring system" in warning for warning in outcome.warnings)


def test_a_charge_change_is_always_dropped() -> None:
    """The protocol refuses a net charge change, so no policy can keep the ligand."""
    spec = ScreeningSpec(policy="warn")

    outcome = screen_series(mol(REFERENCE), "ref", {"anion": mol(ANION)}, spec)

    assert outcome.kept == []
    assert "net charge of -1" in outcome.dropped["anion"]
    assert "does not support a net charge change" in outcome.dropped["anion"]


def test_a_size_mismatch_only_warns_by_default() -> None:
    outcome = screen_series(mol(REFERENCE), "ref", {"tiny": mol(NO_SHARED_ATOMS)}, ScreeningSpec())

    assert [member.name for member in outcome.kept] == ["tiny"]
    assert any("max_size_ratio" in warning for warning in outcome.warnings)


def test_a_size_mismatch_drops_under_the_drop_policy() -> None:
    spec = ScreeningSpec(policy="drop")

    outcome = screen_series(mol(REFERENCE), "ref", {"tiny": mol(NO_SHARED_ATOMS)}, spec)

    assert outcome.kept == []
    assert "max_size_ratio" in outcome.dropped["tiny"]
    assert "'screening.policy' is 'drop'" in outcome.dropped["tiny"]


def test_a_ligand_sharing_nothing_has_no_core() -> None:
    """A core is optional here: the placement step falls back to shape alignment."""
    spec = ScreeningSpec(max_size_ratio=20.0)

    outcome = screen_series(mol(REFERENCE), "ref", {"tiny": mol(NO_SHARED_ATOMS)}, spec)

    assert outcome.kept[0].core is None


def test_the_core_gate_is_available_as_a_knob() -> None:
    spec = ScreeningSpec(min_core_fraction=0.6, policy="drop")

    outcome = screen_series(mol(REFERENCE), "ref", {"hop": mol(HOP)}, spec)

    assert "min_core_fraction" in outcome.dropped["hop"]


def test_a_close_analogue_passes_the_core_gate_when_it_is_set() -> None:
    spec = ScreeningSpec(min_core_fraction=0.6, policy="drop")

    outcome = screen_series(mol(REFERENCE), "ref", {"close": mol(CLOSE)}, spec)

    assert [member.name for member in outcome.kept] == ["close"]


def test_an_empty_screen_says_there_is_no_edge() -> None:
    spec = ScreeningSpec(policy="drop")

    outcome = screen_series(mol(REFERENCE), "ref", {"anion": mol(ANION)}, spec)

    assert any("no edge" in warning for warning in outcome.warnings)


def test_a_clean_pose_passes_the_clash_threshold() -> None:
    assert clash_verdict(ScreeningSpec(), clashes=0) is None


def test_too_many_clashes_fail_the_threshold() -> None:
    verdict = clash_verdict(ScreeningSpec(max_clashes=2), clashes=7)

    assert verdict is not None
    assert "7 heavy atom(s)" in verdict
    assert "max_clashes" in verdict


def test_a_ligand_sharing_no_core_is_pointed_at_septop_by_the_rbfe_gate() -> None:
    outcome = filter_series(mol(REFERENCE), "ref", {"tiny": mol(NO_SHARED_ATOMS)}, SimilaritySpec())

    assert "SepTop" in outcome.dropped["tiny"]
