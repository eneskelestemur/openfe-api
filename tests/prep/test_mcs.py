"""Tests for finding the substructure a series shares with its reference."""

from __future__ import annotations

import pytest
from rdkit import Chem

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.mcs import (
    find_core,
    match_core,
    murcko_scaffold,
    shares_chemotype,
)

REFERENCE = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
ANALOGUE = "CCc1ccc(cc1)C(=O)Nc1ccccc1"
CORE_SMARTS = "O=C(Nc1ccccc1)c1ccccc1"


def build(smiles: str, hydrogens: bool = False) -> Chem.Mol:
    """Build a molecule from SMILES, optionally with explicit hydrogens.

    Args:
        smiles: The molecule's SMILES.
        hydrogens: Whether to add explicit hydrogens.

    Returns:
        The molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.AddHs(molecule) if hydrogens else molecule


def test_core_of_a_close_analogue_covers_the_smaller_molecule() -> None:
    core = find_core(build(REFERENCE), build(ANALOGUE))

    assert core.size == 16
    assert core.coverage == pytest.approx(1.0)
    assert core.perturbed_atoms == 1
    assert core.breaks_ring is False


def test_core_is_found_across_hydrogen_states() -> None:
    """The reference comes from a prepared file with hydrogens, a probe from SMILES without."""
    with_hydrogens = find_core(build(REFERENCE, hydrogens=True), build(ANALOGUE))
    without = find_core(build(REFERENCE), build(ANALOGUE))

    assert with_hydrogens.size == without.size
    assert with_hydrogens.coverage == pytest.approx(without.coverage)


def test_a_distant_molecule_has_low_coverage() -> None:
    core = find_core(build(REFERENCE), build("CCO"))

    assert core.coverage < 0.6


def test_no_common_substructure_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="no common substructure"):
        find_core(build("[Na+].[Cl-]"), build(REFERENCE))


def test_explicit_core_is_used_instead_of_searching() -> None:
    core = find_core(build(REFERENCE), build(ANALOGUE), core_smarts=CORE_SMARTS)

    assert core.smarts == CORE_SMARTS
    assert core.size == 15


def test_explicit_core_that_does_not_match_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="does not match this ligand"):
        find_core(build(REFERENCE), build("CCO"), core_smarts=CORE_SMARTS)


def test_explicit_core_that_misses_the_reference_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="does not match the reference ligand"):
        find_core(build("CCO"), build(ANALOGUE), core_smarts=CORE_SMARTS)


def test_unparseable_core_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="not a valid SMARTS"):
        find_core(build(REFERENCE), build(ANALOGUE), core_smarts="c1cc(")


def test_a_partial_ring_core_is_reported_as_breaking_a_ring() -> None:
    core = find_core(build(REFERENCE), build(ANALOGUE), core_smarts="cccC(=O)N")
    intact = find_core(build(REFERENCE), build(ANALOGUE), core_smarts=CORE_SMARTS)

    assert core.breaks_ring is True
    assert intact.breaks_ring is False


def test_ring_size_change_is_not_a_ring_break() -> None:
    """A ring-size change is a routine edge, so the default gate must not reject it."""
    core = find_core(build("C1CCCCC1C(=O)Nc1ccccc1"), build("C1CCCC1C(=O)Nc1ccccc1"))

    assert core.size > 0
    assert core.breaks_ring is False


def test_a_symmetric_core_reports_every_mapping() -> None:
    """The caller must choose the orientation; taking the first match can place it flipped."""
    probe = build(ANALOGUE)
    core = find_core(build(REFERENCE), probe)

    assert len(match_core(probe, core)) > 1


def test_murcko_scaffold_groups_by_chemotype() -> None:
    reference = murcko_scaffold(build(REFERENCE))

    assert murcko_scaffold(build(ANALOGUE)) == reference
    assert murcko_scaffold(build("CCOc1ccc(cc1)C(=O)Nc1ccccc1")) == reference
    assert murcko_scaffold(build("c1ccc2[nH]ccc2c1")) != reference


def test_murcko_scaffold_of_an_acyclic_molecule_is_empty() -> None:
    assert murcko_scaffold(build("CCCCO")) == ""


def test_scaffold_ignores_stereochemistry() -> None:
    """Two stereoisomers are one chemotype; counting them as two cries wolf."""
    left = murcko_scaffold(build("COc1cc2ncnc(N)c2cc1O[C@H]1CCNC1"))
    right = murcko_scaffold(build("COc1cc2ncnc(N)c2cc1O[C@@H]1CCNC1"))

    assert left == right
    assert "@" not in left


def test_a_ring_addition_is_the_same_chemotype() -> None:
    """Adding a ring substituent changes the scaffold but is a routine edge."""
    plain = murcko_scaffold(build("COc1cc2ncnc(Nc3ccccc3)c2cc1OC"))
    grown = murcko_scaffold(build("COc1cc2ncnc(Nc3ccccc3)c2cc1OC1CCNC1"))

    assert plain != grown
    assert shares_chemotype(plain, grown) is True


def test_a_different_chemotype_shares_nothing() -> None:
    quinazoline = murcko_scaffold(build("COc1cc2ncnc(Nc3ccccc3)c2cc1OC"))
    indole = murcko_scaffold(build("c1ccc2[nH]ccc2c1"))

    assert shares_chemotype(quinazoline, indole) is False


def test_an_acyclic_molecule_shares_any_chemotype() -> None:
    """An acyclic ligand has no scaffold to compare, so it must not be flagged."""
    assert shares_chemotype("", "c1ccccc1") is True
