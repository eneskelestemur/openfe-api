"""Tests for reading structures and selecting molecules inside them."""

from __future__ import annotations

from pathlib import Path

import pytest

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.structure import Structure
from openfe_api.schema.common import Selector

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture
def complex_structure() -> Structure:
    """Load the mini complex fixture.

    Returns:
        The parsed structure, holding a protein chain, two ligand copies and a cofactor.
    """
    return Structure.load(DATA / "mini_complex.cif")


def test_load_reads_mmcif(complex_structure: Structure) -> None:
    assert complex_structure.chain_ids() == ["A", "L", "M", "C"]


def test_load_reads_pdb() -> None:
    structure = Structure.load(DATA / "mini_protein.pdb")

    assert structure.chain_ids() == ["A"]
    assert len(structure.residues()) == 12


def test_load_rejects_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "model.xyz"
    path.write_text("not a structure\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="unsupported structure format"):
        Structure.load(path)


def test_load_reports_unparsable_file(tmp_path: Path) -> None:
    path = tmp_path / "model.pdb"
    path.write_text("this is not a pdb file\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="could not be parsed"):
        Structure.load(path)


def test_select_one_residue_by_chain(complex_structure: Structure) -> None:
    residue = complex_structure.select_one_residue(Selector(chain="L"), "ligand")

    assert residue.name == "LIG"
    assert residue.chain_id == "L"


def test_select_one_residue_rejects_no_match(complex_structure: Structure) -> None:
    with pytest.raises(InputValidationError, match="no residue matches the ligand selector"):
        complex_structure.select_one_residue(Selector(chain="Z"), "ligand")


def test_select_one_residue_rejects_several_matches(complex_structure: Structure) -> None:
    with pytest.raises(InputValidationError, match="matches 2 residues"):
        complex_structure.select_one_residue(Selector(resname="LIG"), "ligand")


def test_select_residues_combines_criteria(complex_structure: Structure) -> None:
    matches = complex_structure.select_residues(Selector(chain="M", resname="LIG"))

    assert len(matches) == 1
    assert matches[0].chain_id == "M"


def test_find_copies_matches_on_composition(complex_structure: Structure) -> None:
    ligand = complex_structure.select_one_residue(Selector(chain="L"), "ligand")

    copies = complex_structure.find_copies(ligand)

    assert [copy.chain_id for copy in copies] == ["M"]


def test_find_copies_ignores_other_molecules(complex_structure: Structure) -> None:
    cofactor = complex_structure.select_one_residue(Selector(chain="C"), "cofactor")

    assert complex_structure.find_copies(cofactor) == []


def test_residue_lookup_rejects_unknown_index(complex_structure: Structure) -> None:
    with pytest.raises(InputValidationError, match="no residue with topology index"):
        complex_structure.residue(9999)


def test_residue_pdb_block_holds_only_that_residue(complex_structure: Structure) -> None:
    ligand = complex_structure.select_one_residue(Selector(chain="L"), "ligand")

    block = complex_structure.residue_pdb_block(ligand)

    atom_lines = [line for line in block.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert len(atom_lines) == ligand.n_atoms


def test_write_pdb_rejects_an_empty_selection(complex_structure: Structure, tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="no residues selected"):
        complex_structure.write_pdb(set(), tmp_path / "empty.pdb")


def test_describe_names_chain_and_residue(complex_structure: Structure) -> None:
    ligand = complex_structure.select_one_residue(Selector(chain="L"), "ligand")

    assert ligand.describe() == "chain L residue LIG 1"
