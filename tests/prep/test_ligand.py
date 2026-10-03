"""Tests for building molecules from a SMILES and a pose."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.ligand import (
    build_from_file,
    build_from_smiles,
    build_molecule,
    parse_template,
    unspecified_stereo,
)
from openfe_api.prep.structure import Structure
from openfe_api.schema.common import Selector

DATA = Path(__file__).parents[1] / "data"

LIGAND_SMILES = "C[C@H](O)c1ccccc1"
WRONG_ENANTIOMER = "C[C@@H](O)c1ccccc1"
COFACTOR_SMILES = "CC(=O)[O-]"


@pytest.fixture
def ligand_block() -> str:
    """Return the PDB block of the mini complex's ligand.

    Returns:
        PDB text for the ligand in chain L, without hydrogens or bond orders.
    """
    structure = Structure.load(DATA / "mini_complex.cif")
    ligand = structure.select_one_residue(Selector(chain="L"), "ligand")
    return structure.residue_pdb_block(ligand)


def test_build_molecule_recovers_bond_orders_and_adds_hydrogens(ligand_block: str) -> None:
    prepared = build_molecule(LIGAND_SMILES, ligand_block, "phenylethanol")

    assert prepared.name == "phenylethanol"
    assert prepared.net_charge == 0
    assert prepared.molecule.GetNumConformers() == 1
    assert any(atom.GetAtomicNum() == 1 for atom in prepared.molecule.GetAtoms())
    aromatic = sum(1 for bond in prepared.molecule.GetBonds() if bond.GetIsAromatic())
    assert aromatic == 6


def test_build_molecule_rejects_the_wrong_enantiomer(ligand_block: str) -> None:
    with pytest.raises(InputValidationError, match="stereochemistry of the pose does not match"):
        build_molecule(WRONG_ENANTIOMER, ligand_block, "phenylethanol")


def test_build_molecule_rejects_a_different_molecule(ligand_block: str) -> None:
    with pytest.raises(InputValidationError, match="heavy atoms but the SMILES has"):
        build_molecule("CCO", ligand_block, "phenylethanol")


def test_build_molecule_rejects_same_size_wrong_connectivity(ligand_block: str) -> None:
    with pytest.raises(InputValidationError, match="could not be matched"):
        build_molecule("CCCCCCCCC", ligand_block, "phenylethanol")


def test_parse_template_rejects_unparsable_smiles() -> None:
    with pytest.raises(InputValidationError, match="could not be parsed"):
        parse_template("not a smiles((", "ligand")


def test_parse_template_accepts_undefined_stereochemistry() -> None:
    template = parse_template("CC(O)c1ccccc1", "ligand")

    assert unspecified_stereo(template) == ({1}, set())


def test_undefined_stereocenter_is_taken_from_the_pose(ligand_block: str) -> None:
    """A racemic input leaves the center undefined, so the pose decides it."""
    prepared = build_molecule("CC(O)c1ccccc1", ligand_block, "phenylethanol")

    assert prepared.smiles == Chem.CanonSmiles(LIGAND_SMILES)
    assert any("taken from the pose" in warning for warning in prepared.warnings)


def test_adopted_stereocenter_is_named_in_the_warning(ligand_block: str) -> None:
    prepared = build_molecule("CC(O)c1ccccc1", ligand_block, "phenylethanol")

    warning = next(w for w in prepared.warnings if "taken from the pose" in w)
    assert "(S)" in warning or "(R)" in warning


def test_defined_stereochemistry_still_must_match(ligand_block: str) -> None:
    """Only undefined centers are adopted; a defined one that disagrees is rejected."""
    with pytest.raises(InputValidationError, match="does not match the SMILES"):
        build_molecule(WRONG_ENANTIOMER, ligand_block, "phenylethanol")


def test_partly_defined_stereochemistry_is_checked_per_center() -> None:
    """One center defined and one left open: the defined one is enforced."""
    smiles = "C[C@H](O)[C@@H](N)CC(=O)O"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    block = Chem.MolToPDBBlock(Chem.RemoveHs(molecule))

    partly_defined = build_molecule("C[C@H](O)C(N)CC(=O)O", block, "partial")
    assert any("taken from the pose" in warning for warning in partly_defined.warnings)

    with pytest.raises(InputValidationError, match="does not match the SMILES"):
        build_molecule("C[C@@H](O)C(N)CC(=O)O", block, "partial")


def test_parse_template_accepts_defined_stereochemistry() -> None:
    template = parse_template(LIGAND_SMILES, "ligand")

    assert template.GetNumAtoms() == 9


def test_phosphate_stereochemistry_is_not_treated_as_a_mismatch() -> None:
    """A phosphate's terminal oxygens are equivalent, so its 3D chirality is not real."""
    smiles = "COP(=O)([O-])OC"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    block = Chem.MolToPDBBlock(Chem.RemoveHs(molecule))

    prepared = build_molecule(smiles, block, "phosphate")

    assert prepared.net_charge == -1


def test_neutralize_changes_the_molecule_and_warns() -> None:
    structure = Structure.load(DATA / "mini_complex.cif")
    cofactor = structure.select_one_residue(Selector(chain="C"), "cofactor")
    block = structure.residue_pdb_block(cofactor)

    prepared = build_molecule(COFACTOR_SMILES, block, "acetate", neutralize=True)

    assert prepared.net_charge == 0
    assert prepared.neutralized is True
    assert any("NEUTRALIZED" in warning for warning in prepared.warnings)


def test_neutralize_is_off_by_default() -> None:
    structure = Structure.load(DATA / "mini_complex.cif")
    cofactor = structure.select_one_residue(Selector(chain="C"), "cofactor")

    prepared = build_molecule(COFACTOR_SMILES, structure.residue_pdb_block(cofactor), "acetate")

    assert prepared.net_charge == -1
    assert prepared.neutralized is False


def test_neutralize_rejects_an_irreducible_charge() -> None:
    """A quaternary ammonium cannot be neutralized by changing protonation."""
    smiles = "C[N+](C)(C)C"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=0xF00D)
    block = Chem.MolToPDBBlock(Chem.RemoveHs(molecule))

    with pytest.raises(InputValidationError, match="still carries a net charge"):
        build_molecule(smiles, block, "quaternary", neutralize=True)


def test_build_from_file_reads_an_sdf() -> None:
    prepared = build_from_file(LIGAND_SMILES, DATA / "ligand.sdf", "phenylethanol")

    assert prepared.net_charge == 0
    assert prepared.molecule.GetNumConformers() == 1


def test_build_from_file_checks_the_smiles_matches() -> None:
    with pytest.raises(InputValidationError, match="stereochemistry"):
        build_from_file(WRONG_ENANTIOMER, DATA / "ligand.sdf", "phenylethanol")


def test_build_from_file_rejects_an_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "ligand.xyz"
    path.write_text("nope\n", encoding="utf-8")

    with pytest.raises(InputValidationError, match="unsupported molecule format"):
        build_from_file(LIGAND_SMILES, path, "phenylethanol")


def test_build_from_file_rejects_multi_molecule_files(tmp_path: Path) -> None:
    path = tmp_path / "two.sdf"
    source = Chem.SDMolSupplier(str(DATA / "ligand.sdf"), removeHs=False)
    molecule = next(iter(source))
    with Chem.SDWriter(str(path)) as writer:
        writer.write(molecule)
        writer.write(molecule)

    with pytest.raises(InputValidationError, match="holds 2 molecules"):
        build_from_file(LIGAND_SMILES, path, "phenylethanol")


def test_build_from_file_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.sdf"
    path.write_text("", encoding="utf-8")

    with pytest.raises(InputValidationError, match=r"could not be read|no readable molecule"):
        build_from_file(LIGAND_SMILES, path, "phenylethanol")


def test_build_from_smiles_needs_no_pose() -> None:
    prepared = build_from_smiles("C[C@H](O)c1ccccc1", "alcohol")

    assert prepared.molecule.GetNumConformers() == 0
    assert prepared.molecule.GetNumAtoms() == 19
    assert prepared.smiles == "C[C@H](O)c1ccccc1"
    assert prepared.net_charge == 0


def test_build_from_smiles_rejects_an_undefined_stereocenter() -> None:
    """Without a pose there is nothing to adopt the configuration from."""
    with pytest.raises(InputValidationError, match="undefined stereocenter"):
        build_from_smiles("CC(O)c1ccccc1", "racemic")


def test_build_from_smiles_names_the_undefined_center() -> None:
    with pytest.raises(InputValidationError, match=r"atom 1 \(C bonded to C,C,O\)"):
        build_from_smiles("CC(O)c1ccccc1", "racemic")


def test_build_from_smiles_rejects_an_undefined_double_bond() -> None:
    with pytest.raises(InputValidationError, match="without defined geometry"):
        build_from_smiles("CC=CCl", "alkene")


def test_build_from_smiles_accepts_a_defined_double_bond() -> None:
    prepared = build_from_smiles(r"C/C=C/Cl", "alkene")

    assert prepared.smiles == r"C/C=C/Cl"


def test_build_from_smiles_keeps_a_charge_by_default() -> None:
    prepared = build_from_smiles("CC(=O)[O-]", "acetate")

    assert prepared.net_charge == -1
    assert prepared.neutralized is False


def test_build_from_smiles_can_neutralize_with_a_warning() -> None:
    prepared = build_from_smiles("CC(=O)[O-]", "acetate", neutralize=True)

    assert prepared.net_charge == 0
    assert prepared.neutralized is True
    assert any("NEUTRALIZED" in warning for warning in prepared.warnings)


def test_build_from_smiles_rejects_an_unparseable_smiles() -> None:
    with pytest.raises(InputValidationError, match="could not be parsed"):
        build_from_smiles("c1cc(", "broken")
