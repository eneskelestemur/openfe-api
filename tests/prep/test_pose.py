"""Tests for placing a ligand into the reference frame on the common core."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign

from openfe_api.exceptions import InputValidationError
from openfe_api.prep.mcs import find_core, match_core
from openfe_api.prep.pose import CLASH_DISTANCE, clash_score, generate_pose, superpose_on_core
from openfe_api.schema.rbfe import AlignmentSpec

REFERENCE = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
ANALOGUE = "CCc1ccc(cc1)C(=O)Nc1ccccc1"


@pytest.fixture
def reference() -> Chem.Mol:
    """Return the reference ligand with an embedded, minimized pose.

    Returns:
        The reference ligand holding one conformer.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(REFERENCE))
    rdDistGeom.EmbedMolecule(molecule, randomSeed=1)
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule)
    return molecule


def probe(smiles: str) -> Chem.Mol:
    """Build a hydrogen-complete molecule from SMILES, without coordinates.

    Args:
        smiles: The molecule's SMILES.

    Returns:
        The molecule.
    """
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_generated_pose_sits_on_the_reference_core(reference: Chem.Mol) -> None:
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)

    pose = generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=10))

    assert pose.core_rmsd is not None
    assert pose.core_rmsd < 0.1
    assert pose.molecule.GetNumConformers() == 1
    assert pose.conformers_accepted > 0


def test_generation_is_reproducible(reference: Chem.Mol) -> None:
    """Replanning a campaign must not silently move its ligands."""
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)
    spec = AlignmentSpec(n_conformers=5)

    first = generate_pose(molecule, reference, core, spec)
    second = generate_pose(molecule, reference, core, spec)

    assert first.molecule.GetConformer().GetPositions() == pytest.approx(
        second.molecule.GetConformer().GetPositions()
    )


def test_a_different_seed_gives_a_different_pose(reference: Chem.Mol) -> None:
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)
    spec = AlignmentSpec(n_conformers=5)

    first = generate_pose(molecule, reference, core, spec, seed=1)
    second = generate_pose(molecule, reference, core, spec, seed=2)

    assert first.molecule.GetConformer().GetPositions() != pytest.approx(
        second.molecule.GetConformer().GetPositions()
    )


def test_core_atoms_land_on_the_reference_coordinates(reference: Chem.Mol) -> None:
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)

    pose = generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=10))

    rmsd, atom_map = superpose_on_core(pose.molecule, reference, core)
    reference_positions = reference.GetConformer().GetPositions()
    pose_positions = pose.molecule.GetConformer().GetPositions()
    for probe_index, reference_index in atom_map:
        separation = np.linalg.norm(
            pose_positions[probe_index] - reference_positions[reference_index]
        )
        assert separation < 0.5
    assert rmsd < 0.1


def test_the_best_mapping_is_chosen_for_a_symmetric_core(reference: Chem.Mol) -> None:
    """Taking the first match of a symmetric core can place the molecule flipped."""
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)
    pose = generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=10))

    best, _ = superpose_on_core(pose.molecule, reference, core)

    reference_match = match_core(reference, core)[0]
    every = [
        rdMolAlign.AlignMol(
            Chem.Mol(pose.molecule),
            reference,
            atomMap=list(zip(probe_match, reference_match, strict=True)),
        )
        for probe_match in match_core(pose.molecule, core)
    ]
    assert best == pytest.approx(min(every), abs=1e-6)


def test_an_unreachable_core_limit_is_rejected(reference: Chem.Mol) -> None:
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)

    with pytest.raises(InputValidationError, match="no conformer placed the common core"):
        generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=2, max_core_rmsd=1e-9))


def test_clash_score_counts_close_contacts(reference: Chem.Mol) -> None:
    positions = reference.GetConformer().GetPositions()

    on_top = clash_score(reference, positions)
    far_away = clash_score(reference, positions + 50.0)

    assert on_top[0] > 0
    assert on_top[1] == pytest.approx(0.0, abs=1e-6)
    assert far_away[0] == 0
    assert far_away[1] is not None and far_away[1] > CLASH_DISTANCE


def test_clash_score_without_a_pocket_reports_nothing(reference: Chem.Mol) -> None:
    assert clash_score(reference, None) == (0, None)
    assert clash_score(reference, np.empty((0, 3))) == (0, None)


def test_clash_ranking_prefers_the_pose_that_fits(reference: Chem.Mol) -> None:
    """A wall of pocket atoms on one side must push the chosen conformer away from it."""
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)
    positions = reference.GetConformer().GetPositions()
    wall = positions + np.array([0.0, 0.0, 3.0])
    spec = AlignmentSpec(n_conformers=20, rank_by="clash")

    without = generate_pose(molecule, reference, core, spec)
    with_wall = generate_pose(molecule, reference, core, spec, pocket=wall)

    assert without.worst_contact is None
    assert with_wall.worst_contact is not None
    assert with_wall.clashes <= clash_score(without.molecule, wall)[0]


def test_energy_ranking_is_available(reference: Chem.Mol) -> None:
    molecule = probe(ANALOGUE)
    core = find_core(reference, molecule)

    pose = generate_pose(
        molecule, reference, core, AlignmentSpec(n_conformers=10, rank_by="energy")
    )

    assert pose.core_rmsd is not None
    assert pose.core_rmsd < 0.1


DRUG_REFERENCE = "COc1cc2ncnc(Nc3ccccc3)c2cc1OC"
DRUG_ANALOGUE = "COc1cc2ncnc(Nc3ccccc3)c2cc1OC1CCNC1"


def test_a_drug_sized_ligand_with_a_large_core_is_placed() -> None:
    """Embedding with the core pinned over-constrains the geometry and returns nothing.

    On a 45 atom ligand with a 20 atom core that approach took minutes per conformer and
    produced none, so this is the case the small test molecules could not reach.
    """
    reference = Chem.AddHs(Chem.MolFromSmiles(DRUG_REFERENCE))
    rdDistGeom.EmbedMolecule(reference, randomSeed=5)
    rdForceFieldHelpers.MMFFOptimizeMolecule(reference)
    molecule = Chem.AddHs(Chem.MolFromSmiles(DRUG_ANALOGUE))
    core = find_core(reference, molecule)

    assert core.size >= 20
    pose = generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=10))

    assert pose.conformers_accepted > 0
    assert pose.core_rmsd is not None
    assert pose.core_rmsd < 0.1


def test_the_core_lands_on_the_reference_for_a_drug_sized_ligand() -> None:
    reference = Chem.AddHs(Chem.MolFromSmiles(DRUG_REFERENCE))
    rdDistGeom.EmbedMolecule(reference, randomSeed=5)
    rdForceFieldHelpers.MMFFOptimizeMolecule(reference)
    molecule = Chem.AddHs(Chem.MolFromSmiles(DRUG_ANALOGUE))
    core = find_core(reference, molecule)

    pose = generate_pose(molecule, reference, core, AlignmentSpec(n_conformers=10))

    _, atom_map = superpose_on_core(pose.molecule, reference, core)
    reference_positions = reference.GetConformer().GetPositions()
    pose_positions = pose.molecule.GetConformer().GetPositions()
    for probe_index, reference_index in atom_map:
        separation = np.linalg.norm(
            pose_positions[probe_index] - reference_positions[reference_index]
        )
        assert separation < 0.5
