"""Tests for fingerprinting the contacts a ligand makes."""

from __future__ import annotations

from pathlib import Path

from helpers import write_mini_pdb
from openfe_api.analysis.interactions import NOISY_INTERACTION, measure_interactions
from openfe_api.analysis.trajectory import TrajectorySource


def _source(path: Path) -> TrajectorySource:
    """Use a topology as its own single-frame trajectory."""
    return TrajectorySource(topology=path, trajectory=path, phase="", multistate=False)


def test_a_system_without_a_protein_has_no_contacts(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "solvent.pdb", ligands=1)

    assert measure_interactions(_source(topology), "solvent", 1) == []


def test_a_system_without_a_ligand_has_no_contacts(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "apo.pdb", protein=4)

    assert measure_interactions(_source(topology), "apo", 1) == []


def test_contacts_are_found_for_a_ligand_against_a_protein(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "complex.pdb", protein=6, ligands=1)

    contacts = measure_interactions(_source(topology), "complex", 1)

    assert contacts
    assert all(0.0 < entry.occupancy <= 1.0 for entry in contacts)
    assert all(entry.run == "complex" for entry in contacts)


def test_a_ligand_far_from_the_protein_has_no_contacts(tmp_path: Path) -> None:
    """The pocket is what sits near the ligand; nothing near it means nothing to report."""
    topology = write_mini_pdb(tmp_path / "far.pdb", protein=4, ligands=1, ligand_offset=60.0)

    assert measure_interactions(_source(topology), "far", 1) == []


def test_the_noisy_interaction_is_named_so_a_figure_can_drop_it() -> None:
    """VdWContact fires on nearly every nearby residue, which buries the rest."""
    assert NOISY_INTERACTION == "VdWContact"
