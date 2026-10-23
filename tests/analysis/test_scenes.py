"""Tests for writing a pose out as something you can look at."""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import pytest

from helpers import write_mini_pdb
from openfe_api.analysis.scenes import representative_frame, write_scene
from openfe_api.analysis.trajectory import TrajectorySource
from openfe_api.exceptions import InputValidationError


def _source(path: Path) -> TrajectorySource:
    """Use a topology as its own single-frame trajectory."""
    return TrajectorySource(topology=path, trajectory=path, phase="", multistate=False)


def test_a_scene_is_written_as_three_files(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "complex.pdb", protein=4, ligands=1)

    scene = write_scene(_source(topology), "mini", tmp_path / "scenes")

    assert scene.structure.is_file()
    assert scene.script.is_file()
    assert scene.page.is_file()


def test_the_script_names_the_structure_beside_it(tmp_path: Path) -> None:
    """It is opened from inside the scenes directory, so the path has to be relative."""
    topology = write_mini_pdb(tmp_path / "complex.pdb", protein=4, ligands=1)

    scene = write_scene(_source(topology), "mini", tmp_path / "scenes")

    script = scene.script.read_text(encoding="utf-8")
    assert "load mini.pdb" in script
    assert str(tmp_path) not in script


def test_the_page_carries_the_structure(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "complex.pdb", protein=4, ligands=1)

    scene = write_scene(_source(topology), "mini", tmp_path / "scenes")

    assert "HETATM" in scene.page.read_text(encoding="utf-8")


def test_a_system_without_a_ligand_has_no_scene(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "apo.pdb", protein=4)

    with pytest.raises(InputValidationError, match="no ligand"):
        write_scene(_source(topology), "apo", tmp_path / "scenes")


def test_the_median_frame_is_chosen_not_the_last(tmp_path: Path) -> None:
    """One frame where the ligand swung away must not decide which pose is shown."""
    topology = write_mini_pdb(tmp_path / "complex.pdb", protein=4, ligands=1)
    universe = mda.Universe(topology)
    atoms = universe.select_atoms("all")
    trajectory = tmp_path / "moving.xtc"
    with mda.Writer(str(trajectory), n_atoms=len(atoms)) as writer:
        for shift in (0.0, 0.1, 0.2, 40.0):
            atoms.positions = atoms.positions + shift
            writer.write(atoms)
    universe.trajectory.close()

    moving = mda.Universe(topology, trajectory)
    try:
        frame = representative_frame(moving, moving.select_atoms("resname UNK"))
    finally:
        moving.trajectory.close()

    assert frame != 3
