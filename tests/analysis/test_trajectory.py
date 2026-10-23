"""Tests for measuring how far a ligand and its protein move."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openfe_api.analysis.trajectory import (
    RepeatMetrics,
    TrajectorySource,
    find_sources,
    measure_repeat,
    summarize_run,
)
from openfe_api.exceptions import InputValidationError


def _pdb(path: Path, protein: int = 0, ligands: int = 0) -> Path:
    """Write a minimal PDB: a CA-only protein and whole methane molecules named UNK.

    Carries a box, explicit hydrogens and CONECT records, as the topologies OpenFE writes
    do. The alignment needs the box, fragment detection needs the bonds, and the
    symmetry-corrected RMSD converts to RDKit, which needs the hydrogens.
    """
    lines = ["CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1"]
    serial = 1
    for index in range(protein):
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA A{index + 1:4d}    "
            f"{index * 4.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1

    offsets = (
        (0.0, 0.0, 0.0),
        (0.6, 0.6, 0.6),
        (-0.6, -0.6, 0.6),
        (0.6, -0.6, -0.6),
        (-0.6, 0.6, -0.6),
    )
    bonds: list[tuple[int, int]] = []
    for residue in range(ligands):
        carbon = serial
        base = 10.0 + 5.0 * residue
        for position, (dx, dy, dz) in enumerate(offsets):
            element = "C" if position == 0 else "H"
            lines.append(
                f"HETATM{serial:5d}  {element}{position:<3d}UNK L{residue + 1:4d}    "
                f"{base + dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00  0.00           {element}"
            )
            serial += 1
        bonds += [(carbon, carbon + offset) for offset in range(1, 5)]

    lines += [f"CONECT{a:5d}{b:5d}" for a, b in bonds]
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


def _repeat_with(tmp_path: Path, **files: int) -> Path:
    """Build a repeat directory holding PDBs of the given atom counts."""
    repeat = tmp_path / "repeat1"
    for name, atoms in files.items():
        shared = repeat / f"shared_{name}Unit-abc_attempt_0"
        shared.mkdir(parents=True)
        _pdb(shared / "alchemical_system.pdb", protein=atoms, ligands=1)
    return repeat


def test_no_trajectory_means_no_source(tmp_path: Path) -> None:
    assert find_sources(_repeat_with(tmp_path, Setup=10)) == []


def test_a_missing_repeat_directory_is_empty(tmp_path: Path) -> None:
    assert find_sources(tmp_path / "absent") == []


def test_a_trajectory_without_a_matching_topology_is_skipped(tmp_path: Path) -> None:
    """Pairing on the wrong topology would read a different molecule."""
    repeat = _repeat_with(tmp_path, Setup=10)
    trajectory = repeat / "shared_SimUnit-abc_attempt_0" / "simulation.xtc"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_bytes(b"not a trajectory")

    assert find_sources(repeat) == []


def test_metrics_aggregate_across_repeats() -> None:
    metrics = [
        RepeatMetrics(
            run="lig",
            repeat=1,
            phase="complex",
            ligand="UNK",
            times_ps=[0.0, 1.0],
            ligand_drift=[0.0, 1.0],
            ligand_rmsd=[0.0, 0.5],
            protein_rmsd=[0.0, 1.5],
        ),
        RepeatMetrics(
            run="lig",
            repeat=2,
            phase="complex",
            ligand="UNK",
            times_ps=[0.0, 1.0],
            ligand_drift=[0.0, 3.0],
            ligand_rmsd=[0.0, 1.5],
            protein_rmsd=[0.0, 0.5],
        ),
    ]

    summary = summarize_run("lig", metrics)

    assert summary.repeats == 2
    assert summary.max_drift == pytest.approx(3.0)
    assert summary.mean_drift == pytest.approx(1.0)
    assert summary.worst_repeat == 2


def test_aggregating_a_run_with_no_frames_reports_nothing() -> None:
    """A run that produced nothing must not read as a run that produced zeros."""
    summary = summarize_run("lig", [RepeatMetrics(run="lig", repeat=1, phase="complex")])

    assert summary.repeats == 1
    assert summary.mean_drift is None
    assert summary.worst_repeat is None


def test_aggregating_nothing_at_all_reports_nothing() -> None:
    assert summarize_run("lig", []).mean_drift is None


def test_a_system_with_neither_protein_nor_ligand_is_refused(tmp_path: Path) -> None:
    water = tmp_path / "water.pdb"
    lines = [
        f"HETATM{index:5d}  O   HOH W{index:4d}    "
        f"{index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           O"
        for index in range(1, 4)
    ]
    water.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    source = TrajectorySource(topology=water, trajectory=water, phase="", multistate=False)

    with pytest.raises(InputValidationError, match="neither a protein nor a ligand"):
        measure_repeat(source, "run", 1)


def test_a_protein_only_system_is_measured_without_a_ligand(tmp_path: Path) -> None:
    """An apo run is a normal input, not a failure."""
    topology = _pdb(tmp_path / "apo.pdb", protein=4)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "apo", 1)

    assert len(measured) == 1
    assert measured[0].ligand == ""
    assert measured[0].ligand_drift == []
    assert measured[0].protein_rmsd


def test_each_ligand_is_measured_separately(tmp_path: Path) -> None:
    """A SepTop end state carries both ligands of its edge."""
    topology = _pdb(tmp_path / "pair.pdb", protein=4, ligands=2)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "edge", 1)

    assert len(measured) == 2
    assert {entry.ligand for entry in measured} == {"UNK1", "UNK2"}


def test_drift_is_left_out_without_a_protein(tmp_path: Path) -> None:
    """Aligned to itself, a ligand in solvent would report zero drift whatever it did."""
    topology = _pdb(tmp_path / "solvent.pdb", ligands=1)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "solvent", 1)

    assert measured[0].ligand_drift == []
    assert measured[0].ligand_rmsd


def test_frame_times_are_recorded(tmp_path: Path) -> None:
    topology = _pdb(tmp_path / "apo.pdb", protein=4)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "apo", 1)

    assert len(measured[0].times_ps) == 1
    assert np.isfinite(measured[0].times_ps[0])
