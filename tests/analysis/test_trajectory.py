"""Tests for measuring how far a ligand and its protein move."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import write_mini_pdb
from openfe_api.analysis.trajectory import (
    RepeatMetrics,
    TrajectorySource,
    find_sources,
    measure_repeat,
    summarize_run,
)
from openfe_api.exceptions import InputValidationError


def _repeat_with(tmp_path: Path, **files: int) -> Path:
    """Build a repeat directory holding PDBs of the given atom counts."""
    repeat = tmp_path / "repeat1"
    for name, atoms in files.items():
        shared = repeat / f"shared_{name}Unit-abc_attempt_0"
        shared.mkdir(parents=True)
        write_mini_pdb(shared / "alchemical_system.pdb", protein=atoms, ligands=1)
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
    topology = write_mini_pdb(tmp_path / "apo.pdb", protein=4)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "apo", 1)

    assert len(measured) == 1
    assert measured[0].ligand == ""
    assert measured[0].ligand_drift == []
    assert measured[0].protein_rmsd


def test_each_ligand_is_measured_separately(tmp_path: Path) -> None:
    """A SepTop end state carries both ligands of its edge."""
    topology = write_mini_pdb(tmp_path / "pair.pdb", protein=4, ligands=2)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "edge", 1)

    assert len(measured) == 2
    assert {entry.ligand for entry in measured} == {"UNK1", "UNK2"}


def test_drift_is_left_out_without_a_protein(tmp_path: Path) -> None:
    """Aligned to itself, a ligand in solvent would report zero drift whatever it did."""
    topology = write_mini_pdb(tmp_path / "solvent.pdb", ligands=1)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "solvent", 1)

    assert measured[0].ligand_drift == []
    assert measured[0].ligand_rmsd


def test_frame_times_are_recorded(tmp_path: Path) -> None:
    topology = write_mini_pdb(tmp_path / "apo.pdb", protein=4)
    source = TrajectorySource(topology=topology, trajectory=topology, phase="", multistate=False)

    measured = measure_repeat(source, "apo", 1)

    assert len(measured[0].times_ps) == 1
    assert np.isfinite(measured[0].times_ps[0])


def test_phases_of_one_repeat_count_as_one_repeat() -> None:
    """An alchemical repeat is measured twice, once per phase."""
    metrics = [
        RepeatMetrics(
            run="lig",
            repeat=1,
            phase="complex",
            ligand="UNK",
            ligand_drift=[0.0, 1.0],
            ligand_rmsd=[0.0, 1.0],
        ),
        RepeatMetrics(run="lig", repeat=1, phase="solvent", ligand="UNK", ligand_rmsd=[0.0, 4.0]),
    ]

    summary = summarize_run("lig", metrics)

    assert summary.repeats == 1


def test_solvent_flexing_is_not_averaged_into_the_bound_rmsd() -> None:
    """A ligand's RMSD in a solvent box is its own flexing, not pose movement."""
    metrics = [
        RepeatMetrics(
            run="lig",
            repeat=1,
            phase="complex",
            ligand="UNK",
            ligand_drift=[0.0, 1.0],
            ligand_rmsd=[1.0, 1.0],
        ),
        RepeatMetrics(run="lig", repeat=1, phase="solvent", ligand="UNK", ligand_rmsd=[9.0, 9.0]),
    ]

    summary = summarize_run("lig", metrics)

    assert summary.mean_ligand_rmsd == pytest.approx(1.0)
