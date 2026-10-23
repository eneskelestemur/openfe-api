"""Measuring how far a ligand and its protein move during a run.

The analyses come from ``openfe_analysis``, so the numbers match what OpenFE's own tooling
reports. An alchemical run is read at its fully interacting end state; plain MD is read
whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import MDAnalysis as mda
import netCDF4
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader
from openfe_analysis.rmsd import (
    LigandCOMDrift,
    RMSDAnalysis,
    SymmetryCorrectedLigandRMSD,
    apply_complex_alignment_transformations,
    apply_ligand_alignment_transformations,
    create_universe_single_state,
)

from openfe_api.exceptions import InputValidationError
from openfe_api.log import get_logger

__all__ = [
    "END_STATE",
    "LIGAND_SELECTION",
    "RepeatMetrics",
    "RunStability",
    "TrajectorySource",
    "find_sources",
    "measure_repeat",
    "summarize_run",
]

logger = get_logger(__name__)

END_STATE = 0
"""The fully interacting Hamiltonian state of a multistate trajectory."""

LIGAND_SELECTION = "resname UNK"
"""What OpenFE names a small molecule in the topologies it writes."""

PROTEIN_SELECTION = "protein"
BACKBONE_SELECTION = "protein and name CA"


@dataclass(frozen=True)
class TrajectorySource:
    """A trajectory and the topology that matches it.

    Attributes:
        topology: Structure file whose atom count matches the trajectory.
        trajectory: The trajectory itself.
        phase: ``complex``, ``solvent``, or empty for plain MD.
        multistate: Whether the trajectory holds every Hamiltonian state.
    """

    topology: Path
    trajectory: Path
    phase: str
    multistate: bool


@dataclass
class RepeatMetrics:
    """What one repeat's trajectory shows.

    Attributes:
        run: Run name.
        repeat: One-based repeat index.
        phase: Which phase the trajectory belongs to.
        ligand: Which ligand was followed, empty for a system without one.
        times_ps: Simulation time of each frame.
        protein_rmsd: Backbone RMSD per frame, in angstrom. Empty without a protein.
        ligand_rmsd: Symmetry-corrected ligand RMSD per frame, in angstrom.
        ligand_drift: Ligand centre-of-mass drift per frame, in angstrom. Empty
            without a protein to measure it against.
    """

    run: str
    repeat: int
    phase: str
    ligand: str = ""
    times_ps: list[float] = field(default_factory=list)
    protein_rmsd: list[float] = field(default_factory=list)
    ligand_rmsd: list[float] = field(default_factory=list)
    ligand_drift: list[float] = field(default_factory=list)


@dataclass
class RunStability:
    """One run's trajectories, reduced to the numbers that say whether the pose held.

    Attributes:
        run: Run name.
        repeats: How many repeats contributed.
        mean_drift: Mean ligand centre-of-mass drift across every frame, in angstrom.
        max_drift: Largest drift any frame reached, in angstrom.
        mean_ligand_rmsd: Mean ligand RMSD, in angstrom.
        mean_protein_rmsd: Mean backbone RMSD, in angstrom.
        worst_repeat: The repeat holding ``max_drift``.
    """

    run: str
    repeats: int
    mean_drift: float | None = None
    max_drift: float | None = None
    mean_ligand_rmsd: float | None = None
    mean_protein_rmsd: float | None = None
    worst_repeat: int | None = None


def _pdb_atoms(path: Path) -> int:
    """Count the atoms a PDB declares."""
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(("ATOM", "HETATM")))


def _matching_topology(repeat_dir: Path, atoms: int) -> Path | None:
    """Find the topology whose atom count matches a trajectory.

    A repeat holds several: one per phase, and several alchemical states besides. Only the
    count tells them apart, and pairing the wrong one reads as a different molecule.
    """
    for candidate in sorted(repeat_dir.glob("shared_*/*.pdb")):
        if _pdb_atoms(candidate) == atoms:
            return candidate
    return None


def find_sources(repeat_dir: Path) -> list[TrajectorySource]:
    """Find the trajectories one repeat produced, each with its matching topology.

    An alchemical repeat also writes equilibration trajectories, so once a multistate
    trajectory is found the rest are left alone.

    Args:
        repeat_dir: A repeat's working directory.

    Returns:
        One source per trajectory that has a topology to read it with, in name order.
    """
    sources: list[TrajectorySource] = []

    for dataset in sorted(repeat_dir.glob("shared_*/*.nc")):
        if dataset.name.endswith("_checkpoint.nc"):
            continue
        with netCDF4.Dataset(dataset) as handle:
            atoms = handle.dimensions["atom"].size
        topology = _matching_topology(repeat_dir, atoms)
        if topology is None:
            logger.warning("No %d atom topology beside %s; skipping it", atoms, dataset.name)
            continue
        sources.append(
            TrajectorySource(
                topology=topology, trajectory=dataset, phase=dataset.stem, multistate=True
            )
        )

    if sources:
        return sources

    for dataset in sorted(repeat_dir.glob("shared_*/*.xtc")):
        try:
            atoms = XTCReader(str(dataset)).n_atoms
        except OSError as error:
            logger.warning("Could not read %s: %s", dataset.name, error)
            continue
        topology = _matching_topology(repeat_dir, atoms)
        if topology is None:
            logger.debug("No topology matches %s; skipping it", dataset.name)
            continue
        sources.append(
            TrajectorySource(topology=topology, trajectory=dataset, phase="", multistate=False)
        )

    return sources


def _open(source: TrajectorySource) -> mda.Universe:
    """Open a trajectory, aligned so motion of the box is not read as motion of the ligand."""
    if source.multistate:
        try:
            universe = create_universe_single_state(
                source.topology, netCDF4.Dataset(source.trajectory), END_STATE
            )
        except IndexError as error:
            # The reader works out its frame spacing from consecutive frames, so a state
            # sampled once has nothing to difference.
            raise InputValidationError(
                f"{source.trajectory}: too few frames at the end state to measure. A run "
                "this short says nothing about whether a pose held."
            ) from error
    else:
        universe = mda.Universe(source.topology, source.trajectory)

    ligands = universe.select_atoms(LIGAND_SELECTION)
    protein = universe.select_atoms(PROTEIN_SELECTION)

    if len(protein) and len(ligands):
        apply_complex_alignment_transformations(universe, protein, [ligands])
    elif len(ligands):
        apply_ligand_alignment_transformations(universe, ligands)
    return universe


def measure_repeat(source: TrajectorySource, run: str, repeat: int) -> list[RepeatMetrics]:
    """Measure one trajectory, one row per ligand it carries.

    A SepTop end state holds both ligands of its edge, and an apo system holds none, so the
    number of rows follows the system rather than being one.

    Args:
        source: The trajectory and its topology.
        run: Run name, for the record.
        repeat: One-based repeat index, for the record.

    Returns:
        One entry per ligand, or a single protein-only entry for a system with no ligand.
    """
    universe = _open(source)
    backbone = universe.select_atoms(BACKBONE_SELECTION)
    protein_rmsd = (
        [float(value) for value in RMSDAnalysis(backbone, superposition=True).run().results["rmsd"]]
        if len(backbone)
        else []
    )
    times = [float(frame.time) for frame in universe.trajectory]
    residues = universe.select_atoms(LIGAND_SELECTION).residues

    if not len(residues):
        if not protein_rmsd:
            raise InputValidationError(
                f"{source.trajectory}: holds neither a protein nor a ligand named "
                f"'{LIGAND_SELECTION}', so there is nothing to measure."
            )
        return [
            RepeatMetrics(
                run=run,
                repeat=repeat,
                phase=source.phase,
                times_ps=times,
                protein_rmsd=protein_rmsd,
            )
        ]

    measured: list[RepeatMetrics] = []
    for residue in residues:
        ligand = residue.atoms
        label = f"{residue.resname}{residue.resid}" if len(residues) > 1 else residue.resname
        measured.append(
            RepeatMetrics(
                run=run,
                repeat=repeat,
                phase=source.phase,
                ligand=label,
                times_ps=times,
                protein_rmsd=protein_rmsd,
                ligand_rmsd=[
                    float(value)
                    for value in SymmetryCorrectedLigandRMSD(ligand).run().results["rmsd"]
                ],
                ligand_drift=(
                    [float(value) for value in LigandCOMDrift(ligand).run().results["com_drift"]]
                    if len(backbone)
                    else []
                ),
            )
        )
        logger.info(
            "Measured '%s' repeat %d%s %s: %d frames, drift up to %.2f A",
            run,
            repeat,
            f" ({source.phase})" if source.phase else "",
            label,
            len(times),
            max(measured[-1].ligand_drift, default=0.0),
        )
    return measured


def summarize_run(run: str, metrics: list[RepeatMetrics]) -> RunStability:
    """Reduce a run's repeats to one row.

    Args:
        run: Run name.
        metrics: Every repeat measured for it.

    Returns:
        The aggregate. Values are None when no repeat produced a frame.
    """
    drift = [value for entry in metrics for value in entry.ligand_drift]
    ligand = [value for entry in metrics for value in entry.ligand_rmsd]
    protein = [value for entry in metrics for value in entry.protein_rmsd]
    summary = RunStability(run=run, repeats=len(metrics))
    if not drift:
        return summary

    summary.mean_drift = float(np.mean(drift))
    summary.max_drift = float(np.max(drift))
    summary.mean_ligand_rmsd = float(np.mean(ligand)) if ligand else None
    summary.mean_protein_rmsd = float(np.mean(protein)) if protein else None
    summary.worst_repeat = max(
        (entry for entry in metrics if entry.ligand_drift),
        key=lambda entry: max(entry.ligand_drift),
    ).repeat
    return summary
