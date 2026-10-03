"""Turning finished runs into per-edge and per-ligand relative free energies.

RBFE and SepTop differ only in how one edge is read, so the per-edge reader is passed in and
everything after it is shared.

Per-ligand values come from a maximum likelihood fit over the network and are relative to its
mean, not absolute binding free energies.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from cinnabar import FEMap

from openfe_api.graph import connected_components, unreachable_names
from openfe_api.log import get_logger
from openfe_api.protocols.rbfe import PHASES
from openfe_api.results.gather import KCAL, as_array, load_result_json, to_kcal

__all__ = [
    "CycleClosure",
    "EdgeReader",
    "EdgeResult",
    "LigandEstimate",
    "NetworkResults",
    "PhaseResult",
    "combine_repeats",
    "gather_network",
    "read_phase",
    "read_rbfe_edge",
    "write_edge_tsv",
    "write_ligand_tsv",
]

type EdgeReader = Callable[[Path, str, str, str], EdgeResult]
"""Reads one edge of a network: campaign directory, edge name, and its two ligand names."""

logger = get_logger(__name__)

FIT_SOURCE = "openfe-api"
"""Source label every edge shares, so the fit runs once over the whole network."""


@dataclass
class PhaseResult:
    """One repeat of one phase of one edge.

    Attributes:
        path: The result file this came from.
        phase: ``solvent`` or ``complex``.
        repeat: One-based repeat index taken from the directory name.
        estimate: Free energy of mutating ligand A into ligand B in this phase, in kcal/mol.
        uncertainty: MBAR uncertainty, in kcal/mol.
        production_iterations: Iterations of production sampling analyzed.
        overlap_matrix: MBAR overlap matrix, if reported.
        exchange_matrix: Replica exchange probability matrix, if reported.
        forward_reverse: Forward and reverse convergence data, if reported.
        failure: Why this repeat is unusable, if it is.
    """

    path: Path
    phase: str
    repeat: int
    estimate: float | None = None
    uncertainty: float | None = None
    production_iterations: float | None = None
    overlap_matrix: np.ndarray | None = None
    exchange_matrix: np.ndarray | None = None
    forward_reverse: dict[str, Any] | None = None
    failure: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this repeat produced a usable estimate.

        Returns:
            True if there is no recorded failure and an estimate is present.
        """
        return self.failure is None and self.estimate is not None


@dataclass
class EdgeResult:
    """The combined result for one edge.

    Attributes:
        name: Edge label.
        ligand_a: End state A ligand name.
        ligand_b: End state B ligand name.
        phases: Repeats keyed by phase.
        ddg: Relative binding free energy in kcal/mol, complex minus solvent.
        uncertainty: Uncertainty in kcal/mol.
        uncertainty_kind: ``std`` when repeats were averaged, ``mbar`` for a single repeat.
        failure: Why the edge has no result, if it has none.
    """

    name: str
    ligand_a: str
    ligand_b: str
    phases: dict[str, list[PhaseResult]] = field(default_factory=dict)
    ddg: float | None = None
    uncertainty: float | None = None
    uncertainty_kind: str | None = None
    failure: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this edge produced a usable relative free energy.

        Returns:
            True if both phases finished and a difference was computed.
        """
        return self.failure is None and self.ddg is not None


@dataclass
class LigandEstimate:
    """One ligand's fitted free energy.

    Attributes:
        name: Ligand name.
        dg: Free energy in kcal/mol, relative to the network's mean rather than absolute.
        uncertainty: Uncertainty from the fit, in kcal/mol.
        degree: Edges connecting this ligand to the network.
    """

    name: str
    dg: float
    uncertainty: float
    degree: int


@dataclass
class CycleClosure:
    """How far one cycle of the network fails to close.

    A cycle's free energies must sum to zero, so what they actually sum to measures the
    error the calculation carries. This is the only internal quality check RBFE has that
    needs no experimental data.

    Attributes:
        ligands: The ligands around the cycle.
        closure: Sum of the cycle's relative free energies, in kcal/mol.
        per_edge: Closure spread over the cycle's edges, in kcal/mol.
        normalized: Closure divided by its propagated uncertainty.
    """

    ligands: tuple[str, ...]
    closure: float
    per_edge: float
    normalized: float


@dataclass
class NetworkResults:
    """Everything gathered from a finished or partly finished network.

    Attributes:
        edges: One entry per planned edge, in name order.
        ligands: Fitted free energies, for ligands the network reaches.
        cycles: Cycle closures found in the network.
        unreachable: Ligands with no path to the largest connected component.
        warnings: Concerns the user should look at.
    """

    edges: list[EdgeResult] = field(default_factory=list)
    ligands: list[LigandEstimate] = field(default_factory=list)
    cycles: list[CycleClosure] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_edges(self) -> list[EdgeResult]:
        """Edges that produced a relative free energy.

        Returns:
            The usable edges, in name order.
        """
        return [edge for edge in self.edges if edge.ok]


def read_phase(path: Path, phase: str, repeat: int) -> PhaseResult:
    """Read one phase's result file.

    A file recording a failed simulation is returned with ``failure`` set rather than
    raising, so one broken repeat does not hide the others.

    Args:
        path: The result JSON.
        phase: Which phase this file belongs to.
        repeat: One-based repeat index.

    Returns:
        The phase result.

    Raises:
        OpenFEAPIError: If the file cannot be read at all.
    """
    result = load_result_json(path)
    outcome = PhaseResult(path=path, phase=phase, repeat=repeat)

    unit_results = result.get("unit_results", {})
    if unit_results and all("exception" in entry for entry in unit_results.values()):
        outcome.failure = "every protocol unit raised an exception"
        return outcome

    if result.get("estimate") is None or result.get("uncertainty") is None:
        outcome.failure = "no estimate was written; the simulation did not finish"
        return outcome

    outcome.estimate = to_kcal(result["estimate"])

    # Overwritten below by the analysis unit's MBAR error: this top-level value is the spread
    # across repeats, and one repeat per process makes it zero.
    outcome.uncertainty = to_kcal(result["uncertainty"])

    for entry in unit_results.values():
        outputs = entry.get("outputs", {})
        if "unit_estimate" not in outputs:
            continue
        overlap = outputs.get("unit_mbar_overlap") or {}
        exchange = outputs.get("replica_exchange_statistics") or {}
        mbar_error = to_kcal(outputs.get("unit_estimate_error"))
        if mbar_error is not None:
            outcome.uncertainty = mbar_error
        outcome.production_iterations = outputs.get("production_iterations")
        outcome.overlap_matrix = as_array(overlap.get("matrix"))
        outcome.exchange_matrix = as_array(exchange.get("matrix"))
        outcome.forward_reverse = outputs.get("forward_and_reverse_energies")
        break

    return outcome


def combine_repeats(results: list[PhaseResult]) -> tuple[float, float, str] | None:
    """Average a phase's repeats into one value.

    Args:
        results: Every repeat of one phase.

    Returns:
        The mean estimate, its uncertainty and which kind it is: ``std`` across repeats, or
        ``mbar`` for a single repeat. None if no repeat is usable.
    """
    usable = [entry for entry in results if entry.ok]
    if not usable:
        return None

    estimates = [entry.estimate for entry in usable if entry.estimate is not None]
    if len(estimates) == 1:
        single = usable[0]
        return estimates[0], single.uncertainty or 0.0, "mbar"
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1)), "std"


def read_rbfe_edge(campaign_dir: Path, name: str, ligand_a: str, ligand_b: str) -> EdgeResult:
    """Read every repeat of both phases of one RBFE edge and combine them.

    Args:
        campaign_dir: The campaign directory.
        name: The edge's name.
        ligand_a: End state A ligand name.
        ligand_b: End state B ligand name.

    Returns:
        The edge result, with ``failure`` set if either phase is missing.
    """
    edge = EdgeResult(name=name, ligand_a=ligand_a, ligand_b=ligand_b)
    combined: dict[str, tuple[float, float, str]] = {}

    for phase in PHASES:
        directory = campaign_dir / "runs" / name / phase
        results: list[PhaseResult] = []
        for repeat_dir in sorted(directory.glob("repeat*")):
            path = repeat_dir / "results.json"
            if not path.is_file():
                continue
            index = int(repeat_dir.name.removeprefix("repeat") or 0)
            results.append(read_phase(path, phase, index))
        edge.phases[phase] = results

        merged = combine_repeats(results)
        if merged is not None:
            combined[phase] = merged

    missing = [phase for phase in PHASES if phase not in combined]
    if missing:
        edge.failure = (
            f"no usable result for the {' and '.join(missing)} phase, so the relative "
            "binding free energy cannot be formed; both phases must finish"
        )
        return edge

    complex_dg, complex_error, complex_kind = combined["complex"]
    solvent_dg, solvent_error, solvent_kind = combined["solvent"]
    edge.ddg = complex_dg - solvent_dg
    edge.uncertainty = float(np.hypot(complex_error, solvent_error))
    edge.uncertainty_kind = "std" if "std" in (complex_kind, solvent_kind) else "mbar"
    return edge


def _fit(edges: list[EdgeResult]) -> tuple[list[LigandEstimate], list[CycleClosure]]:
    """Fit per-ligand free energies and measure every cycle's closure."""
    femap = FEMap()
    for edge in edges:
        if edge.ddg is None:
            continue
        # One source for every edge: cinnabar fits each source group separately, so per-edge
        # sources would fit each edge alone instead of the network.
        femap.add_relative_calculation(
            edge.ligand_a,
            edge.ligand_b,
            edge.ddg * KCAL,
            (edge.uncertainty or 0.0) * KCAL,
            source=FIT_SOURCE,
        )

    femap.generate_absolute_values()
    frame = femap.get_absolute_dataframe()
    degrees = {edge.ligand_a: 0 for edge in edges}
    for edge in edges:
        degrees[edge.ligand_a] = degrees.get(edge.ligand_a, 0) + 1
        degrees[edge.ligand_b] = degrees.get(edge.ligand_b, 0) + 1

    estimates = [
        LigandEstimate(
            name=str(row["label"]),
            dg=float(row["DG (kcal/mol)"]),
            uncertainty=float(row["uncertainty (kcal/mol)"]),
            degree=degrees.get(str(row["label"]), 0),
        )
        for _, row in frame.iterrows()
        if str(row["source"]).startswith("MLE")
    ]

    cycles: list[CycleClosure] = []
    closure_frame = femap.get_cycle_closure_dataframe()
    for _, row in closure_frame.iterrows():
        cycles.append(
            CycleClosure(
                ligands=tuple(str(name) for name in row["cycle"]),
                closure=float(row["cc (kcal/mol)"]),
                per_edge=float(row["cc_per_edge (kcal/mol)"]),
                normalized=float(row["cc_unc_normalized"]),
            )
        )

    return sorted(estimates, key=lambda entry: entry.dg), cycles


def gather_network(
    campaign_dir: Path,
    planned: dict[str, Any],
    read_edge: EdgeReader = read_rbfe_edge,
) -> NetworkResults:
    """Gather a network campaign's finished runs into edge and ligand free energies.

    Args:
        campaign_dir: The campaign directory.
        planned: The ``network.json`` summary written by planning.
        read_edge: Reader for one edge, which is what differs between the protocols.

    Returns:
        The gathered results, including what is still missing.
    """
    results = NetworkResults()
    ligands: set[str] = set()

    for entry in planned.get("edges", []):
        name = str(entry["name"])
        ligand_a, ligand_b = str(entry["ligand_a"]), str(entry["ligand_b"])
        ligands.update({ligand_a, ligand_b})
        results.edges.append(read_edge(campaign_dir, name, ligand_a, ligand_b))

    results.edges.sort(key=lambda edge: edge.name)
    usable = results.usable_edges

    for edge in results.edges:
        if edge.failure is not None:
            results.warnings.append(f"{edge.name}: {edge.failure}")

    if not usable:
        results.warnings.append("no edge has finished, so there is nothing to fit yet")
        results.unreachable = sorted(ligands)
        return results

    pairs = [(edge.ligand_a, edge.ligand_b) for edge in usable]
    groups = connected_components(ligands, pairs)
    largest = groups[-1] if groups else set()
    results.unreachable = unreachable_names(ligands, pairs)
    if results.unreachable:
        results.warnings.append(
            f"{len(results.unreachable)} ligand(s) are not connected to the rest of the "
            f"finished network, so they have no comparable free energy: "
            f"{', '.join(results.unreachable)}"
        )

    connected = [edge for edge in usable if edge.ligand_a in largest and edge.ligand_b in largest]
    try:
        results.ligands, results.cycles = _fit(connected)
    except (ValueError, FloatingPointError, KeyError) as error:
        warning = (
            f"the per-ligand fit could not be solved, so only the per-edge values are "
            f"available: {error}"
        )
        results.warnings.append(warning)
        logger.warning(warning)
        return results

    logger.info(
        "Gathered %d usable edge(s) over %d ligand(s), %d cycle(s)",
        len(connected),
        len(results.ligands),
        len(results.cycles),
    )
    return results


def write_edge_tsv(edges: list[EdgeResult], path: Path) -> None:
    """Write the per-edge relative free energies as a TSV file.

    Args:
        edges: The edges to write, in the order they should appear.
        path: File to write to. Parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "edge",
                "ligand_a",
                "ligand_b",
                "DDG (kcal/mol)",
                "uncertainty (kcal/mol)",
                "uncertainty_kind",
                "repeats",
                "note",
            ]
        )
        for edge in edges:
            repeats = min(
                (len([r for r in entries if r.ok]) for entries in edge.phases.values()),
                default=0,
            )
            writer.writerow(
                [
                    edge.name,
                    edge.ligand_a,
                    edge.ligand_b,
                    "" if edge.ddg is None else f"{edge.ddg:.3f}",
                    "" if edge.uncertainty is None else f"{edge.uncertainty:.3f}",
                    edge.uncertainty_kind or "",
                    repeats,
                    edge.failure or "",
                ]
            )


def write_ligand_tsv(ligands: list[LigandEstimate], path: Path) -> None:
    """Write the fitted per-ligand free energies as a TSV file.

    The values are relative to the network's mean, so differences between ligands are
    meaningful while a single value on its own is not.

    Args:
        ligands: The estimates to write.
        path: File to write to. Parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["ligand", "DG relative to network mean (kcal/mol)", "uncertainty (kcal/mol)", "edges"]
        )
        for ligand in ligands:
            writer.writerow(
                [ligand.name, f"{ligand.dg:.3f}", f"{ligand.uncertainty:.3f}", ligand.degree]
            )
