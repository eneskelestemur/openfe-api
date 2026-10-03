"""Reading one SepTop edge, whose two phases live in a single result file.

The top-level estimate is the relative binding free energy with both standard state
corrections applied, so it is taken as given; the per-phase analysis units supply the
uncertainty and the convergence data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openfe_api.log import get_logger
from openfe_api.protocols.rbfe import PHASES
from openfe_api.results.gather import as_array, load_result_json, to_kcal
from openfe_api.results.network import EdgeResult, PhaseResult

__all__ = ["read_septop_edge"]

logger = get_logger(__name__)


def _read_repeat(path: Path, repeat: int) -> tuple[float | None, list[PhaseResult], str | None]:
    """Read one repeat's file into its relative free energy and its two phases."""
    result = load_result_json(path)
    unit_results = result.get("unit_results", {})

    if unit_results and all("exception" in entry for entry in unit_results.values()):
        return None, [], "every protocol unit raised an exception"
    if result.get("estimate") is None:
        return None, [], "no estimate was written; the simulation did not finish"

    phases: list[PhaseResult] = []
    for entry in unit_results.values():
        outputs = entry.get("outputs", {})
        if "unit_estimate" not in outputs:
            continue
        overlap = outputs.get("unit_mbar_overlap") or {}
        exchange = outputs.get("replica_exchange_statistics") or {}
        phases.append(
            PhaseResult(
                path=path,
                phase=str(outputs.get("simtype", "unknown")),
                repeat=repeat,
                estimate=to_kcal(outputs.get("unit_estimate")),
                # The phase's MBAR error: the top-level value spreads across one repeat.
                uncertainty=to_kcal(outputs.get("unit_estimate_error")),
                production_iterations=outputs.get("production_iterations"),
                overlap_matrix=as_array(overlap.get("matrix")),
                exchange_matrix=as_array(exchange.get("matrix")),
                forward_reverse=outputs.get("forward_and_reverse_energies"),
            )
        )

    return to_kcal(result["estimate"]), phases, None


def read_septop_edge(campaign_dir: Path, name: str, ligand_a: str, ligand_b: str) -> EdgeResult:
    """Read every repeat of one SepTop edge and combine them.

    Args:
        campaign_dir: The campaign directory.
        name: The edge's name.
        ligand_a: End state A ligand name.
        ligand_b: End state B ligand name.

    Returns:
        The edge result, with ``failure`` set if no repeat produced an estimate.
    """
    edge = EdgeResult(name=name, ligand_a=ligand_a, ligand_b=ligand_b)
    edge.phases = {phase: [] for phase in PHASES}

    estimates: list[float] = []
    errors: list[float] = []
    failures: list[str] = []

    for repeat_dir in sorted((campaign_dir / "runs" / name).glob("repeat*")):
        path = repeat_dir / "results.json"
        if not path.is_file():
            continue
        repeat = int(repeat_dir.name.removeprefix("repeat") or 0)
        estimate, phases, failure = _read_repeat(path, repeat)
        for phase in phases:
            edge.phases.setdefault(phase.phase, []).append(phase)
        if failure is not None or estimate is None:
            failures.append(f"repeat {repeat}: {failure or 'no estimate'}")
            continue
        estimates.append(estimate)
        squared = [(phase.uncertainty or 0.0) ** 2 for phase in phases if phase.phase in PHASES]
        errors.append(float(np.sqrt(sum(squared))))

    if not estimates:
        edge.failure = "no repeat produced a usable relative binding free energy" + (
            f" ({'; '.join(failures)})" if failures else ""
        )
        return edge

    if len(estimates) == 1:
        edge.ddg = estimates[0]
        edge.uncertainty = errors[0]
        edge.uncertainty_kind = "mbar"
    else:
        edge.ddg = float(np.mean(estimates))
        edge.uncertainty = float(np.std(estimates, ddof=1))
        edge.uncertainty_kind = "std"

    if failures:
        logger.warning("Edge '%s' has unusable repeat(s): %s", name, "; ".join(failures))
    return edge
