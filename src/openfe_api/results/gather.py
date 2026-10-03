"""Reading ABFE result files into per-ligand binding free energies.

Everything needed is inside the result JSON that ``openfe quickrun`` writes: the overall
cycle estimate, the per-leg MBAR estimates, and the convergence data. The trajectory files
are not touched, so gathering is fast and works on a login node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gufe.tokenization import JSON_HANDLER
from openff.units import unit

from openfe_api.exceptions import OpenFEAPIError
from openfe_api.log import get_logger

__all__ = [
    "KCAL",
    "LegResult",
    "LigandResult",
    "RepeatResult",
    "as_array",
    "gather_results",
    "load_result_json",
    "read_repeat",
    "result_is_successful",
    "to_kcal",
    "write_tsv",
]

logger = get_logger(__name__)

TRAJECTORY_OUTPUT = "nc"
"""Key the simulation units report their production trajectory under."""

KCAL = unit.kilocalorie_per_mole


def load_result_json(path: Path) -> dict[str, Any]:
    """Load a result file written by ``openfe quickrun``.

    Args:
        path: The result JSON.

    Returns:
        The deserialized result, with quantities restored.

    Raises:
        OpenFEAPIError: If the file cannot be read or is not valid JSON.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, cls=JSON_HANDLER.decoder)
    except (OSError, json.JSONDecodeError) as error:
        raise OpenFEAPIError(f"{path}: could not be read as a result file: {error}") from error


def result_is_successful(path: Path, expects_estimate: bool = True) -> bool:
    """Whether a result file records a run that finished.

    ``openfe quickrun`` writes a result file even when the run failed, recording a null
    estimate and the unit that raised, so the file existing does not mean the repeat is
    done.

    Args:
        path: The result JSON.
        expects_estimate: Whether this protocol produces a free energy. Plain MD does not:
            its ``get_estimate`` returns None by design, so a null estimate there means the
            protocol has nothing to report rather than that the run failed.

    Returns:
        True if the file records a finished run.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    units = data.get("unit_results", {})
    if expects_estimate:
        if data.get("estimate") is None:
            return False
        return not (units and all("exception" in entry for entry in units.values()))

    # Plain MD reports no estimate, so a trajectory is what marks it finished; a unit that
    # raised has no outputs.
    return any(TRAJECTORY_OUTPUT in entry.get("outputs", {}) for entry in units.values())


def to_kcal(value: Any) -> float | None:
    """Convert a quantity to kcal/mol.

    Args:
        value: A unit-aware quantity, or None.

    Returns:
        The value in kcal/mol, or None if it was None or not a quantity.
    """
    if value is None:
        return None
    try:
        return float(value.to(KCAL).m)
    except AttributeError:
        return None


@dataclass
class LegResult:
    """One leg of the thermodynamic cycle, for one repeat.

    Attributes:
        simtype: ``complex`` or ``solvent``.
        estimate: MBAR estimate for the leg, in kcal/mol.
        error: MBAR uncertainty for the leg, in kcal/mol.
        standard_state_correction: Restraint correction applied, in kcal/mol.
        production_iterations: Iterations of production sampling analyzed.
        overlap_matrix: MBAR overlap matrix, if reported.
        exchange_matrix: Replica exchange probability matrix, if reported.
        forward_reverse: Forward and reverse convergence data, if reported.
    """

    simtype: str
    estimate: float | None
    error: float | None
    standard_state_correction: float | None = None
    production_iterations: float | None = None
    overlap_matrix: np.ndarray | None = None
    exchange_matrix: np.ndarray | None = None
    forward_reverse: dict[str, Any] | None = None


@dataclass
class RepeatResult:
    """One repeat of one ligand's ABFE calculation.

    Attributes:
        path: The result file this came from.
        ligand: Ligand name taken from the result.
        estimate: Overall binding free energy, in kcal/mol.
        uncertainty: Overall MBAR uncertainty, in kcal/mol.
        legs: Legs keyed by simulation type.
        failure: Why the repeat is unusable, if it is.
    """

    path: Path
    ligand: str | None = None
    estimate: float | None = None
    uncertainty: float | None = None
    legs: dict[str, LegResult] = field(default_factory=dict)
    failure: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this repeat produced a usable estimate.

        Returns:
            True if there is no recorded failure and an estimate is present.
        """
        return self.failure is None and self.estimate is not None


@dataclass
class LigandResult:
    """The combined result for one ligand.

    Attributes:
        name: Ligand name.
        dg: Mean binding free energy across repeats, in kcal/mol.
        uncertainty: Uncertainty in kcal/mol.
        uncertainty_kind: ``std`` across repeats, or ``mbar`` for a single repeat.
        repeats: The repeats that contributed.
        failed_repeats: Repeats that could not be used.
    """

    name: str
    dg: float
    uncertainty: float
    uncertainty_kind: str
    repeats: list[RepeatResult] = field(default_factory=list)
    failed_repeats: list[RepeatResult] = field(default_factory=list)


def _ligand_name(result: dict[str, Any]) -> str | None:
    """Extract the alchemical ligand's name from a result."""
    try:
        solvent_data = next(iter(result["protocol_result"]["data"]["solvent"].values()))[0]
    except (KeyError, IndexError, TypeError):
        return None

    inputs = solvent_data.get("inputs", {})
    for holder in (inputs.get("setup_results", {}).get("inputs", {}), inputs):
        try:
            return str(holder["alchemical_components"]["stateA"][0]["molprops"]["ofe-name"])
        except (KeyError, IndexError, TypeError):
            continue
    return None


def read_repeat(path: Path) -> RepeatResult:
    """Read one result file into a repeat result.

    A file that records a failed simulation is returned with ``failure`` set rather than
    raising, so one broken repeat does not hide the others.

    Args:
        path: The result JSON.

    Returns:
        The repeat result.

    Raises:
        OpenFEAPIError: If the file cannot be read at all.
    """
    result = load_result_json(path)
    repeat = RepeatResult(path=path, ligand=_ligand_name(result))

    unit_results = result.get("unit_results", {})
    if unit_results and all("exception" in entry for entry in unit_results.values()):
        repeat.failure = "every protocol unit raised an exception"
        return repeat

    if result.get("estimate") is None or result.get("uncertainty") is None:
        repeat.failure = "no estimate was written; the simulation did not finish"
        return repeat

    repeat.estimate = to_kcal(result["estimate"])
    repeat.uncertainty = to_kcal(result["uncertainty"])

    for entry in unit_results.values():
        source = str(entry.get("source_key", ""))
        if "Setup" in source or "Simulation" in source:
            continue
        outputs = entry.get("outputs", {})
        if "unit_estimate" not in outputs:
            continue

        overlap = outputs.get("unit_mbar_overlap") or {}
        exchange = outputs.get("replica_exchange_statistics") or {}
        repeat.legs[str(outputs.get("simtype", "unknown"))] = LegResult(
            simtype=str(outputs.get("simtype", "unknown")),
            estimate=to_kcal(outputs.get("unit_estimate")),
            error=to_kcal(outputs.get("unit_estimate_error")),
            standard_state_correction=to_kcal(outputs.get("standard_state_correction")),
            production_iterations=outputs.get("production_iterations"),
            overlap_matrix=as_array(overlap.get("matrix")),
            exchange_matrix=as_array(exchange.get("matrix")),
            forward_reverse=outputs.get("forward_and_reverse_energies"),
        )

    return repeat


def as_array(value: Any) -> np.ndarray | None:
    """Convert a serialized matrix to a numpy array.

    Args:
        value: The value from the result file.

    Returns:
        The array, or None if the value was missing or not array-like.
    """
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _combine(name: str, repeats: list[RepeatResult]) -> LigandResult:
    """Combine a ligand's repeats into one result.

    With more than one repeat the spread between repeats is the uncertainty, matching
    ``openfe gather-abfe``. With a single repeat the MBAR errors of the two legs are
    combined instead, since there is no spread to measure.
    """
    estimates = [repeat.estimate for repeat in repeats if repeat.estimate is not None]
    mean = float(np.mean(estimates))

    if len(estimates) > 1:
        spread = float(np.std(estimates, ddof=1))
        return LigandResult(
            name=name,
            dg=mean,
            uncertainty=0.0 if np.isnan(spread) else spread,
            uncertainty_kind="std",
            repeats=repeats,
        )

    errors = [
        leg.error for repeat in repeats for leg in repeat.legs.values() if leg.error is not None
    ]
    combined = float(np.sqrt(np.sum(np.square(errors)))) if errors else 0.0
    return LigandResult(
        name=name,
        dg=mean,
        uncertainty=combined,
        uncertainty_kind="mbar",
        repeats=repeats,
    )


def gather_results(paths: list[Path]) -> tuple[list[LigandResult], list[RepeatResult]]:
    """Read result files and combine them per ligand.

    Args:
        paths: Result JSON files.

    Returns:
        A tuple of the combined per-ligand results, sorted by name, and the repeats that
        could not be used.
    """
    by_ligand: dict[str, list[RepeatResult]] = {}
    failed: list[RepeatResult] = []

    for path in paths:
        repeat = read_repeat(path)
        if not repeat.ok:
            failed.append(repeat)
            logger.warning("Skipping %s: %s", path, repeat.failure)
            continue
        name = repeat.ligand or path.parent.parent.name
        by_ligand.setdefault(name, []).append(repeat)

    results = [_combine(name, repeats) for name, repeats in sorted(by_ligand.items())]
    for result in results:
        result.failed_repeats = [
            repeat for repeat in failed if (repeat.ligand or "") == result.name
        ]
    return results, failed


def write_tsv(results: list[LigandResult], path: Path) -> None:
    """Write per-ligand results as a tab-separated file.

    Args:
        results: The combined results.
        path: File to write. Parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ligand\tDG(kcal/mol)\tuncertainty(kcal/mol)\tuncertainty_kind\trepeats"]
    for result in results:
        lines.append(
            f"{result.name}\t{result.dg:.3f}\t{result.uncertainty:.3f}\t"
            f"{result.uncertainty_kind}\t{len(result.repeats)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d result(s) to %s", len(results), path)
