"""Convergence checks on ABFE results.

The checks follow OpenFE's own guidance: MBAR overlap should be at least 0.03 between
neighboring lambda states, the replica exchange matrix should connect neighboring states,
and forward and reverse estimates should agree within error. A check that cannot be made,
because the data is absent, is reported as unknown rather than as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openfe_api.results.gather import LigandResult, RepeatResult
from openfe_api.results.network import EdgeResult, NetworkResults

__all__ = [
    "MAX_CYCLE_CLOSURE",
    "MIN_NEIGHBOR_OVERLAP",
    "Check",
    "QualityReport",
    "check_edge",
    "check_ligand",
    "check_network",
]

MIN_NEIGHBOR_OVERLAP = 0.03
"""Smallest acceptable MBAR overlap between neighboring lambda states.

OpenFE's protocol documentation asks for matrix elements adjacent to the diagonal to be at
least this value, so that the end states stay energetically connected.
"""

MAX_CYCLE_CLOSURE = 1.0
"""Cycle closure in kcal/mol beyond which a cycle is reported as not closing."""

REPEAT_SPREAD_WARNING = 1.0
"""Spread between repeats, in kcal/mol, above which the estimate is called unreliable."""


@dataclass(frozen=True)
class Check:
    """The outcome of one quality check.

    Attributes:
        name: Short name of the check.
        status: ``pass``, ``fail`` or ``unknown``.
        detail: What was measured, in words.
    """

    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        """Whether the check failed.

        Returns:
            True if the status is ``fail``.
        """
        return self.status == "fail"


@dataclass
class QualityReport:
    """Quality checks for one ligand.

    Attributes:
        ligand: Ligand name.
        checks: The checks that were run.
    """

    ligand: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        """Checks that failed.

        Returns:
            The failing checks.
        """
        return [check for check in self.checks if check.failed]

    @property
    def verdict(self) -> str:
        """One-word summary of the report.

        Returns:
            ``fail`` if any check failed, ``unknown`` if any could not be made, else ``pass``.
        """
        if self.failures:
            return "fail"
        if any(check.status == "unknown" for check in self.checks):
            return "unknown"
        return "pass"


def _neighbor_values(matrix: np.ndarray) -> np.ndarray:
    """Return the elements immediately above and below the diagonal."""
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.empty(0)
    upper = np.diagonal(matrix, offset=1)
    lower = np.diagonal(matrix, offset=-1)
    return np.concatenate([upper, lower])


def _overlap_verdict(worst: tuple[float, str] | None) -> Check:
    """Turn the worst neighboring overlap found anywhere into a check."""
    if worst is None:
        return Check("mbar_overlap", "unknown", "no overlap matrix was reported")

    smallest, where = worst
    return Check(
        "mbar_overlap",
        "pass" if smallest >= MIN_NEIGHBOR_OVERLAP else "fail",
        f"smallest neighboring overlap {smallest:.3f} (want >= {MIN_NEIGHBOR_OVERLAP}) "
        f"in the {where}",
    )


def _overlap_check(repeats: list[RepeatResult]) -> Check:
    """Check MBAR overlap between neighboring lambda states."""
    worst: tuple[float, str] | None = None
    for repeat in repeats:
        for leg in repeat.legs.values():
            if leg.overlap_matrix is None:
                continue
            values = _neighbor_values(leg.overlap_matrix)
            if values.size == 0:
                continue
            smallest = float(values.min())
            if worst is None or smallest < worst[0]:
                worst = (smallest, f"{leg.simtype} leg of repeat {repeat.path.parent.name}")

    return _overlap_verdict(worst)


def _exchange_check(repeats: list[RepeatResult]) -> Check:
    """Check that replica exchange connects neighboring lambda states."""
    worst: tuple[float, str] | None = None
    for repeat in repeats:
        for leg in repeat.legs.values():
            if leg.exchange_matrix is None:
                continue
            values = _neighbor_values(leg.exchange_matrix)
            if values.size == 0:
                continue
            smallest = float(values.min())
            if worst is None or smallest < worst[0]:
                worst = (smallest, f"{leg.simtype} leg")

    if worst is None:
        return Check("replica_exchange", "unknown", "no exchange matrix was reported")

    smallest, where = worst
    status = "pass" if smallest > 0.0 else "fail"
    detail = (
        f"smallest neighboring exchange probability {smallest:.3f} in the {where}"
        if status == "pass"
        else f"neighboring lambda states never exchanged in the {where}, so the "
        "end states are not connected"
    )
    return Check("replica_exchange", status, detail)


def _forward_reverse_check(repeats: list[RepeatResult]) -> Check:
    """Check that forward and reverse estimates agree within their errors."""
    worst: tuple[float, float, str] | None = None
    for repeat in repeats:
        for leg in repeat.legs.values():
            measured = _late_forward_reverse_gap(leg.forward_reverse)
            if measured is None:
                continue
            sigmas, gap = measured
            if worst is None or sigmas > worst[0]:
                worst = (sigmas, gap, f"{leg.simtype} leg")

    if worst is None:
        return Check("forward_reverse", "unknown", "no convergence data was reported")

    sigmas, gap, where = worst
    status = "pass" if sigmas <= 1.0 else "fail"
    return Check(
        "forward_reverse",
        status,
        f"over the second half of the data, forward and reverse estimates differ by up to "
        f"{gap:.2f} kcal/mol ({sigmas:.1f}x their combined error, want <= 1) in the {where}",
    )


def _late_forward_reverse_gap(
    data: dict[str, Any] | None,
) -> tuple[float, float] | None:
    """Measure how far apart the forward and reverse estimates stay late in the data.

    Only the second half of the data is judged. Early fractions disagree even for a
    converged run, since each direction has seen little data, and the final point is
    identical by construction because both directions then use everything, so neither
    says anything about convergence.
    """
    if not data:
        return None
    try:
        fractions = np.asarray(data["fractions"], dtype=float)
        forward = np.asarray([value.m for value in data["forward_DGs"]], dtype=float)
        reverse = np.asarray([value.m for value in data["reverse_DGs"]], dtype=float)
        forward_error = np.asarray([value.m for value in data["forward_dDGs"]], dtype=float)
        reverse_error = np.asarray([value.m for value in data["reverse_dDGs"]], dtype=float)
    except (KeyError, TypeError, AttributeError, ValueError):
        return None

    if fractions.size < 2:
        return None

    late = (fractions >= 0.5) & (np.arange(fractions.size) < fractions.size - 1)
    if not late.any():
        return None

    gaps = np.abs(forward[late] - reverse[late])
    tolerance = forward_error[late] + reverse_error[late]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigmas = gaps / np.where(tolerance > 0, tolerance, np.nan)
    if np.all(np.isnan(sigmas)):
        return None

    index = int(np.nanargmax(sigmas))
    return float(sigmas[index]), float(gaps[index])


def _repeat_check(result: LigandResult) -> Check:
    """Check the spread between repeats."""
    if len(result.repeats) < 2:
        return Check(
            "repeat_spread",
            "unknown",
            "only one repeat was run, so the spread between repeats is unknown",
        )

    status = "pass" if result.uncertainty <= REPEAT_SPREAD_WARNING else "fail"
    return Check(
        "repeat_spread",
        status,
        f"{len(result.repeats)} repeats spread by {result.uncertainty:.2f} kcal/mol "
        f"(want <= {REPEAT_SPREAD_WARNING})",
    )


def check_ligand(result: LigandResult) -> QualityReport:
    """Run every quality check for one ligand.

    Args:
        result: The ligand's combined result.

    Returns:
        The quality report.
    """
    report = QualityReport(ligand=result.name)
    report.checks = [
        _overlap_check(result.repeats),
        _exchange_check(result.repeats),
        _forward_reverse_check(result.repeats),
        _repeat_check(result),
    ]
    if result.failed_repeats:
        report.checks.append(
            Check(
                "failed_repeats",
                "fail",
                f"{len(result.failed_repeats)} repeat(s) did not produce a result",
            )
        )
    return report


def _phase_overlap_check(edge: EdgeResult) -> Check:
    """Check MBAR overlap across both phases of one edge."""
    worst: tuple[float, str] | None = None
    for phase, results in edge.phases.items():
        for result in results:
            if not result.ok or result.overlap_matrix is None:
                continue
            values = _neighbor_values(result.overlap_matrix)
            if values.size == 0:
                continue
            smallest = float(values.min())
            if worst is None or smallest < worst[0]:
                worst = (smallest, f"{phase} phase of repeat {result.repeat}")

    return _overlap_verdict(worst)


def _phase_spread_check(edge: EdgeResult) -> Check:
    """Check how far an edge's repeats disagree within each phase."""
    worst: tuple[float, str] | None = None
    for phase, results in edge.phases.items():
        estimates = [result.estimate for result in results if result.ok]
        values = [value for value in estimates if value is not None]
        if len(values) < 2:
            continue
        spread = float(np.max(values) - np.min(values))
        if worst is None or spread > worst[0]:
            worst = (spread, phase)

    if worst is None:
        return Check("repeat_spread", "unknown", "fewer than two usable repeats in any phase")

    spread, phase = worst
    status = "pass" if spread <= REPEAT_SPREAD_WARNING else "fail"
    return Check(
        "repeat_spread",
        status,
        f"repeats of the {phase} phase span {spread:.2f} kcal/mol "
        f"(want <= {REPEAT_SPREAD_WARNING})",
    )


def _both_phases_check(edge: EdgeResult) -> Check:
    """Check that both phases produced a result, since their difference is the answer."""
    missing = [phase for phase, results in edge.phases.items() if not any(r.ok for r in results)]
    if missing:
        return Check(
            "both_phases",
            "fail",
            f"no usable result for the {' and '.join(sorted(missing))} phase",
        )
    return Check("both_phases", "pass", "both phases produced an estimate")


def check_edge(edge: EdgeResult) -> QualityReport:
    """Run the quality checks for one RBFE edge.

    Args:
        edge: The gathered edge.

    Returns:
        The report, named after the edge.
    """
    return QualityReport(
        ligand=edge.name,
        checks=[_both_phases_check(edge), _phase_overlap_check(edge), _phase_spread_check(edge)],
    )


def check_network(results: NetworkResults) -> QualityReport:
    """Run the network-level checks: cycle closure and connectivity.

    Cycle closure is the only internal quality check RBFE has that needs no experimental
    data: a cycle's free energies must sum to zero, so what they do sum to is the error the
    network carries.

    Args:
        results: The gathered network.

    Returns:
        The report, named ``network``.
    """
    checks: list[Check] = []

    if not results.cycles:
        checks.append(
            Check(
                "cycle_closure",
                "unknown",
                "the network has no cycles, so there is nothing to close; plan a redundant "
                "network to get this check",
            )
        )
    else:
        worst = max(results.cycles, key=lambda cycle: abs(cycle.closure))
        status = "pass" if abs(worst.closure) <= MAX_CYCLE_CLOSURE else "fail"
        checks.append(
            Check(
                "cycle_closure",
                status,
                f"worst cycle closes to {worst.closure:+.2f} kcal/mol over "
                f"{len(worst.ligands)} ligands ({' to '.join(worst.ligands)}), want "
                f"|closure| <= {MAX_CYCLE_CLOSURE}",
            )
        )

    if results.unreachable:
        checks.append(
            Check(
                "connectivity",
                "fail",
                f"{len(results.unreachable)} ligand(s) have no path to the rest of the "
                f"network: {', '.join(results.unreachable)}",
            )
        )
    else:
        checks.append(Check("connectivity", "pass", "every ligand is connected to the network"))

    return QualityReport(ligand="network", checks=checks)
