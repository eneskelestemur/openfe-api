"""Making the plots OpenFE already writes reachable from the analysis directory.

Every alchemical repeat leaves overlap, exchange and convergence plots inside a
``shared_<UnitName>-<hash>_attempt_0`` directory, where nothing will find them. They are
linked rather than copied, so a campaign does not carry two of each.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from openfe_api.campaign import ANALYSIS_DIR, RUNS_DIR, Campaign
from openfe_api.log import get_logger

__all__ = ["PLOTS_DIR", "PLOT_NAMES", "CollectedPlot", "collect_plots"]

logger = get_logger(__name__)

PLOT_NAMES = (
    "mbar_overlap_matrix.png",
    "replica_exchange_matrix.png",
    "replica_state_timeseries.png",
    "forward_reverse_convergence.png",
)
"""The plots an OpenFE analysis unit writes."""

PLOTS_DIR = "plots"

_REPEAT = re.compile(r"^repeat(\d+)$")
_PHASE = re.compile(r"(complex|solvent)", re.IGNORECASE)


@dataclass(frozen=True)
class CollectedPlot:
    """One of OpenFE's plots, and where it is reachable from.

    Attributes:
        source: The plot as OpenFE wrote it, inside the run's working directory.
        link: Where it is reachable from, inside the analysis directory.
        run: Which run produced it.
        repeat: Which repeat produced it.
        phase: ``complex`` or ``solvent``, empty when the run has no phases.
        kind: Which plot it is, without the extension.
    """

    source: Path
    link: Path
    run: str
    repeat: int
    phase: str
    kind: str


def _describe(relative: Path, shared: str) -> tuple[str, int, str] | None:
    """Read the run, repeat and phase out of a plot's path, or None if it has no repeat."""
    parts = relative.parts
    repeats = [index for index, part in enumerate(parts) if _REPEAT.match(part)]
    if not repeats:
        return None

    position = repeats[0]
    repeat = int(_REPEAT.match(parts[position]).group(1))  # type: ignore[union-attr]
    run = parts[0]
    # An RBFE edge runs its phases in separate directories; every other protocol carries
    # the phase in the name of the unit that wrote the plot.
    between = parts[1:position]
    found = _PHASE.search(between[0]) if between else _PHASE.search(shared)
    return run, repeat, found.group(1).lower() if found else ""


def collect_plots(campaign: Campaign) -> list[CollectedPlot]:
    """Link OpenFE's own plots into the campaign's analysis directory.

    Args:
        campaign: The campaign to search.

    Returns:
        One entry per plot linked, in path order. Empty for a campaign whose protocol
        writes none, such as plain MD.
    """
    runs = campaign.directory / RUNS_DIR
    target = campaign.directory / ANALYSIS_DIR / PLOTS_DIR
    if not runs.is_dir():
        return []

    collected: list[CollectedPlot] = []
    for source in sorted(runs.rglob("shared_*/*.png")):
        if source.name not in PLOT_NAMES:
            continue
        described = _describe(source.relative_to(runs), source.parent.name)
        if described is None:
            logger.debug("No repeat in the path of %s; skipping it", source)
            continue

        run, repeat, phase = described
        kind = source.stem
        parts = [run, f"repeat{repeat}", phase, kind]
        link = target / ("_".join(part for part in parts if part) + ".png")
        target.mkdir(parents=True, exist_ok=True)

        # Relative, so the campaign directory can be moved or mounted elsewhere, and
        # replaced every time because an analysis is meant to be re-runnable.
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(source, link.parent))
        collected.append(
            CollectedPlot(source=source, link=link, run=run, repeat=repeat, phase=phase, kind=kind)
        )

    logger.info("Linked %d plot(s) into %s", len(collected), target)
    return collected
