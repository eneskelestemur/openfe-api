"""The figures an analysis writes.

Drawing goes through matplotlib's object interface rather than pyplot, whose global state
two analyses in background threads would share.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from openfe_api.analysis.correlation import Correlation, Prediction
from openfe_api.analysis.interactions import NOISY_INTERACTION, ResidueContact
from openfe_api.analysis.style import SEQUENTIAL, Theme, style_axes
from openfe_api.analysis.trajectory import RunStability
from openfe_api.log import get_logger

__all__ = [
    "MIN_OCCUPANCY",
    "TOLERANCE",
    "correlation_figure",
    "interaction_figure",
    "ranking_figure",
    "stability_figure",
]

logger = get_logger(__name__)

TOLERANCE = 1.0
"""Half-width of the band drawn around agreement, in kcal/mol."""

MIN_OCCUPANCY = 0.1
"""Occupancy a contact must reach in some run before it earns a row in the heatmap."""

MARKER_SIZE = 46
MARKER_RING = 1.2
RAIL_WIDTH = 0.3
LABEL_LIMIT = 12


def _save(figure: Figure, path: Path) -> Path:
    """Render a figure to disk and return where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    # The figure's own color, not the matplotlib default, or a dark figure is saved with a
    # white border around it.
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    logger.info("Wrote %s", path)
    return path


def _metrics_text(outcome: Correlation) -> str:
    """Summarize the metrics for the corner of a correlation figure."""
    lines = [f"N = {len(outcome.paired)}"]
    for label, value in (
        ("R", outcome.pearson),
        ("ρ", outcome.spearman),
        ("τ", outcome.kendall),
    ):
        if value is not None:
            lines.append(f"{label} = {value:.2f}")
    for label, value in (("MUE", outcome.mue), ("RMSE", outcome.rmse)):
        if value is not None:
            lines.append(f"{label} = {value:.2f} kcal/mol")
    return "\n".join(lines)


def correlation_figure(outcome: Correlation, title: str, theme: Theme, path: Path) -> Path:
    """Draw predicted against measured free energies.

    Ligands with no measurement are drawn at their predicted value in a panel on the right.

    Args:
        outcome: The paired points, the unpaired predictions and the metrics.
        title: Figure title, usually the campaign name.
        theme: The colors to draw with.
        path: Where to write the figure.

    Returns:
        The path written.

    Raises:
        ValueError: If there is nothing at all to draw.
    """
    if not outcome.paired and not outcome.predicted_only:
        raise ValueError("no ligands to draw")

    figure = Figure(figsize=(6.4, 4.8), layout="constrained")
    figure.set_facecolor(theme.surface)
    has_rail = bool(outcome.predicted_only)
    axes_grid = figure.add_gridspec(1, 2, width_ratios=[1, RAIL_WIDTH] if has_rail else [1, 0.001])
    axes = figure.add_subplot(axes_grid[0, 0])
    style_axes(axes, theme)

    measured = np.array([point.measured for point in outcome.paired], dtype=float)
    predicted = np.array([point.predicted for point in outcome.paired], dtype=float)
    rail_values = np.array([entry.dg for entry in outcome.predicted_only], dtype=float)

    spread = (
        np.concatenate([measured, predicted, rail_values])
        if has_rail
        else np.concatenate([measured, predicted])
    )
    low, high = float(spread.min()), float(spread.max())
    margin = max(0.6, 0.12 * (high - low))
    limits = (low - margin, high + margin)

    if outcome.paired:
        edge = np.array(limits)
        axes.fill_between(
            edge, edge - TOLERANCE, edge + TOLERANCE, color=theme.band, zorder=1, linewidth=0
        )
        axes.plot(edge, edge, color=theme.axis, linewidth=1.4, zorder=2)
        axes.errorbar(
            measured,
            predicted,
            yerr=[point.predicted_uncertainty for point in outcome.paired],
            xerr=[point.measured_uncertainty or 0.0 for point in outcome.paired],
            fmt="none",
            ecolor=theme.measured,
            elinewidth=1.1,
            alpha=0.55,
            zorder=3,
        )
        axes.scatter(
            measured,
            predicted,
            s=MARKER_SIZE,
            color=theme.measured,
            edgecolor=theme.surface,
            linewidth=MARKER_RING,
            zorder=4,
            label="measured",
        )

    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("Measured ΔG (kcal/mol)")
    axes.set_ylabel("Predicted ΔG (kcal/mol)")
    subtitle = "predictions shifted onto the measured mean" if outcome.aligned else ""
    axes.set_title(title, color=theme.primary_ink, loc="left", pad=20 if subtitle else 10)
    if subtitle:
        axes.text(
            0.0,
            1.015,
            subtitle,
            transform=axes.transAxes,
            fontsize=7.5,
            color=theme.muted_ink,
            va="bottom",
            ha="left",
        )

    if outcome.paired:
        axes.text(
            0.04,
            0.96,
            _metrics_text(outcome),
            transform=axes.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=theme.secondary_ink,
            linespacing=1.5,
        )
        axes.text(
            0.04,
            0.04,
            f"shaded: within {TOLERANCE:.0f} kcal/mol",
            transform=axes.transAxes,
            va="bottom",
            ha="left",
            fontsize=7.5,
            color=theme.muted_ink,
        )

    if has_rail:
        _draw_rail(figure, axes_grid, outcome.predicted_only, limits, theme)

    # A legend only once there are two kinds of mark to tell apart; with one the title says it.
    if outcome.paired and has_rail:
        axes.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=6,
                    markerfacecolor=theme.measured,
                    markeredgecolor=theme.surface,
                    label="measured",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=theme.predicted_only,
                    label="no measurement",
                ),
            ],
            loc="lower right",
            fontsize=7.5,
            labelcolor=theme.secondary_ink,
            handletextpad=0.5,
            borderpad=0.8,
            # Backed in the surface color, or the default light box hides the labels on a
            # dark figure and the band shows through on a light one.
            frameon=True,
            facecolor=theme.surface,
            edgecolor=theme.axis,
            framealpha=1.0,
        )

    return _save(figure, path)


def _jitter(values: np.ndarray, span: float) -> np.ndarray:
    """Offset marks sideways so neighbors in the rail stop sitting on each other.

    Deterministic, so the same campaign always draws the same figure.
    """
    close = 0.06 * span
    order = np.argsort(values)
    offsets = np.zeros(values.size)
    group: list[int] = []
    for index in order:
        if group and values[index] - values[group[-1]] > close:
            _spread_group(group, offsets)
            group = []
        group.append(int(index))
    _spread_group(group, offsets)
    return offsets


def _spread_group(group: list[int], offsets: np.ndarray) -> None:
    """Spread one cluster of marks evenly across the rail."""
    if len(group) < 2:
        return
    width = min(0.30, 0.09 * len(group))
    steps = np.linspace(-width, width, len(group))
    for position, index in zip(steps, group, strict=True):
        offsets[index] = position


def _spread_labels(values: np.ndarray, span: float) -> list[float]:
    """Nudge label positions apart so neighboring names stay readable."""
    minimum = 0.035 * span
    order = np.argsort(values)
    placed = values.astype(float).copy()
    for index in range(1, order.size):
        previous, current = order[index - 1], order[index]
        if placed[current] - placed[previous] < minimum:
            placed[current] = placed[previous] + minimum
    return [float(placed[index]) for index in range(placed.size)]


def _draw_rail(
    figure: Figure,
    axes_grid: GridSpec,
    entries: list[Prediction],
    limits: tuple[float, float],
    theme: Theme,
) -> None:
    """Draw the ligands that have a prediction but nothing to compare it against."""
    rail = figure.add_subplot(axes_grid[0, 1])
    style_axes(rail, theme)
    rail.grid(False)
    rail.set_facecolor(theme.band)
    rail.set_ylim(*limits)
    rail.set_xlim(-0.5, 0.5)
    rail.set_xticks([])
    rail.spines["left"].set_visible(False)
    rail.spines["bottom"].set_visible(False)

    values = np.array([entry.dg for entry in entries], dtype=float)
    offsets = _jitter(values, limits[1] - limits[0])
    rail.errorbar(
        offsets,
        values,
        yerr=[entry.uncertainty for entry in entries],
        fmt="none",
        ecolor=theme.predicted_only,
        elinewidth=1.1,
        alpha=0.55,
    )
    rail.scatter(
        offsets,
        values,
        s=MARKER_SIZE,
        facecolor="none",
        edgecolor=theme.predicted_only,
        linewidth=1.4,
        zorder=4,
    )
    rail.set_title("no measurement", color=theme.muted_ink, fontsize=7.5, pad=8)

    # Names ride on the right-hand ticks: matplotlib then reserves room for them, where
    # free-floating text would overflow the panel. Dropped once they stop being legible.
    if len(entries) <= LABEL_LIMIT:
        rail.yaxis.tick_right()
        rail.set_yticks(_spread_labels(values, limits[1] - limits[0]))
        rail.set_yticklabels([entry.name for entry in entries], fontsize=7)
        rail.tick_params(axis="y", length=0, colors=theme.secondary_ink)
    else:
        rail.set_yticks([])


def ranking_figure(predictions: list[Prediction], title: str, theme: Theme, path: Path) -> Path:
    """Draw ligands ordered by predicted free energy, with their uncertainties.

    This is what a campaign with no measured values has to show: an ordering, and how firmly
    each place in it is held.

    Args:
        predictions: The calculated free energies.
        title: Figure title, usually the campaign name.
        theme: The colors to draw with.
        path: Where to write the figure.

    Returns:
        The path written.

    Raises:
        ValueError: If there are no predictions to draw.
    """
    if not predictions:
        raise ValueError("no predictions to draw")

    ordered = sorted(predictions, key=lambda entry: entry.dg)
    height = max(2.4, 0.26 * len(ordered) + 1.2)
    figure = Figure(figsize=(6.0, height), layout="constrained")
    figure.set_facecolor(theme.surface)
    axes = figure.add_subplot(111)
    style_axes(axes, theme)
    axes.grid(axis="y", visible=False)

    positions = np.arange(len(ordered), dtype=float)
    values = np.array([entry.dg for entry in ordered], dtype=float)
    axes.errorbar(
        values,
        positions,
        xerr=[entry.uncertainty for entry in ordered],
        fmt="none",
        ecolor=theme.measured,
        elinewidth=1.1,
        alpha=0.55,
    )
    axes.scatter(
        values,
        positions,
        s=MARKER_SIZE,
        color=theme.measured,
        edgecolor=theme.surface,
        linewidth=MARKER_RING,
        zorder=4,
    )

    axes.set_yticks(positions)
    axes.set_yticklabels([entry.name for entry in ordered], fontsize=8)
    axes.set_ylim(-0.8, len(ordered) - 0.2)
    axes.invert_yaxis()
    axes.set_xlabel("Predicted ΔG (kcal/mol)")
    axes.set_title(title, color=theme.primary_ink, loc="left", pad=10)

    return _save(figure, path)


def _run_labels(runs: list[str], limit: int = 26) -> list[str]:
    """Shorten run names to what tells them apart.

    Edge names share a prefix long enough to crowd the plot off the figure.
    """
    shared = 0
    if len(runs) > 1:
        for letters in zip(*runs, strict=False):
            if len(set(letters)) > 1:
                break
            shared += 1
    labels = [run[shared:] or run for run in runs]
    return [label if len(label) <= limit else f"{label[: limit - 1]}\u2026" for label in labels]


def stability_figure(summaries: list[RunStability], title: str, theme: Theme, path: Path) -> Path:
    """Draw how far each run's ligand moved, ordered worst first.

    Args:
        summaries: One aggregate per run.
        title: Figure title, usually the campaign name.
        theme: The colors to draw with.
        path: Where to write the figure.

    Returns:
        The path written.

    Raises:
        ValueError: If no run has a drift to draw.
    """
    measured = [entry for entry in summaries if entry.max_drift is not None]
    if not measured:
        raise ValueError("no run has a measured drift")

    ordered = sorted(measured, key=lambda entry: entry.max_drift or 0.0, reverse=True)
    height = max(2.4, 0.26 * len(ordered) + 1.2)
    figure = Figure(figsize=(6.0, height), layout="constrained")
    figure.set_facecolor(theme.surface)
    axes = figure.add_subplot(111)
    style_axes(axes, theme)
    axes.grid(axis="y", visible=False)

    positions = np.arange(len(ordered), dtype=float)
    means = np.array([entry.mean_drift or 0.0 for entry in ordered])
    maxima = np.array([entry.max_drift or 0.0 for entry in ordered])

    axes.hlines(positions, means, maxima, color=theme.measured, linewidth=1.6, alpha=0.5)
    axes.scatter(
        maxima,
        positions,
        s=MARKER_SIZE,
        facecolor="none",
        edgecolor=theme.measured,
        linewidth=1.4,
        zorder=4,
        label="largest",
    )
    axes.scatter(
        means,
        positions,
        s=MARKER_SIZE,
        color=theme.measured,
        edgecolor=theme.surface,
        linewidth=MARKER_RING,
        zorder=5,
        label="mean",
    )

    axes.set_yticks(positions)
    axes.set_yticklabels(_run_labels([entry.run for entry in ordered]), fontsize=8)
    axes.set_ylim(-0.8, len(ordered) - 0.2)
    axes.set_xlim(left=0.0)
    axes.set_xlabel("Ligand center-of-mass drift (\u00c5)")
    axes.set_title(title, color=theme.primary_ink, loc="left", pad=10)
    axes.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        fontsize=7.5,
        labelcolor=theme.secondary_ink,
        frameon=False,
    )

    return _save(figure, path)


def interaction_figure(
    contacts: list[ResidueContact], title: str, theme: Theme, path: Path
) -> Path:
    """Draw contact occupancy for every run side by side.

    One column per run, so an analogue that lost a hinge bond is visible against the rest.
    Van der Waals contacts are left out: they fire on nearly every nearby residue and bury
    the interactions that distinguish one ligand from another.

    Args:
        contacts: Occupancies from every run, already aggregated across repeats.
        title: Figure title, usually the campaign name.
        theme: The colors to draw with.
        path: Where to write the figure.

    Returns:
        The path written.

    Raises:
        ValueError: If nothing is left to draw once the noisy contacts are dropped.
    """
    useful = [entry for entry in contacts if entry.interaction != NOISY_INTERACTION]
    if not useful:
        raise ValueError("no interactions to draw")

    # A contact seen in a handful of frames of one run is noise, and a row of it crowds out
    # the contacts that actually separate the ligands.
    strongest: dict[tuple[str, str], float] = {}
    for entry in useful:
        key = (entry.residue, entry.interaction)
        strongest[key] = max(strongest.get(key, 0.0), entry.occupancy)
    keep = {key for key, value in strongest.items() if value >= MIN_OCCUPANCY}
    useful = [entry for entry in useful if (entry.residue, entry.interaction) in keep]
    if not useful:
        raise ValueError("no interactions to draw")

    runs = sorted({entry.run for entry in useful})
    rows = sorted({(entry.residue, entry.interaction) for entry in useful})
    grid = np.zeros((len(rows), len(runs)))
    positions = {key: index for index, key in enumerate(rows)}
    columns = {run: index for index, run in enumerate(runs)}
    for entry in useful:
        row = positions[(entry.residue, entry.interaction)]
        grid[row, columns[entry.run]] = max(grid[row, columns[entry.run]], entry.occupancy)

    height = max(2.6, 0.22 * len(rows) + 1.4)
    width = max(4.0, 0.5 * len(runs) + 3.0)
    figure = Figure(figsize=(width, height), layout="constrained")
    figure.set_facecolor(theme.surface)
    axes = figure.add_subplot(111)
    style_axes(axes, theme)
    axes.grid(False)

    colors = LinearSegmentedColormap.from_list("occupancy", SEQUENTIAL)
    image = axes.imshow(grid, cmap=colors, vmin=0.0, vmax=1.0, aspect="auto")
    axes.set_xticks(np.arange(len(runs)))
    axes.set_xticklabels(_run_labels(runs), fontsize=7, rotation=30, ha="right")
    axes.set_yticks(np.arange(len(rows)))
    axes.set_yticklabels([f"{residue}  {kind}" for residue, kind in rows], fontsize=7)
    axes.set_title(title, color=theme.primary_ink, loc="left", pad=10)

    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("occupancy", color=theme.secondary_ink, fontsize=8)
    bar.ax.tick_params(colors=theme.muted_ink, labelsize=7)
    for spine in bar.ax.spines.values():
        spine.set_edgecolor(theme.axis)

    return _save(figure, path)
