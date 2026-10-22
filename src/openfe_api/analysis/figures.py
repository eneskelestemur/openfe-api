"""The figures an analysis writes.

Drawing goes through matplotlib's object interface rather than pyplot: the service runs an
analysis in a background thread, and pyplot keeps global state that two of them would share.
Every figure has its numbers written beside it, so a reader never depends on the picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from openfe_api.analysis.correlation import Correlation, Prediction
from openfe_api.analysis.style import Theme, style_axes
from openfe_api.log import get_logger

__all__ = ["TOLERANCE", "correlation_figure", "ranking_figure"]

logger = get_logger(__name__)

TOLERANCE = 1.0
"""Half-width of the band drawn around agreement, in kcal/mol."""

MARKER_SIZE = 46
MARKER_RING = 1.2
RAIL_WIDTH = 0.3
LABEL_LIMIT = 12


def _save(figure: Figure, path: Path) -> Path:
    """Render a figure to disk.

    Args:
        figure: The figure to write.
        path: Where to write it.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    # The figure's own colour, not the matplotlib default, or a dark figure is saved with a
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

    Ligands with no measurement are drawn in their own narrow panel on the right, at their
    predicted value, so a campaign is never reduced to only the subset that has data.

    Args:
        outcome: The paired points, the unpaired predictions and the metrics.
        title: Figure title, usually the campaign name.
        theme: The colours to draw with.
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
    subtitle = "both axes centred on their mean" if outcome.centered else ""
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
            0.96,
            0.04,
            f"shaded: within {TOLERANCE:.0f} kcal/mol",
            transform=axes.transAxes,
            va="bottom",
            ha="right",
            fontsize=7.5,
            color=theme.muted_ink,
        )

    if has_rail:
        _draw_rail(figure, axes_grid, outcome.predicted_only, limits, theme)

    return _save(figure, path)


def _spread_labels(values: np.ndarray, span: float) -> list[float]:
    """Nudge label positions apart so neighbouring names stay readable.

    Args:
        values: Where each label would sit, in data coordinates.
        span: Height of the axis in data coordinates.

    Returns:
        A position per label, in the order given.
    """
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
    axes_grid: object,
    entries: list[Prediction],
    limits: tuple[float, float],
    theme: Theme,
) -> None:
    """Draw the ligands that have a prediction but nothing to compare it against."""
    rail = figure.add_subplot(axes_grid[0, 1])  # type: ignore[index]
    style_axes(rail, theme)
    rail.grid(False)
    rail.set_facecolor(theme.band)
    rail.set_ylim(*limits)
    rail.set_xlim(-0.5, 0.5)
    rail.set_xticks([])
    rail.spines["left"].set_visible(False)
    rail.spines["bottom"].set_visible(False)

    values = np.array([entry.dg for entry in entries], dtype=float)
    rail.errorbar(
        np.zeros(values.size),
        values,
        yerr=[entry.uncertainty for entry in entries],
        fmt="none",
        ecolor=theme.predicted_only,
        elinewidth=1.1,
        alpha=0.55,
    )
    rail.scatter(
        np.zeros(values.size),
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
        theme: The colours to draw with.
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
