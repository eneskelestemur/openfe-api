"""Colors and matplotlib defaults shared by every figure.

The two categorical hues are validated for color-vision deficiency and for contrast against
their own surface, in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DARK", "LIGHT", "Theme", "style_axes", "theme_for"]


@dataclass(frozen=True)
class Theme:
    """One mode's colors.

    Attributes:
        surface: Figure and axes background.
        primary_ink: Titles and emphasised text.
        secondary_ink: Axis labels and annotations.
        muted_ink: Tick labels.
        grid: Hairline gridlines.
        axis: Axis spines and the identity line.
        measured: Marks for a ligand that has a measured value.
        predicted_only: Marks for a ligand with a prediction but no measurement.
        band: Fill for the tolerance band.
    """

    surface: str
    primary_ink: str
    secondary_ink: str
    muted_ink: str
    grid: str
    axis: str
    measured: str
    predicted_only: str
    band: str


LIGHT = Theme(
    surface="#fcfcfb",
    primary_ink="#0b0b0b",
    secondary_ink="#52514e",
    muted_ink="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    measured="#2a78d6",
    predicted_only="#eb6834",
    band="#ebeae4",
)

DARK = Theme(
    surface="#1a1a19",
    primary_ink="#f0efec",
    secondary_ink="#c3c2b7",
    muted_ink="#898781",
    grid="#2c2c2a",
    axis="#383835",
    measured="#3987e5",
    predicted_only="#d95926",
    band="#262624",
)


def theme_for(dark: bool) -> Theme:
    """Return the colors for one mode.

    Args:
        dark: Whether to use the dark surface.

    Returns:
        The theme.
    """
    return DARK if dark else LIGHT


def style_axes(axes: Any, theme: Theme) -> None:
    """Put a theme on one set of axes.

    Applied per figure rather than through matplotlib's global settings, because the
    service draws inside a background thread and those settings are shared.

    Args:
        axes: The axes to style.
        theme: The colors to apply.
    """
    axes.set_facecolor(theme.surface)
    axes.grid(True, color=theme.grid, linewidth=0.7, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme.axis)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=theme.muted_ink, labelsize=8, length=3, width=0.8)
    axes.xaxis.label.set_color(theme.secondary_ink)
    axes.yaxis.label.set_color(theme.secondary_ink)
