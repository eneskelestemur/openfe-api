"""Colors and matplotlib defaults shared by every figure.

The two categorical hues are validated for colour-vision deficiency and for contrast
against their own surface, in both modes, so a figure never carries meaning in colour alone
that a reader cannot separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["DARK", "LIGHT", "Theme", "rc_params", "theme_for"]


@dataclass(frozen=True)
class Theme:
    """One mode's colours.

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
    """Return the colours for one mode.

    Args:
        dark: Whether to use the dark surface.

    Returns:
        The theme.
    """
    return DARK if dark else LIGHT


def rc_params(theme: Theme) -> dict[str, Any]:
    """Return the matplotlib settings that put a theme on a figure.

    Args:
        theme: The colours to apply.

    Returns:
        Settings to pass to ``matplotlib.rc_context``.
    """
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.secondary_ink,
        "axes.titlecolor": theme.primary_ink,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "xtick.color": theme.muted_ink,
        "ytick.color": theme.muted_ink,
        "xtick.labelcolor": theme.muted_ink,
        "ytick.labelcolor": theme.muted_ink,
        "text.color": theme.primary_ink,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.frameon": False,
        "figure.dpi": 160,
    }
