"""Tests for the figures an analysis writes."""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import colors, image

from openfe_api.analysis.correlation import Correlation, Prediction, correlate
from openfe_api.analysis.experimental import Measurement
from openfe_api.analysis.figures import (
    LABEL_LIMIT,
    _run_labels,
    correlation_figure,
    ranking_figure,
)
from openfe_api.analysis.style import DARK, LIGHT, Theme, theme_for


def _predictions(count: int) -> list[Prediction]:
    """Build a spread of predictions."""
    return [Prediction(f"lig_{index}", -9.0 + 0.4 * index, 0.25) for index in range(count)]


def _correlation(paired: int = 5, unpaired: int = 0) -> Correlation:
    """Build an outcome with the given number of paired and unpaired ligands."""
    predictions = _predictions(paired + unpaired)
    measurements = {
        entry.name: Measurement(entry.name, entry.dg + 0.2) for entry in predictions[:paired]
    }
    return correlate(predictions, measurements, aligned=False)


def _corner(path: Path) -> list[float]:
    """Return the color of the saved figure's top-left pixel."""
    pixels = image.imread(path)
    return [float(channel) for channel in pixels[0, 0, :3]]


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_the_correlation_figure_is_written(theme: Theme, tmp_path: Path) -> None:
    written = correlation_figure(_correlation(), "campaign", theme, tmp_path / "corr.png")

    assert written.is_file()
    assert written.stat().st_size > 0


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_a_figure_is_saved_on_its_own_surface(theme: Theme, tmp_path: Path) -> None:
    """Saving on the matplotlib default would frame a dark figure in white."""
    path = correlation_figure(_correlation(), "campaign", theme, tmp_path / "corr.png")

    assert _corner(path) == pytest.approx(list(colors.to_rgb(theme.surface)), abs=0.01)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_the_ranking_figure_is_saved_on_its_own_surface(theme: Theme, tmp_path: Path) -> None:
    path = ranking_figure(_predictions(6), "campaign", theme, tmp_path / "rank.png")

    assert _corner(path) == pytest.approx(list(colors.to_rgb(theme.surface)), abs=0.01)


def test_a_campaign_with_no_measurements_still_draws(tmp_path: Path) -> None:
    """The unpaired ligands are the whole figure when nothing was measured."""
    outcome = _correlation(paired=0, unpaired=4)

    written = correlation_figure(outcome, "campaign", theme_for(False), tmp_path / "corr.png")

    assert written.is_file()


def test_the_rail_drops_its_labels_once_they_stop_fitting(tmp_path: Path) -> None:
    crowded = _correlation(paired=2, unpaired=LABEL_LIMIT + 1)

    written = correlation_figure(crowded, "campaign", theme_for(False), tmp_path / "corr.png")

    assert written.is_file()


def test_an_empty_correlation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no ligands"):
        correlation_figure(Correlation(), "campaign", theme_for(False), tmp_path / "corr.png")


def test_an_empty_ranking_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no predictions"):
        ranking_figure([], "campaign", theme_for(False), tmp_path / "rank.png")


def test_the_directory_is_created(tmp_path: Path) -> None:
    written = ranking_figure(
        _predictions(3), "campaign", theme_for(False), tmp_path / "deep" / "rank.png"
    )

    assert written.is_file()


def test_run_labels_drop_the_prefix_every_edge_shares() -> None:
    """Edge names repeat the protein and the reference ligand, which crowds out the plot."""
    labels = _run_labels(
        ["pgk2_lig_a_to_pgk2_lig_x", "pgk2_lig_a_to_pgk2_lig_y", "pgk2_lig_a_to_pgk2_lig_z"]
    )

    assert labels == ["x", "y", "z"]


def test_a_single_run_keeps_its_name() -> None:
    assert _run_labels(["pgk2_lig_a_to_pgk2_lig_x"]) == ["pgk2_lig_a_to_pgk2_lig_x"]


def test_a_long_label_is_cut_rather_than_left_to_overflow() -> None:
    labels = _run_labels(["short", "a" * 60], limit=10)

    assert labels[1] == "a" * 9 + "\u2026"
