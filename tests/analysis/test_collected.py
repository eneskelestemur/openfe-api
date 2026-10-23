"""Tests for linking OpenFE's own plots into the analysis directory."""

from __future__ import annotations

import os
from pathlib import Path

from helpers import abfe_campaign
from openfe_api.analysis.collected import PLOT_NAMES, collect_plots
from openfe_api.campaign import Campaign


def _plot(campaign: Campaign, run: str, repeat: str, unit: str, name: str) -> Path:
    """Write a stand-in for one of OpenFE's plots."""
    path = campaign.directory / "runs" / run / repeat / f"shared_{unit}-abc_attempt_0" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG stand-in")
    return path


def test_plots_are_linked_not_copied(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    source = _plot(campaign, "mini", "repeat1", "ABFEComplexAnalysisUnit", PLOT_NAMES[0])

    collected = collect_plots(campaign)

    assert len(collected) == 1
    assert collected[0].link.is_symlink()
    assert collected[0].link.resolve() == source.resolve()


def test_the_link_name_says_where_it_came_from(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _plot(campaign, "mini", "repeat2", "ABFEComplexAnalysisUnit", "mbar_overlap_matrix.png")

    collected = collect_plots(campaign)

    assert collected[0].link.name == "mini_repeat2_complex_mbar_overlap_matrix.png"
    assert collected[0].phase == "complex"
    assert collected[0].repeat == 2


def test_an_rbfe_edge_takes_its_phase_from_the_path(tmp_path: Path) -> None:
    """An RBFE edge runs each phase in its own directory rather than one unit."""
    campaign = abfe_campaign(tmp_path, planned=False)
    path = (
        campaign.directory
        / "runs"
        / "a_to_b"
        / "solvent"
        / "repeat1"
        / "shared_RelativeHybridTopologyProtocolUnit-abc_attempt_0"
        / PLOT_NAMES[0]
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x89PNG stand-in")

    collected = collect_plots(campaign)

    assert collected[0].phase == "solvent"
    assert collected[0].link.name.startswith("a_to_b_repeat1_solvent_")


def test_links_are_relative_so_the_campaign_can_move(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _plot(campaign, "mini", "repeat1", "ABFEComplexAnalysisUnit", PLOT_NAMES[0])

    collected = collect_plots(campaign)

    assert not os.path.isabs(os.readlink(collected[0].link))


def test_collecting_twice_replaces_the_links(tmp_path: Path) -> None:
    """An analysis is meant to be re-runnable."""
    campaign = abfe_campaign(tmp_path, planned=False)
    _plot(campaign, "mini", "repeat1", "ABFEComplexAnalysisUnit", PLOT_NAMES[0])

    collect_plots(campaign)
    collected = collect_plots(campaign)

    assert len(collected) == 1
    assert collected[0].link.is_symlink()


def test_other_images_are_left_alone(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    _plot(campaign, "mini", "repeat1", "ABFEComplexAnalysisUnit", "something_else.png")

    assert collect_plots(campaign) == []


def test_a_campaign_with_no_runs_collects_nothing(tmp_path: Path) -> None:
    assert collect_plots(abfe_campaign(tmp_path, planned=False)) == []


def test_a_plot_outside_a_repeat_is_skipped(tmp_path: Path) -> None:
    campaign = abfe_campaign(tmp_path, planned=False)
    path = campaign.directory / "runs" / "mini" / "shared_Unit-abc_attempt_0" / PLOT_NAMES[0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x89PNG stand-in")

    assert collect_plots(campaign) == []
