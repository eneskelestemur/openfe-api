"""Gathering a campaign's results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openfe_api.campaign import RESULTS_DIR, Campaign, RunState
from openfe_api.exceptions import OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.results.gather import LigandResult, RepeatResult, gather_results, write_tsv
from openfe_api.results.md import MdRunResult, gather_md, write_md_tsv
from openfe_api.results.network import (
    EdgeReader,
    NetworkResults,
    gather_network,
    read_rbfe_edge,
    write_edge_tsv,
    write_ligand_tsv,
)
from openfe_api.results.qc import QualityReport, check_edge, check_ligand, check_network
from openfe_api.results.septop import read_septop_edge
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = ["GatherOutcome", "gather_campaign", "result_paths"]

logger = get_logger(__name__)


@dataclass
class GatherOutcome:
    """What gathering found.

    Attributes:
        results: Combined per-ligand results.
        reports: Quality reports keyed by ligand name.
        failed: Repeats that could not be used.
        tsv: The results table written to disk, if one was written.
        missing: Names of runs with no results at all.
        network: The gathered network, for RBFE and SepTop.
        md: The gathered simulations, for plain MD.
        tables: Tables written to disk, keyed by what they hold.
    """

    results: list[LigandResult] = field(default_factory=list)
    reports: dict[str, QualityReport] = field(default_factory=dict)
    failed: list[RepeatResult] = field(default_factory=list)
    tsv: Path | None = None
    missing: list[str] = field(default_factory=list)
    network: NetworkResults | None = None
    md: list[MdRunResult] = field(default_factory=list)
    tables: dict[str, Path] = field(default_factory=dict)


def result_paths(campaign: Campaign, only: list[str] | None = None) -> dict[str, list[Path]]:
    """Find the result files a campaign has produced.

    Args:
        campaign: The campaign to search.
        only: Names of runs to include. If None, every run is included.

    Returns:
        Result file paths keyed by run name, including runs with none.
    """
    selected = set(only) if only is not None else None
    found: dict[str, list[Path]] = {}

    for name in campaign.manifest.runs:
        if selected is not None and name not in selected:
            continue
        run_dir = campaign.directory / "runs" / name
        found[name] = sorted(run_dir.glob("repeat*/results.json")) if run_dir.is_dir() else []

    return found


def gather_campaign(
    campaign: Campaign,
    only: list[str] | None = None,
    write: bool = True,
) -> GatherOutcome:
    """Collect a campaign's results, check them, and write the results tables.

    Runs whose repeats have all finished are marked done, so gathering also brings the
    manifest up to date after a batch of jobs completes.

    Args:
        campaign: The campaign to gather.
        only: Names of runs to gather. If None, every run is gathered.
        write: Whether to write the result tables in the campaign directory.

    Returns:
        The gathered results and their quality reports.

    Raises:
        OpenFEAPIError: If the campaign's protocol has no gathering path.
    """
    match campaign.manifest.request:
        case AbfeRequest():
            return _gather_complexes(campaign, only, write)
        case RbfeRequest():
            return _gather_edges(campaign, write, read_rbfe_edge)
        case SepTopRequest():
            return _gather_edges(campaign, write, read_septop_edge)
        case MdRequest():
            return _gather_md(campaign, only, write)
        case _:
            raise OpenFEAPIError(
                f"gathering does not handle the '{campaign.manifest.protocol}' protocol"
            )


def _gather_md(campaign: Campaign, only: list[str] | None, write: bool) -> GatherOutcome:
    """Collect what a plain MD campaign produced, which is files rather than free energies."""
    outcome = GatherOutcome()
    selected = set(only) if only is not None else None
    names = [name for name in campaign.manifest.runs if selected is None or name in selected]

    outcome.md = gather_md(campaign.directory, names)
    outcome.missing = [run.name for run in outcome.md if not run.repeats]

    for run in outcome.md:
        record = campaign.manifest.runs.get(run.name)
        if record is None or record.state is RunState.DONE:
            continue
        if len(run.finished) >= record.repeats:
            try:
                campaign.set_state(run.name, RunState.DONE)
            except Exception:
                logger.debug("Could not mark '%s' done from its current state", run.name)

    if write:
        path = campaign.directory / RESULTS_DIR / "simulations.tsv"
        write_md_tsv(outcome.md, path)
        outcome.tables = {"simulations": path}
        outcome.tsv = path

    return outcome


def _gather_edges(campaign: Campaign, write: bool, read_edge: EdgeReader) -> GatherOutcome:
    """Gather a network campaign into per-edge and per-ligand free energies."""
    outcome = GatherOutcome()
    summary = campaign.directory / "plans" / "network.json"
    if not summary.is_file():
        logger.warning("No planned network found at %s", summary)
        return outcome

    try:
        planned = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        logger.warning("Could not read %s: %s", summary, error)
        return outcome

    results = gather_network(campaign.directory, planned, read_edge)
    outcome.network = results
    outcome.missing = [edge.name for edge in results.edges if not edge.ok]
    outcome.reports = {edge.name: check_edge(edge) for edge in results.edges}
    outcome.reports["network"] = check_network(results)

    for edge in results.edges:
        if not edge.ok:
            continue
        record = campaign.manifest.runs.get(edge.name)
        if record is None or record.state is RunState.DONE:
            continue
        finished = all(
            len([entry for entry in entries if entry.ok]) >= record.repeats
            for entries in edge.phases.values()
        )
        if finished:
            try:
                campaign.set_state(edge.name, RunState.DONE)
            except Exception:
                logger.debug("Could not mark '%s' done from its current state", edge.name)

    if write:
        edges_path = campaign.directory / RESULTS_DIR / "edges.tsv"
        ligands_path = campaign.directory / RESULTS_DIR / "ligands.tsv"
        write_edge_tsv(results.edges, edges_path)
        write_ligand_tsv(results.ligands, ligands_path)
        outcome.tables = {"edges": edges_path, "ligands": ligands_path}
        outcome.tsv = edges_path

    return outcome


def _gather_complexes(campaign: Campaign, only: list[str] | None, write: bool) -> GatherOutcome:
    """Gather an ABFE campaign into per-ligand binding free energies."""
    by_run = result_paths(campaign, only=only)
    outcome = GatherOutcome(missing=[name for name, paths in by_run.items() if not paths])

    paths = [path for found in by_run.values() for path in found]
    if not paths:
        logger.warning("No result files found in %s", campaign.directory / "runs")
        return outcome

    outcome.results, outcome.failed = gather_results(paths)
    outcome.reports = {result.name: check_ligand(result) for result in outcome.results}

    for name, found in by_run.items():
        record = campaign.manifest.runs[name]
        if found and len(found) == record.repeats and record.state is not RunState.DONE:
            usable = [path for path in found if path.is_file()]
            if len(usable) == record.repeats:
                try:
                    campaign.set_state(name, RunState.DONE)
                except Exception:
                    logger.debug("Could not mark '%s' done from its current state", name)

    if write:
        outcome.tsv = campaign.directory / RESULTS_DIR / "results.tsv"
        write_tsv(outcome.results, outcome.tsv)
        outcome.tables = {"results": outcome.tsv}

    return outcome
