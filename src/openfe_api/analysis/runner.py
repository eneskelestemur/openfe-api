"""Turning a finished campaign into figures and tables.

Read-only: these readers take result files rather than the gathering entry points, which
mark finished runs done as a side effect.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

from openfe_api.analysis.collected import collect_plots
from openfe_api.analysis.correlation import Correlation, Prediction, correlate
from openfe_api.analysis.experimental import Measurement, read_measurements
from openfe_api.analysis.figures import (
    correlation_figure,
    interaction_figure,
    ranking_figure,
    stability_figure,
)
from openfe_api.analysis.interactions import ResidueContact, measure_interactions
from openfe_api.analysis.scenes import SCENES_DIR, Scene, write_scene
from openfe_api.analysis.style import Theme, theme_for
from openfe_api.analysis.trajectory import (
    RepeatMetrics,
    RunStability,
    TrajectorySource,
    find_sources,
    measure_repeat,
    summarize_run,
)
from openfe_api.campaign import ANALYSIS_DIR, PLANS_DIR, RUNS_DIR, Campaign
from openfe_api.exceptions import InputValidationError, OpenFEAPIError
from openfe_api.log import get_logger
from openfe_api.provenance import collect_provenance
from openfe_api.results.gather import gather_results
from openfe_api.results.network import gather_network, read_rbfe_edge
from openfe_api.results.runner import result_paths
from openfe_api.results.septop import read_septop_edge
from openfe_api.schema.abfe import AbfeRequest
from openfe_api.schema.md import MdRequest
from openfe_api.schema.rbfe import RbfeRequest
from openfe_api.schema.septop import SepTopRequest

__all__ = [
    "AnalysisOutcome",
    "NoFreeEnergyError",
    "analyze_campaign",
    "measure_trajectories",
    "read_predictions",
]


class NoFreeEnergyError(OpenFEAPIError):
    """Raised when a protocol produces no free energy to compare."""


logger = get_logger(__name__)

CORRELATION_FIGURE = "correlation.png"
RANKING_FIGURE = "ranking.png"
PREDICTIONS_TABLE = "predictions.tsv"
CORRELATION_TABLE = "correlation.tsv"
METRICS_FILE = "metrics.json"
TRAJECTORY_TABLE = "trajectory.tsv"
STABILITY_TABLE = "stability.tsv"
STABILITY_FIGURE = "stability.png"
INTERACTION_TABLE = "interactions.tsv"
INTERACTION_FIGURE = "interactions.png"

_REPEAT_GLOBS = ("*/repeat*", "*/*/repeat*")


@dataclass
class AnalysisOutcome:
    """What an analysis produced.

    Attributes:
        directory: Where the files were written.
        protocol: The campaign's protocol.
        predictions: The calculated free energies the analysis read.
        correlation: The comparison against measured values, when any were given.
        stability: One aggregate per run whose trajectories were measured.
        scenes: One representative pose per run, when the trajectories were measured.
        figures: Figures written, keyed by what each one shows.
        tables: Tables written, keyed by what each one holds.
        warnings: Concerns the reader should know about.
    """

    directory: Path
    protocol: str
    predictions: list[Prediction] = field(default_factory=list)
    correlation: Correlation | None = None
    stability: list[RunStability] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    figures: dict[str, Path] = field(default_factory=dict)
    tables: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def read_predictions(campaign: Campaign) -> tuple[list[Prediction], bool]:
    """Read a campaign's calculated free energies.

    Args:
        campaign: The campaign to read.

    Returns:
        The predictions, and whether they are relative to the network rather than absolute.

    Raises:
        OpenFEAPIError: If the protocol produces no free energy to analyze.
        InputValidationError: If the campaign has nothing gathered yet.
    """
    match campaign.manifest.request:
        case AbfeRequest():
            paths = [path for found in result_paths(campaign).values() for path in found]
            results, _ = gather_results(paths)
            predictions = [
                Prediction(result.name, result.dg, result.uncertainty) for result in results
            ]
            relative = False
        case RbfeRequest() | SepTopRequest():
            reader = (
                read_septop_edge
                if isinstance(campaign.manifest.request, SepTopRequest)
                else read_rbfe_edge
            )
            summary = campaign.directory / PLANS_DIR / "network.json"
            if not summary.is_file():
                raise InputValidationError(
                    f"{campaign.manifest.name}: no planned network at {summary}, so there is "
                    "nothing to analyze yet. Run 'plan' and 'submit' first."
                )
            planned = json.loads(summary.read_text(encoding="utf-8"))
            network = gather_network(campaign.directory, planned, reader)
            predictions = [
                Prediction(entry.name, entry.dg, entry.uncertainty) for entry in network.ligands
            ]
            relative = True
        case MdRequest():
            raise NoFreeEnergyError(
                f"{campaign.manifest.name}: a plain MD campaign produces no free energy. Pass "
                "'--trajectories' to measure how far its ligands moved instead."
            )
        case _:
            raise OpenFEAPIError(
                f"analysis does not handle the '{campaign.manifest.protocol}' protocol"
            )

    if not predictions:
        raise InputValidationError(
            f"{campaign.manifest.name}: no finished results to analyze. Check 'status', and "
            "run 'gather' to see what each run produced."
        )
    return predictions, relative


def _write_predictions(predictions: list[Prediction], path: Path) -> Path:
    """Write the calculated free energies as a table."""
    lines = ["ligand\tdg_kcal_per_mol\tuncertainty_kcal_per_mol"]
    lines += [
        f"{entry.name}\t{entry.dg:.4f}\t{entry.uncertainty:.4f}"
        for entry in sorted(predictions, key=lambda entry: entry.dg)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_correlation(outcome: Correlation, path: Path) -> Path:
    """Write the paired points and the unpaired predictions as one table."""
    lines = [
        "ligand\tpredicted_kcal_per_mol\tpredicted_uncertainty\tmeasured_kcal_per_mol\tmeasured_uncertainty"
    ]
    for point in sorted(outcome.paired, key=lambda point: point.measured):
        measured_uncertainty = (
            f"{point.measured_uncertainty:.4f}" if point.measured_uncertainty is not None else ""
        )
        lines.append(
            f"{point.name}\t{point.predicted:.4f}\t{point.predicted_uncertainty:.4f}"
            f"\t{point.measured:.4f}\t{measured_uncertainty}"
        )
    for entry in sorted(outcome.predicted_only, key=lambda entry: entry.dg):
        lines.append(f"{entry.name}\t{entry.dg:.4f}\t{entry.uncertainty:.4f}\t\t")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_metrics(outcome: Correlation | None, path: Path) -> Path:
    """Write the metrics as JSON, so a caller can read them without parsing a figure.

    Written whether or not anything was measured: it is also where the versions behind
    every other file in the directory are recorded.
    """
    payload: dict[str, object] = {
        "paired": len(outcome.paired) if outcome else 0,
        "predicted_only": len(outcome.predicted_only) if outcome else 0,
        "unmatched": outcome.unmatched if outcome else [],
        "aligned": outcome.aligned if outcome else False,
        "shift_kcal_per_mol": outcome.shift if outcome else None,
        "pearson": outcome.pearson if outcome else None,
        "spearman": outcome.spearman if outcome else None,
        "kendall": outcome.kendall if outcome else None,
        "mue_kcal_per_mol": outcome.mue if outcome else None,
        "rmse_kcal_per_mol": outcome.rmse if outcome else None,
        "provenance": collect_provenance(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _work_items(campaign: Campaign) -> list[tuple[TrajectorySource, str, int]]:
    """List every trajectory to measure, with the run and repeat it belongs to."""
    runs = campaign.directory / RUNS_DIR
    if not runs.is_dir():
        return []

    items: list[tuple[TrajectorySource, str, int]] = []
    seen: set[Path] = set()
    for pattern in _REPEAT_GLOBS:
        for repeat_dir in sorted(runs.glob(pattern)):
            if repeat_dir in seen or not repeat_dir.is_dir():
                continue
            seen.add(repeat_dir)
            run = repeat_dir.relative_to(runs).parts[0]
            repeat = int(repeat_dir.name.removeprefix("repeat"))
            items += [(source, run, repeat) for source in find_sources(repeat_dir)]
    return items


def _measure_one(
    item: tuple[TrajectorySource, str, int],
) -> tuple[list[RepeatMetrics], list[ResidueContact], str | None]:
    """Measure one trajectory, returning the failure instead of raising it.

    Any failure is caught, not only ours: a worker that raises takes the pool down with it,
    and one unreadable repeat should not cost a campaign an hour of measurement.
    """
    source, run, repeat = item
    try:
        return measure_repeat(source, run, repeat), measure_interactions(source, run, repeat), None
    except OpenFEAPIError as error:
        return [], [], str(error)
    except Exception as error:
        return [], [], f"{source.trajectory}: {type(error).__name__}: {error}"


def measure_trajectories(
    campaign: Campaign, workers: int = 1
) -> tuple[list[RepeatMetrics], list[ResidueContact], list[str]]:
    """Measure every repeat of every run that produced a trajectory.

    A repeat that cannot be measured is reported and the rest carry on: a short or
    interrupted run should not cost the campaign its whole analysis.

    Args:
        campaign: The campaign to read.
        workers: How many trajectories to measure at once. A worker spends a couple of
            seconds starting, so more of them only pays where each trajectory takes
            minutes, which is the production case rather than a smoke test.

    Returns:
        Every repeat measured, the contacts its ligands made, and a message for each one
        that could not be measured.
    """
    items = _work_items(campaign)
    if not items:
        return [], [], []

    if workers <= 1:
        outcomes = [_measure_one(item) for item in items]
    else:
        # Spawned rather than forked: the readers hold netCDF and HDF5 handles, which do not
        # survive a fork reliably.
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                outcomes = list(pool.map(_measure_one, items))
        except BrokenProcessPool:
            # A worker killed outright, usually for memory, cannot report anything back.
            logger.warning(
                "A worker died with %d of them running; measuring the rest one at a time. "
                "Lower --workers or ask for more memory.",
                workers,
            )
            outcomes = [_measure_one(item) for item in items]

    measured: list[RepeatMetrics] = []
    contacts: list[ResidueContact] = []
    skipped: list[str] = []
    for metrics, found, failure in outcomes:
        measured += metrics
        contacts += found
        if failure is not None:
            skipped.append(failure)
            logger.warning("Skipped a trajectory: %s", failure)

    return measured, contacts, skipped


def _write_trajectory(metrics: list[RepeatMetrics], path: Path) -> Path:
    """Write one row per frame."""
    header = "run\trepeat\tphase\tligand\ttime_ps\tprotein_rmsd\tligand_rmsd\tligand_drift"
    lines = [header]
    for entry in metrics:
        for index, time_ps in enumerate(entry.times_ps):
            values = [
                entry.run,
                str(entry.repeat),
                entry.phase,
                entry.ligand,
                f"{time_ps:.3f}",
                _at(entry.protein_rmsd, index),
                _at(entry.ligand_rmsd, index),
                _at(entry.ligand_drift, index),
            ]
            lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _at(values: list[float], index: int) -> str:
    """Render one frame's value, blank where the metric does not apply."""
    return f"{values[index]:.4f}" if index < len(values) else ""


def _write_stability(summaries: list[RunStability], path: Path) -> Path:
    """Write one row per run."""
    lines = [
        "run\trepeats\tmean_drift\tmax_drift\tmean_ligand_rmsd\tmean_protein_rmsd\tworst_repeat"
    ]
    for entry in sorted(summaries, key=lambda item: item.max_drift or 0.0, reverse=True):
        lines.append(
            "\t".join(
                [
                    entry.run,
                    str(entry.repeats),
                    _number(entry.mean_drift),
                    _number(entry.max_drift),
                    _number(entry.mean_ligand_rmsd),
                    _number(entry.mean_protein_rmsd),
                    "" if entry.worst_repeat is None else str(entry.worst_repeat),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _number(value: float | None) -> str:
    """Render a value that may not have been measured."""
    return "" if value is None else f"{value:.4f}"


def analyze_campaign(
    campaign: Campaign,
    measurements: Path | dict[str, Measurement] | None = None,
    dark: bool = False,
    trajectories: bool = False,
    workers: int = 1,
) -> AnalysisOutcome:
    """Read a campaign's results and write its analysis.

    Args:
        campaign: The campaign to analyze.
        measurements: A table of measured free energies, or the measurements themselves.
            Without them the analysis reports the predicted ranking alone.
        dark: Whether to draw the figures for a dark surface.
        trajectories: Whether to measure how far each ligand moved during its run. This
            reads the trajectories, which takes far longer than the rest of the analysis.
        workers: How many trajectories to measure at once.

    Returns:
        What was written, and the metrics behind it.

    Raises:
        OpenFEAPIError: If the protocol produces no free energy to analyze.
        InputValidationError: If nothing has been gathered, or the measured table is unusable.
    """
    try:
        predictions, relative = read_predictions(campaign)
    except NoFreeEnergyError:
        if not trajectories:
            raise
        predictions, relative = [], False

    directory = campaign.directory / ANALYSIS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    theme = theme_for(dark)

    outcome = AnalysisOutcome(
        directory=directory, protocol=campaign.manifest.protocol, predictions=predictions
    )
    name = campaign.manifest.name

    if predictions:
        outcome.tables["predictions"] = _write_predictions(
            predictions, directory / PREDICTIONS_TABLE
        )
        outcome.figures["ranking"] = ranking_figure(
            predictions, f"{name} · predicted ranking", theme, directory / RANKING_FIGURE
        )

    for plot in collect_plots(campaign):
        outcome.figures.setdefault(f"{plot.run}_{plot.kind}", plot.link)

    if trajectories:
        _add_trajectories(campaign, outcome, theme, directory, workers)

    if measurements is None or not predictions:
        outcome.tables["metrics"] = _write_metrics(None, directory / METRICS_FILE)
        logger.info("Analyzed '%s': %d ligand(s), no measured values", name, len(predictions))
        return outcome

    table = read_measurements(measurements) if isinstance(measurements, Path) else measurements
    correlation = correlate(predictions, table, aligned=relative)
    outcome.correlation = correlation
    outcome.tables["metrics"] = _write_metrics(correlation, directory / METRICS_FILE)
    if correlation.unmatched:
        outcome.warnings.append(
            f"{len(correlation.unmatched)} measured ligand(s) are not in the campaign: "
            + ", ".join(correlation.unmatched)
        )
    if not correlation.paired:
        outcome.warnings.append(
            "no ligand has both a prediction and a measurement, so only the ranking is shown"
        )
        return outcome

    if correlation.pearson is not None and correlation.pearson < 0.0:
        outcome.warnings.append(
            f"the predictions run against the measurements (Pearson R = {correlation.pearson:.2f})"
            ". Check that the measured column is a dG in kcal/mol: an affinity such as a pIC50 "
            "rises where a free energy falls, which reads as an anti-correlation."
        )

    outcome.tables["correlation"] = _write_correlation(correlation, directory / CORRELATION_TABLE)
    outcome.figures["correlation"] = correlation_figure(
        correlation, f"{name} · predicted vs measured", theme, directory / CORRELATION_FIGURE
    )
    logger.info(
        "Analyzed '%s': %d paired, %d without a measurement",
        name,
        len(correlation.paired),
        len(correlation.predicted_only),
    )
    return outcome


def _add_trajectories(
    campaign: Campaign, outcome: AnalysisOutcome, theme: Theme, directory: Path, workers: int
) -> None:
    """Measure the trajectories and write what they show into the analysis."""
    metrics, contacts, skipped = measure_trajectories(campaign, workers)
    outcome.warnings += skipped
    if not metrics:
        outcome.warnings.append("no trajectory could be measured")
        return

    by_run: dict[str, list[RepeatMetrics]] = {}
    for entry in metrics:
        by_run.setdefault(entry.run, []).append(entry)
    outcome.stability = [summarize_run(run, rows) for run, rows in sorted(by_run.items())]

    outcome.tables["trajectory"] = _write_trajectory(metrics, directory / TRAJECTORY_TABLE)
    outcome.tables["stability"] = _write_stability(outcome.stability, directory / STABILITY_TABLE)
    if contacts:
        outcome.tables["interactions"] = _write_interactions(
            contacts, directory / INTERACTION_TABLE
        )
        try:
            outcome.figures["interactions"] = interaction_figure(
                contacts,
                f"{campaign.manifest.name} · contact occupancy",
                theme,
                directory / INTERACTION_FIGURE,
            )
        except ValueError:
            outcome.warnings.append("only van der Waals contacts were found, so no heatmap")

    outcome.scenes = _write_scenes(campaign, directory)
    for scene in outcome.scenes:
        outcome.figures[f"{scene.run}_scene"] = scene.page

    try:
        outcome.figures["stability"] = stability_figure(
            outcome.stability,
            f"{campaign.manifest.name} · ligand drift",
            theme,
            directory / STABILITY_FIGURE,
        )
    except ValueError:
        outcome.warnings.append("no run had a drift to draw, so no stability figure was written")


def _write_interactions(contacts: list[ResidueContact], path: Path) -> Path:
    """Write occupancy per run, residue and interaction, averaged over repeats."""
    totals: dict[tuple[str, str, str, str], list[float]] = {}
    for entry in contacts:
        key = (entry.run, entry.ligand, entry.residue, entry.interaction)
        totals.setdefault(key, []).append(entry.occupancy)

    lines = ["run\tligand\tresidue\tinteraction\toccupancy\trepeats"]
    for (run, ligand, residue, interaction), values in sorted(totals.items()):
        mean = sum(values) / len(values)
        lines.append(f"{run}\t{ligand}\t{residue}\t{interaction}\t{mean:.4f}\t{len(values)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_scenes(campaign: Campaign, directory: Path) -> list[Scene]:
    """Write one representative pose per run, skipping runs that have no ligand."""
    scenes: list[Scene] = []
    seen: set[str] = set()
    for source, run, repeat in _work_items(campaign):
        if run in seen or repeat != 1:
            continue
        try:
            scenes.append(write_scene(source, run, directory / SCENES_DIR))
            seen.add(run)
        except OpenFEAPIError as error:
            logger.debug("No scene for '%s': %s", run, error)
    return scenes
