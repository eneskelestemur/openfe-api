"""Turning a finished campaign into figures and tables.

Read-only: these readers take result files rather than the gathering entry points, which
mark finished runs done as a side effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openfe_api.analysis.correlation import Correlation, Prediction, correlate
from openfe_api.analysis.experimental import Measurement, read_measurements
from openfe_api.analysis.figures import correlation_figure, ranking_figure
from openfe_api.analysis.style import theme_for
from openfe_api.campaign import ANALYSIS_DIR, PLANS_DIR, Campaign
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

__all__ = ["AnalysisOutcome", "analyze_campaign", "read_predictions"]

logger = get_logger(__name__)

CORRELATION_FIGURE = "correlation.png"
RANKING_FIGURE = "ranking.png"
PREDICTIONS_TABLE = "predictions.tsv"
CORRELATION_TABLE = "correlation.tsv"
METRICS_FILE = "metrics.json"


@dataclass
class AnalysisOutcome:
    """What an analysis produced.

    Attributes:
        directory: Where the files were written.
        protocol: The campaign's protocol.
        predictions: The calculated free energies the analysis read.
        correlation: The comparison against measured values, when any were given.
        figures: Figures written, keyed by what each one shows.
        tables: Tables written, keyed by what each one holds.
        warnings: Concerns the reader should know about.
    """

    directory: Path
    protocol: str
    predictions: list[Prediction] = field(default_factory=list)
    correlation: Correlation | None = None
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
            raise OpenFEAPIError(
                f"{campaign.manifest.name}: a plain MD campaign produces no free energy to "
                "analyze. Metrics over its trajectories are a later milestone; its files are "
                "listed by 'gather'."
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


def _write_metrics(outcome: Correlation, path: Path) -> Path:
    """Write the metrics as JSON, so a caller can read them without parsing a figure."""
    payload = {
        "paired": len(outcome.paired),
        "predicted_only": len(outcome.predicted_only),
        "unmatched": outcome.unmatched,
        "aligned": outcome.aligned,
        "shift_kcal_per_mol": outcome.shift,
        "pearson": outcome.pearson,
        "spearman": outcome.spearman,
        "kendall": outcome.kendall,
        "mue_kcal_per_mol": outcome.mue,
        "rmse_kcal_per_mol": outcome.rmse,
        "provenance": collect_provenance(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def analyze_campaign(
    campaign: Campaign,
    measurements: Path | dict[str, Measurement] | None = None,
    dark: bool = False,
) -> AnalysisOutcome:
    """Read a campaign's results and write its analysis.

    Args:
        campaign: The campaign to analyze.
        measurements: A table of measured free energies, or the measurements themselves.
            Without them the analysis reports the predicted ranking alone.
        dark: Whether to draw the figures for a dark surface.

    Returns:
        What was written, and the metrics behind it.

    Raises:
        OpenFEAPIError: If the protocol produces no free energy to analyze.
        InputValidationError: If nothing has been gathered, or the measured table is unusable.
    """
    predictions, relative = read_predictions(campaign)
    directory = campaign.directory / ANALYSIS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    theme = theme_for(dark)

    outcome = AnalysisOutcome(
        directory=directory, protocol=campaign.manifest.protocol, predictions=predictions
    )
    name = campaign.manifest.name

    outcome.tables["predictions"] = _write_predictions(predictions, directory / PREDICTIONS_TABLE)
    outcome.figures["ranking"] = ranking_figure(
        predictions, f"{name} · predicted ranking", theme, directory / RANKING_FIGURE
    )

    if measurements is None:
        logger.info("Analyzed '%s': %d ligand(s), no measured values", name, len(predictions))
        return outcome

    table = read_measurements(measurements) if isinstance(measurements, Path) else measurements
    correlation = correlate(predictions, table, aligned=relative)
    outcome.correlation = correlation
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

    outcome.tables["correlation"] = _write_correlation(correlation, directory / CORRELATION_TABLE)
    outcome.tables["metrics"] = _write_metrics(correlation, directory / METRICS_FILE)
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
