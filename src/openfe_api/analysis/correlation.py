"""Comparing predicted free energies against measured ones.

A relative protocol fits its ligands against the network mean, so predictions and
measurements are each centred on their own mean before they are compared. That shifts both
axes by a constant, which leaves every correlation untouched and makes the error metrics
read as the spread around the trend rather than around an arbitrary origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openfe_api.analysis.experimental import Measurement
from openfe_api.log import get_logger

__all__ = ["Correlation", "PairedPoint", "Prediction", "correlate"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One ligand's calculated free energy.

    Attributes:
        name: Ligand name.
        dg: Calculated free energy in kcal/mol.
        uncertainty: Its uncertainty in kcal/mol.
    """

    name: str
    dg: float
    uncertainty: float


@dataclass(frozen=True)
class PairedPoint:
    """One ligand that has both a prediction and a measurement.

    Attributes:
        name: Ligand name.
        predicted: Calculated free energy in kcal/mol, centred for a relative protocol.
        predicted_uncertainty: Its uncertainty in kcal/mol.
        measured: Measured free energy in kcal/mol, centred the same way.
        measured_uncertainty: Its uncertainty in kcal/mol, when the table carried one.
    """

    name: str
    predicted: float
    predicted_uncertainty: float
    measured: float
    measured_uncertainty: float | None


@dataclass
class Correlation:
    """How a campaign's predictions compare with measurement.

    Attributes:
        paired: Ligands carrying both a prediction and a measurement.
        predicted_only: Predictions with nothing to compare against.
        unmatched: Measured ligand names the campaign does not contain.
        centered: Whether both axes were shifted onto their own mean.
        pearson: Pearson correlation of the paired values.
        spearman: Spearman rank correlation.
        kendall: Kendall rank correlation.
        mue: Mean unsigned error in kcal/mol.
        rmse: Root mean square error in kcal/mol.
    """

    paired: list[PairedPoint] = field(default_factory=list)
    predicted_only: list[Prediction] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    centered: bool = False
    pearson: float | None = None
    spearman: float | None = None
    kendall: float | None = None
    mue: float | None = None
    rmse: float | None = None


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return the Pearson correlation, or None when either side does not vary."""
    if first.size < 2 or first.std() == 0 or second.std() == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks, so ties do not bias the rank correlation."""
    order = values.argsort()
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _kendall(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return Kendall's tau-b.

    Counted pair by pair rather than through scipy: a campaign holds tens of ligands, so
    the quadratic count is immediate and the dependency stays out of the runtime.
    """
    size = first.size
    if size < 2:
        return None
    concordant = discordant = first_ties = second_ties = 0
    for i in range(size):
        for j in range(i + 1, size):
            a = first[i] - first[j]
            b = second[i] - second[j]
            if a == 0 and b == 0:
                first_ties += 1
                second_ties += 1
            elif a == 0:
                first_ties += 1
            elif b == 0:
                second_ties += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1

    pairs = size * (size - 1) / 2
    denominator = np.sqrt((pairs - first_ties) * (pairs - second_ties))
    if denominator == 0:
        return None
    return float((concordant - discordant) / denominator)


def correlate(
    predictions: list[Prediction],
    measurements: dict[str, Measurement],
    centered: bool,
) -> Correlation:
    """Pair predictions with measurements and score the agreement.

    Args:
        predictions: The campaign's calculated free energies.
        measurements: Measured free energies, keyed by ligand name.
        centered: Whether to shift both axes onto their own mean, which a relative
            protocol needs because its values are relative to the network already.

    Returns:
        The paired points, whatever could not be paired, and the metrics. Every metric is
        None when fewer than two ligands pair up.
    """
    by_name = {prediction.name: prediction for prediction in predictions}
    shared = [name for name in by_name if name in measurements]

    outcome = Correlation(
        predicted_only=[by_name[name] for name in by_name if name not in measurements],
        unmatched=sorted(name for name in measurements if name not in by_name),
        centered=centered,
    )
    if outcome.unmatched:
        logger.warning(
            "%d measured ligand(s) are not in the campaign: %s",
            len(outcome.unmatched),
            ", ".join(outcome.unmatched),
        )
    if not shared:
        return outcome

    predicted = np.array([by_name[name].dg for name in shared], dtype=float)
    measured = np.array([measurements[name].dg for name in shared], dtype=float)
    if centered:
        # The unpaired predictions move by the same offset, so every predicted value stays
        # on one axis and a ligand without a measurement is still placed against the rest.
        shift = float(predicted.mean())
        predicted = predicted - shift
        measured = measured - measured.mean()
        outcome.predicted_only = [
            Prediction(entry.name, entry.dg - shift, entry.uncertainty)
            for entry in outcome.predicted_only
        ]

    outcome.paired = [
        PairedPoint(
            name=name,
            predicted=float(predicted[index]),
            predicted_uncertainty=by_name[name].uncertainty,
            measured=float(measured[index]),
            measured_uncertainty=measurements[name].uncertainty,
        )
        for index, name in enumerate(shared)
    ]

    if len(shared) >= 2:
        difference = predicted - measured
        outcome.pearson = _pearson(predicted, measured)
        outcome.spearman = _pearson(_ranks(predicted), _ranks(measured))
        outcome.kendall = _kendall(predicted, measured)
        outcome.mue = float(np.abs(difference).mean())
        outcome.rmse = float(np.sqrt((difference**2).mean()))

    return outcome
