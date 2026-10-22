"""Tests for scoring predictions against measured free energies."""

from __future__ import annotations

import numpy as np
import pytest

from openfe_api.analysis.correlation import Prediction, correlate
from openfe_api.analysis.experimental import Measurement


def _predictions(values: dict[str, float]) -> list[Prediction]:
    """Build predictions with a fixed uncertainty."""
    return [Prediction(name=name, dg=dg, uncertainty=0.2) for name, dg in values.items()]


def _measurements(values: dict[str, float]) -> dict[str, Measurement]:
    """Build measurements with no uncertainty."""
    return {name: Measurement(name=name, dg=dg) for name, dg in values.items()}


def test_perfect_agreement_scores_one() -> None:
    values = {"a": -9.0, "b": -8.0, "c": -7.0}

    outcome = correlate(_predictions(values), _measurements(values), centered=False)

    assert outcome.pearson == pytest.approx(1.0)
    assert outcome.spearman == pytest.approx(1.0)
    assert outcome.kendall == pytest.approx(1.0)
    assert outcome.mue == pytest.approx(0.0)
    assert outcome.rmse == pytest.approx(0.0)


def test_reversed_ranking_scores_minus_one() -> None:
    predictions = _predictions({"a": -9.0, "b": -8.0, "c": -7.0})
    measurements = _measurements({"a": -7.0, "b": -8.0, "c": -9.0})

    outcome = correlate(predictions, measurements, centered=False)

    assert outcome.pearson == pytest.approx(-1.0)
    assert outcome.kendall == pytest.approx(-1.0)


def test_centering_leaves_correlation_alone_but_moves_the_error() -> None:
    """A relative protocol's values carry an arbitrary offset, which centering removes."""
    predictions = _predictions({"a": 1.0, "b": 2.0, "c": 3.0})
    measurements = _measurements({"a": -9.0, "b": -8.0, "c": -7.0})

    raw = correlate(predictions, measurements, centered=False)
    centered = correlate(predictions, measurements, centered=True)

    assert raw.pearson == pytest.approx(centered.pearson)
    assert raw.mue == pytest.approx(10.0)
    assert centered.mue == pytest.approx(0.0)


def test_ligands_without_a_measurement_are_kept_apart() -> None:
    predictions = _predictions({"a": -9.0, "b": -8.0, "c": -7.0})
    measurements = _measurements({"a": -9.0, "b": -8.0})

    outcome = correlate(predictions, measurements, centered=False)

    assert [point.name for point in outcome.paired] == ["a", "b"]
    assert [entry.name for entry in outcome.predicted_only] == ["c"]


def test_a_measured_ligand_the_campaign_lacks_is_reported() -> None:
    outcome = correlate(
        _predictions({"a": -9.0}), _measurements({"a": -9.0, "ghost": -8.0}), centered=False
    )

    assert outcome.unmatched == ["ghost"]


def test_a_single_pair_reports_no_metrics() -> None:
    """One point has no spread to correlate, so a number there would be meaningless."""
    outcome = correlate(_predictions({"a": -9.0}), _measurements({"a": -9.0}), centered=False)

    assert len(outcome.paired) == 1
    assert outcome.pearson is None
    assert outcome.kendall is None


def test_nothing_in_common_reports_no_pairs() -> None:
    outcome = correlate(_predictions({"a": -9.0}), _measurements({"b": -8.0}), centered=False)

    assert outcome.paired == []
    assert outcome.pearson is None


def test_a_flat_prediction_reports_no_correlation() -> None:
    """Dividing by a zero spread would be undefined rather than uncorrelated."""
    predictions = _predictions({"a": -8.0, "b": -8.0, "c": -8.0})

    outcome = correlate(predictions, _measurements({"a": -9.0, "b": -8.0, "c": -7.0}), False)

    assert outcome.pearson is None
    assert outcome.mue is not None


def test_the_statistics_match_scipy() -> None:
    """The coefficients are hand-rolled to keep scipy out of the runtime."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    names = [f"lig_{index}" for index in range(12)]
    predicted = rng.normal(-9.0, 1.5, size=12)
    measured = predicted + rng.normal(0.0, 0.8, size=12)

    outcome = correlate(
        [Prediction(name, float(dg), 0.2) for name, dg in zip(names, predicted, strict=True)],
        {name: Measurement(name, float(dg)) for name, dg in zip(names, measured, strict=True)},
        centered=False,
    )

    assert outcome.pearson == pytest.approx(scipy_stats.pearsonr(predicted, measured)[0])
    assert outcome.spearman == pytest.approx(scipy_stats.spearmanr(predicted, measured)[0])
    assert outcome.kendall == pytest.approx(scipy_stats.kendalltau(predicted, measured)[0])


def test_the_statistics_match_scipy_with_ties() -> None:
    """Ties are where a rank correlation is easiest to get wrong."""
    scipy_stats = pytest.importorskip("scipy.stats")
    predicted = np.array([-9.0, -9.0, -8.0, -8.0, -7.0])
    measured = np.array([-9.5, -8.5, -8.5, -7.5, -7.0])
    names = [f"lig_{index}" for index in range(5)]

    outcome = correlate(
        [Prediction(name, float(dg), 0.2) for name, dg in zip(names, predicted, strict=True)],
        {name: Measurement(name, float(dg)) for name, dg in zip(names, measured, strict=True)},
        centered=False,
    )

    assert outcome.spearman == pytest.approx(scipy_stats.spearmanr(predicted, measured)[0])
    assert outcome.kendall == pytest.approx(scipy_stats.kendalltau(predicted, measured)[0])
