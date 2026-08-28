from __future__ import annotations

import pytest

from decisionlab.evaluation.metrics import evaluate_brate_predictions, regression_metrics


def test_unweighted_metric_formulas() -> None:
    metrics = regression_metrics([0.0, 1.0], [0.25, 0.5])

    assert metrics["mae"] == pytest.approx(0.375)
    assert metrics["rmse"] == pytest.approx(0.15625**0.5)
    assert metrics["r2"] == pytest.approx(0.375)
    assert metrics["mean_bias"] == pytest.approx(-0.125)


def test_participant_count_weighted_metric_formulas() -> None:
    metrics = evaluate_brate_predictions([0.0, 1.0], [0.25, 0.5], [1, 3])
    weighted = metrics["participant_count_weighted"]

    assert weighted["mae"] == pytest.approx(0.4375)
    assert weighted["rmse"] == pytest.approx(0.203125**0.5)
    assert weighted["r2"] == pytest.approx(-1.0 / 12.0)
    assert weighted["mean_bias"] == pytest.approx(-0.3125)


@pytest.mark.parametrize("prediction", [[-0.01, 0.5], [0.5, 1.01]])
def test_metrics_reject_out_of_range_predictions(prediction: list[float]) -> None:
    with pytest.raises(ValueError, match="never implicit"):
        regression_metrics([0.0, 1.0], prediction)


def test_metrics_reject_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        regression_metrics([0.0, 1.0], [0.1, 0.9], [1, 0])


def test_r2_rejects_constant_observed_values() -> None:
    with pytest.raises(ValueError, match="undefined"):
        regression_metrics([0.5, 0.5], [0.4, 0.6])
