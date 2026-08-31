from __future__ import annotations

import pytest

from decisionlab.evaluation.metrics import (
    evaluate_brate_predictions,
    problem_group_mae_interval,
    problem_group_regression_metrics,
    regression_metrics,
)


def test_problem_bootstrap_resamples_group_losses_not_rows() -> None:
    interval = problem_group_mae_interval(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        ["paired", "paired", "singleton"],
        repeats=500,
        confidence_level=0.95,
        random_seed=7,
    )

    assert interval["resampling_unit"] == "structural_problem_group"
    assert interval["lower"] <= 0.5 <= interval["upper"]


def test_unweighted_metric_formulas() -> None:
    metrics = regression_metrics([0.0, 1.0], [0.25, 0.5])

    assert metrics["mae"] == pytest.approx(0.375)
    assert metrics["rmse"] == pytest.approx(0.15625**0.5)
    assert metrics["r2"] == pytest.approx(0.375)
    assert metrics["mean_bias"] == pytest.approx(-0.125)


def test_participant_count_weighted_metric_formulas() -> None:
    metrics = evaluate_brate_predictions([0.0, 1.0], [0.25, 0.5], ["a", "b"], [1, 3])
    weighted = metrics["participant_count_weighted"]

    assert weighted["mae"] == pytest.approx(0.4375)
    assert weighted["rmse"] == pytest.approx(0.203125**0.5)
    assert weighted["r2"] == pytest.approx(-1.0 / 12.0)
    assert weighted["mean_bias"] == pytest.approx(-0.3125)


def test_problem_with_two_rows_has_same_total_primary_weight_as_singleton() -> None:
    metrics = evaluate_brate_predictions(
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        ["paired", "paired", "singleton"],
        [1, 1, 1],
    )

    assert metrics["problem_group_equal_weighted"]["mae"] == pytest.approx(0.5)
    assert metrics["condition_row_unweighted"]["mae"] == pytest.approx(1.0 / 3.0)


def test_duplicating_a_condition_row_does_not_change_problem_group_mae() -> None:
    original = problem_group_regression_metrics([0.0, 1.0], [0.4, 0.8], ["a", "b"])
    duplicated = problem_group_regression_metrics([0.0, 0.0, 1.0], [0.4, 0.4, 0.8], ["a", "a", "b"])

    assert original["mae"] == pytest.approx(0.3)
    assert duplicated["mae"] == pytest.approx(original["mae"])


def test_problem_group_metrics_reject_malformed_groups() -> None:
    with pytest.raises(ValueError, match="target length"):
        problem_group_regression_metrics([0.0, 1.0], [0.1, 0.9], ["only-one"])
    with pytest.raises(ValueError, match="nonempty strings"):
        problem_group_regression_metrics([0.0, 1.0], [0.1, 0.9], ["valid", ""])


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
