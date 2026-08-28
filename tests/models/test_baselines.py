from __future__ import annotations

import numpy as np
import pytest

from decisionlab.models.baselines import (
    constant_mean_baseline,
    decision_tree_baseline,
    expected_value_heuristic,
    ridge_baseline,
)


def test_constant_baseline_uses_unweighted_training_mean() -> None:
    result = constant_mean_baseline(np.asarray([0.1, 0.4, 1.0]), rows=2)

    assert result.name == "constant_training_mean"
    assert result.predictions == pytest.approx([0.5, 0.5])
    assert result.metadata["constant_value"] == pytest.approx(0.5)


def test_expected_value_rule_handles_better_worse_and_tied_gambles() -> None:
    result = expected_value_heuristic(np.asarray([-2.0, 0.0, 3.0]), tie_prediction=0.5)

    assert result.predictions == pytest.approx([0.0, 0.5, 1.0])
    assert result.metadata["oracle_under_ambiguity"] is True


def test_ridge_predictions_are_explicitly_clipped_and_coefficients_are_named() -> None:
    training_features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    training_target = np.asarray([0.0, 0.2, 0.8, 1.0])
    evaluation_features = np.asarray([[-100.0], [100.0]])

    result = ridge_baseline(
        training_features,
        training_target,
        evaluation_features,
        ["interpretable_feature"],
        alpha=1.0,
        fit_intercept=True,
    )

    assert np.all((result.predictions >= 0.0) & (result.predictions <= 1.0))
    assert result.metadata["prediction_postprocessing"] == "explicit_clip_0_1"
    assert result.metadata["clipped_prediction_count"] == 2
    assert set(result.feature_values) == {"interpretable_feature"}


def test_shallow_tree_is_deterministic_and_bounded() -> None:
    training_features = np.arange(40, dtype=float).reshape(-1, 1)
    training_target = np.linspace(0.0, 1.0, 40)
    evaluation_features = np.asarray([[3.0], [25.0], [39.0]])
    arguments = {
        "feature_names": ["interpretable_feature"],
        "criterion": "squared_error",
        "max_depth": 3,
        "min_samples_leaf": 5,
        "random_seed": 42,
    }

    first = decision_tree_baseline(
        training_features,
        training_target,
        evaluation_features,
        **arguments,
    )
    second = decision_tree_baseline(
        training_features,
        training_target,
        evaluation_features,
        **arguments,
    )

    assert first.predictions == pytest.approx(second.predictions)
    assert np.all((first.predictions >= 0.0) & (first.predictions <= 1.0))
    assert first.metadata["actual_depth"] <= 3
