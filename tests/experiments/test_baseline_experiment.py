from __future__ import annotations

import numpy as np
import pytest

from decisionlab.experiments.baselines import (
    MODEL_ORDER,
    audited_feature_names,
    partition_indices,
    run_fixed_baselines,
)


def test_audited_feature_names_rejects_target_columns() -> None:
    summary = {
        "engineered_feature_columns": ["expected_value_a", "bRate"],
        "leakage_audit": {"status": "PASS", "features": {}},
    }

    with pytest.raises(ValueError, match="Prohibited"):
        audited_feature_names(summary)


def test_audited_feature_names_requires_complete_passing_audit() -> None:
    summary = {
        "engineered_feature_columns": ["expected_value_a"],
        "leakage_audit": {
            "status": "PASS",
            "features": {
                "expected_value_a": {
                    "status": "PASS",
                    "target_derived": False,
                }
            },
        },
    }

    assert audited_feature_names(summary) == ["expected_value_a"]


def test_partition_indices_returns_no_test_rows() -> None:
    assignments = [
        {"row_index": "0", "split": "train"},
        {"row_index": "1", "split": "validation"},
        {"row_index": "2", "split": "test"},
    ]

    train, validation = partition_indices(assignments)

    assert train.tolist() == [0]
    assert validation.tolist() == [1]
    assert 2 not in train and 2 not in validation


def test_fixed_baselines_return_prespecified_order_and_bounded_predictions() -> None:
    feature_names = [
        "expected_value_difference_b_minus_a_oracle",
        "payoff_range_a",
    ]
    training_features = np.asarray([[-2.0, 1.0], [-1.0, 2.0], [0.5, 3.0], [2.0, 4.0], [3.0, 5.0]])
    training_target = np.asarray([0.1, 0.2, 0.6, 0.8, 0.9])
    validation_features = np.asarray([[-0.5, 2.0], [1.5, 4.0]])
    config = {
        "random_seed": 42,
        "ridge": {"alpha": 1.0, "fit_intercept": True},
        "decision_tree": {
            "criterion": "squared_error",
            "max_depth": 6,
            "min_samples_leaf": 2,
        },
        "expected_value_heuristic": {"tie_prediction": 0.5},
    }

    results = run_fixed_baselines(
        training_features,
        training_target,
        validation_features,
        feature_names,
        config,
    )

    assert tuple(result.name for result in results) == MODEL_ORDER
    assert all(result.predictions.shape == (2,) for result in results)
    assert all(
        np.all((result.predictions >= 0.0) & (result.predictions <= 1.0)) for result in results
    )
