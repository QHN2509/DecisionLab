from __future__ import annotations

import numpy as np

from decisionlab.experiments.model_selection import (
    candidate_parameter_sets,
    grouped_cv_splits,
    select_model,
)


def test_grouped_cv_has_no_group_overlap_and_shared_row_coverage() -> None:
    groups = np.asarray([f"group-{index // 2}" for index in range(40)])

    splits = grouped_cv_splits(groups, folds=5, random_seed=42)

    validation_rows: list[int] = []
    for train_indices, fold_indices in splits:
        assert not (set(groups[train_indices]) & set(groups[fold_indices]))
        validation_rows.extend(fold_indices.tolist())
    assert sorted(validation_rows) == list(range(groups.size))


def test_parameter_grid_is_declared_not_generated_from_targets() -> None:
    config = {
        "random_forest": {
            "fixed": {"n_estimators": 10},
            "grid": {"max_depth": [3, 5], "min_samples_leaf": [2, 4]},
        }
    }

    values = candidate_parameter_sets(config, "random_forest")

    assert len(values) == 4
    assert all(value["n_estimators"] == 10 for value in values)


def test_selection_prefers_lower_complexity_inside_performance_tolerance() -> None:
    config = {
        "selection": {
            "validation_mae_tolerance": 0.005,
            "rule": "lowest_complexity_within_validation_mae_tolerance_then_cv_mae_then_cv_std",
        },
        "random_forest": {"complexity_rank": 2},
        "gradient_boosting": {"complexity_rank": 3},
    }
    candidates = {
        "random_forest": {
            "validation_metrics": {"unweighted": {"mae": 0.101}},
            "tuning": {"mean_cv_mae": 0.100, "std_cv_mae": 0.004},
        },
        "gradient_boosting": {
            "validation_metrics": {"unweighted": {"mae": 0.098}},
            "tuning": {"mean_cv_mae": 0.097, "std_cv_mae": 0.003},
        },
    }

    selected, rationale = select_model(candidates, config)

    assert selected == "random_forest"
    assert set(rationale["eligible_models"]) == {"random_forest", "gradient_boosting"}


def test_selection_requires_performance_to_be_inside_tolerance() -> None:
    config = {
        "selection": {
            "validation_mae_tolerance": 0.001,
            "rule": "lowest_complexity_within_validation_mae_tolerance_then_cv_mae_then_cv_std",
        },
        "random_forest": {"complexity_rank": 2},
        "gradient_boosting": {"complexity_rank": 3},
    }
    candidates = {
        "random_forest": {
            "validation_metrics": {"unweighted": {"mae": 0.101}},
            "tuning": {"mean_cv_mae": 0.100, "std_cv_mae": 0.004},
        },
        "gradient_boosting": {
            "validation_metrics": {"unweighted": {"mae": 0.095}},
            "tuning": {"mean_cv_mae": 0.097, "std_cv_mae": 0.003},
        },
    }

    selected, _ = select_model(candidates, config)

    assert selected == "gradient_boosting"
