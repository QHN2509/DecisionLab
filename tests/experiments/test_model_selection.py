from __future__ import annotations

import numpy as np
import pytest

from decisionlab.evaluation.splitting import create_nested_fold_assignments
from decisionlab.experiments.model_selection import (
    audited_oracle_feature_names,
    candidate_parameter_sets,
    evaluate_parameter_set,
    grouped_cv_splits,
    run_nested_cv,
    select_model,
)


class _PredictionColumnPipeline:
    def fit(self, features: np.ndarray, target: np.ndarray):
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def test_oracle_feature_metadata_comes_from_leakage_audit() -> None:
    feature_names = ["visible", "oracle"]
    summary = {
        "leakage_audit": {
            "features": {
                "visible": {"availability": "participant-visible"},
                "oracle": {"availability": "oracle under ambiguity"},
            }
        }
    }

    assert audited_oracle_feature_names(summary, feature_names) == ["oracle"]


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


def test_cv_parameter_scoring_uses_equal_problem_weighting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "decisionlab.experiments.model_selection.build_candidate_pipeline",
        lambda *args, **kwargs: _PredictionColumnPipeline(),
    )
    features = np.asarray([[0.0], [1.0], [1.0], [0.0], [1.0]])
    target = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0])
    groups = np.asarray(["paired", "paired", "singleton-error", "c", "d"])
    splits = [
        (np.asarray([3, 4]), np.asarray([0, 1, 2])),
        (np.asarray([0, 1, 2]), np.asarray([3, 4])),
    ]

    result = evaluate_parameter_set(
        "random_forest",
        {},
        features,
        target,
        groups,
        splits,
        random_seed=42,
    )

    assert result["mean_cv_mae"] == pytest.approx(0.25)
    assert result["mean_cv_mae"] != pytest.approx(0.2)


def test_selection_prefers_lower_complexity_inside_performance_tolerance() -> None:
    config = {
        "selection": {
            "inner_oof_mae_tolerance": 0.005,
            "rule": "lowest_complexity_within_inner_oof_mae_tolerance_then_inner_oof_mae",
        },
        "random_forest": {"complexity_rank": 2},
        "gradient_boosting": {"complexity_rank": 3},
    }
    candidates = {
        "random_forest": {"mean_cv_mae": 0.101, "std_cv_mae": 0.004},
        "gradient_boosting": {"mean_cv_mae": 0.098, "std_cv_mae": 0.003},
    }

    selected, rationale = select_model(candidates, config)

    assert selected == "random_forest"
    assert set(rationale["eligible_models"]) == {"random_forest", "gradient_boosting"}


def test_selection_requires_performance_to_be_inside_tolerance() -> None:
    config = {
        "selection": {
            "inner_oof_mae_tolerance": 0.001,
            "rule": "lowest_complexity_within_inner_oof_mae_tolerance_then_inner_oof_mae",
        },
        "random_forest": {"complexity_rank": 2},
        "gradient_boosting": {"complexity_rank": 3},
    }
    candidates = {
        "random_forest": {"mean_cv_mae": 0.101, "std_cv_mae": 0.004},
        "gradient_boosting": {"mean_cv_mae": 0.095, "std_cv_mae": 0.003},
    }

    selected, _ = select_model(candidates, config)

    assert selected == "gradient_boosting"


class _MeanPipeline:
    fit_targets: list[np.ndarray] = []

    def fit(self, features: np.ndarray, target: np.ndarray):
        self.mean = float(np.mean(target))
        self.fit_targets.append(target.copy())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def _tiny_nested_config() -> dict:
    family = {
        "fixed": {},
        "grid": {},
        "complexity_rank": 1,
        "interpretability": "test",
    }
    return {
        "random_seed": 10,
        "nested_cross_validation": {
            "outer_folds": 2,
            "inner_folds": 2,
            "outer_seed": 1,
            "inner_seed": 2,
            "primary_metric": "problem_group_equal_weighted_mae",
        },
        "selection": {"inner_oof_mae_tolerance": 0.0, "rule": "test"},
        "random_forest": family,
        "gradient_boosting": family | {"complexity_rank": 2},
    }


def test_nested_cv_covers_all_rows_and_never_fits_outer_test_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = np.asarray([f"g{index // 2}" for index in range(16)])
    problems = np.arange(16)
    outer, inner = create_nested_fold_assignments(
        problems, groups, outer_folds=2, inner_folds=2, outer_seed=1, inner_seed=2
    )
    outer_fold = np.asarray([row.outer_fold for row in outer])
    target = np.linspace(0.0, 1.0, 16)
    _MeanPipeline.fit_targets = []
    monkeypatch.setattr(
        "decisionlab.experiments.model_selection.build_candidate_pipeline",
        lambda *args, **kwargs: _MeanPipeline(),
    )

    result = run_nested_cv(
        np.arange(16, dtype=float).reshape(-1, 1),
        target,
        groups,
        np.ones(16),
        outer_fold,
        inner,
        _tiny_nested_config(),
    )

    assert np.isfinite(result["selected_oof_predictions"]).all()
    for fold in range(2):
        outer_test_targets = set(target[outer_fold == fold])
        outer_train_targets = set(target[outer_fold != fold])
        relevant_fits = [
            set(values)
            for values in _MeanPipeline.fit_targets
            if set(values) <= outer_train_targets
        ]
        assert relevant_fits
        assert all(not (values & outer_test_targets) for values in relevant_fits)


def test_outer_test_targets_cannot_change_inner_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = np.asarray([f"g{index // 2}" for index in range(16)])
    outer, inner = create_nested_fold_assignments(
        np.arange(16), groups, outer_folds=2, inner_folds=2, outer_seed=1, inner_seed=2
    )
    outer_fold = np.asarray([row.outer_fold for row in outer])
    monkeypatch.setattr(
        "decisionlab.experiments.model_selection.build_candidate_pipeline",
        lambda *args, **kwargs: _MeanPipeline(),
    )
    target = np.linspace(0.1, 0.9, 16)
    first = run_nested_cv(
        np.ones((16, 1)), target, groups, np.ones(16), outer_fold, inner, _tiny_nested_config()
    )
    changed = target.copy()
    changed[outer_fold == 0] = 1.0 - changed[outer_fold == 0]
    second = run_nested_cv(
        np.ones((16, 1)), changed, groups, np.ones(16), outer_fold, inner, _tiny_nested_config()
    )

    assert (
        first["fold_selections"][0]["selected_model"]
        == second["fold_selections"][0]["selected_model"]
    )
    assert (
        first["fold_selections"][0]["selected_parameters"]
        == second["fold_selections"][0]["selected_parameters"]
    )
