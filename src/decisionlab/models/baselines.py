"""Fixed, untuned baselines for aggregate bRate prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True, slots=True)
class BaselinePredictions:
    """Predictions and auditable model metadata for one baseline."""

    name: str
    predictions: np.ndarray
    metadata: dict[str, Any]
    feature_values: dict[str, float]


def constant_mean_baseline(training_target: np.ndarray, rows: int) -> BaselinePredictions:
    """Predict the unweighted training-set target mean for every evaluation row."""
    if training_target.ndim != 1 or training_target.size == 0 or rows <= 0:
        raise ValueError("Constant baseline requires a nonempty training target and output rows")
    value = float(np.mean(training_target))
    return BaselinePredictions(
        name="constant_training_mean",
        predictions=np.full(rows, value, dtype=float),
        metadata={
            "fit_rows": int(training_target.size),
            "constant_value": value,
            "training_weighting": "unweighted",
            "prediction_postprocessing": "none",
        },
        feature_values={},
    )


def expected_value_heuristic(
    expected_value_difference: np.ndarray, tie_prediction: float = 0.5
) -> BaselinePredictions:
    """Predict B for positive EV advantage, A for negative advantage, and tie otherwise."""
    if expected_value_difference.ndim != 1 or expected_value_difference.size == 0:
        raise ValueError("EV heuristic requires a nonempty one-dimensional EV difference")
    if not 0.0 <= tie_prediction <= 1.0:
        raise ValueError("Tie prediction must be in [0, 1]")
    predictions = np.where(
        expected_value_difference > 0.0,
        1.0,
        np.where(expected_value_difference < 0.0, 0.0, tie_prediction),
    )
    return BaselinePredictions(
        name="expected_value_hard_rule_oracle",
        predictions=predictions,
        metadata={
            "fit_rows": 0,
            "rule": "1 if EV_B > EV_A; 0 if EV_B < EV_A; tie_prediction otherwise",
            "tie_prediction": tie_prediction,
            "oracle_under_ambiguity": True,
            "prediction_postprocessing": "none",
        },
        feature_values={},
    )


def ridge_baseline(
    training_features: np.ndarray,
    training_target: np.ndarray,
    evaluation_features: np.ndarray,
    feature_names: list[str],
    *,
    alpha: float,
    fit_intercept: bool,
) -> BaselinePredictions:
    """Fit standardized ridge regression and explicitly clip predictions to bRate bounds."""
    if alpha < 0.0:
        raise ValueError("Ridge alpha must be nonnegative")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept)),
        ]
    )
    pipeline.fit(training_features, training_target)
    raw_predictions = pipeline.predict(evaluation_features)
    predictions = np.clip(raw_predictions, 0.0, 1.0)
    estimator = pipeline.named_steps["ridge"]
    coefficients = {
        name: float(value) for name, value in zip(feature_names, estimator.coef_, strict=True)
    }
    return BaselinePredictions(
        name="ridge_engineered_oracle",
        predictions=predictions,
        metadata={
            "fit_rows": int(training_target.size),
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "standardized_features": True,
            "intercept": float(estimator.intercept_),
            "prediction_postprocessing": "explicit_clip_0_1",
            "raw_prediction_min": float(np.min(raw_predictions)),
            "raw_prediction_max": float(np.max(raw_predictions)),
            "clipped_prediction_count": int(np.sum(raw_predictions != predictions)),
        },
        feature_values=coefficients,
    )


def decision_tree_baseline(
    training_features: np.ndarray,
    training_target: np.ndarray,
    evaluation_features: np.ndarray,
    feature_names: list[str],
    *,
    criterion: str,
    max_depth: int,
    min_samples_leaf: int,
    random_seed: int,
) -> BaselinePredictions:
    """Fit one fixed shallow regression tree without hyperparameter search."""
    estimator = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
    )
    estimator.fit(training_features, training_target)
    predictions = estimator.predict(evaluation_features)
    importances = {
        name: float(value)
        for name, value in zip(feature_names, estimator.feature_importances_, strict=True)
    }
    return BaselinePredictions(
        name="shallow_decision_tree_engineered_oracle",
        predictions=predictions,
        metadata={
            "fit_rows": int(training_target.size),
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_seed": random_seed,
            "actual_depth": int(estimator.get_depth()),
            "leaf_count": int(estimator.get_n_leaves()),
            "prediction_postprocessing": "none",
        },
        feature_values=importances,
    )
