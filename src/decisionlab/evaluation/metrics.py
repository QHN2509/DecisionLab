"""Prespecified metrics for aggregate bRate prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    include_r2: bool = True,
) -> dict[str, float]:
    """Compute MAE, RMSE, R², and signed bias with optional positive weights."""
    y_true = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("Observed and predicted values must be equal-length nonempty vectors")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Observed and predicted values must be finite")
    if np.any((y_true < 0.0) | (y_true > 1.0)):
        raise ValueError("Observed bRate values must be in [0, 1]")
    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted bRate values must be in [0, 1]; clipping is never implicit")

    if weights is None:
        metric_weights = np.ones(y_true.size, dtype=float)
    else:
        metric_weights = np.asarray(weights, dtype=float)
        if metric_weights.shape != y_true.shape:
            raise ValueError("Weights must match the target shape")
        if not np.all(np.isfinite(metric_weights)) or np.any(metric_weights <= 0.0):
            raise ValueError("Weights must be finite and strictly positive")
    metric_weights = metric_weights / np.sum(metric_weights)
    residual = y_pred - y_true
    absolute_error = np.abs(residual)
    squared_error = np.square(residual)
    target_mean = float(np.sum(metric_weights * y_true))
    result = {
        "mae": float(np.sum(metric_weights * absolute_error)),
        "rmse": float(np.sqrt(np.sum(metric_weights * squared_error))),
        "mean_bias": float(np.sum(metric_weights * residual)),
    }
    if include_r2:
        denominator = float(np.sum(metric_weights * np.square(y_true - target_mean)))
        if denominator == 0.0:
            raise ValueError("R² is undefined when all observed values are identical")
        result["r2"] = 1.0 - float(np.sum(metric_weights * squared_error)) / denominator
    return result


def problem_group_regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    structural_groups: Sequence[str],
    *,
    include_r2: bool = True,
) -> dict[str, float]:
    """Compute metrics while giving every structural problem equal total weight."""
    groups = np.asarray(structural_groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("Structural groups must be a nonempty vector")
    if groups.size != len(observed):
        raise ValueError("Structural groups must match the target length")
    if any(not isinstance(group, str) or not group for group in groups.tolist()):
        raise ValueError("Structural groups must be nonempty strings")

    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    group_count = counts.size
    row_weights = 1.0 / (group_count * counts[inverse])
    return regression_metrics(observed, predicted, row_weights, include_r2=include_r2)


def evaluate_brate_predictions(
    observed: Sequence[float],
    predicted: Sequence[float],
    structural_groups: Sequence[str],
    participant_counts: Sequence[int],
) -> dict[str, Any]:
    """Return primary equal-problem metrics and explicitly secondary row metrics."""
    return {
        "problem_group_equal_weighted": problem_group_regression_metrics(
            observed, predicted, structural_groups
        ),
        "condition_row_unweighted": regression_metrics(observed, predicted),
        "participant_count_weighted": regression_metrics(
            observed,
            predicted,
            participant_counts,
        ),
    }


def problem_group_mae_interval(
    observed: Sequence[float],
    predicted: Sequence[float],
    structural_groups: Sequence[str],
    *,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    """Bootstrap equal-problem MAE by resampling structural problems equally."""
    target = np.asarray(observed, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    groups = np.asarray(structural_groups)
    if not (target.size == estimates.size == groups.size) or target.size == 0:
        raise ValueError("Grouped-bootstrap inputs must be aligned and nonempty")
    if repeats < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Grouped-bootstrap settings are invalid")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    group_loss = np.bincount(inverse, weights=np.abs(estimates - target)) / counts
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, group_loss.size, size=(repeats, group_loss.size))
    values = np.mean(group_loss[sampled], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "confidence_level": confidence_level,
        "lower": float(np.quantile(values, tail)),
        "upper": float(np.quantile(values, 1.0 - tail)),
        "resampling_unit": "structural_problem_group",
    }
