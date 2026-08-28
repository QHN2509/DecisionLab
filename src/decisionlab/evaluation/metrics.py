"""Prespecified metrics for aggregate bRate prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def regression_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    weights: Sequence[float] | None = None,
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
    denominator = float(np.sum(metric_weights * np.square(y_true - target_mean)))
    if denominator == 0.0:
        raise ValueError("R² is undefined when all observed values are identical")
    r2 = 1.0 - float(np.sum(metric_weights * squared_error)) / denominator
    return {
        "mae": float(np.sum(metric_weights * absolute_error)),
        "rmse": float(np.sqrt(np.sum(metric_weights * squared_error))),
        "r2": r2,
        "mean_bias": float(np.sum(metric_weights * residual)),
    }


def evaluate_brate_predictions(
    observed: Sequence[float], predicted: Sequence[float], participant_counts: Sequence[int]
) -> dict[str, Any]:
    """Return primary problem-level and participant-count-weighted sensitivity metrics."""
    return {
        "unweighted": regression_metrics(observed, predicted),
        "participant_count_weighted": regression_metrics(
            observed,
            predicted,
            participant_counts,
        ),
    }
