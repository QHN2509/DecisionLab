"""Contracts for complete nested outer out-of-fold prediction artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_complete_outer_oof_predictions(
    path: Path,
    *,
    expected_rows: int,
    prediction_column: str = "selected_procedure",
) -> dict[str, np.ndarray]:
    """Read an OOF table and reject missing, duplicate, ineligible, or leaking rows."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "row_index",
        "structural_fingerprint",
        "outer_fold",
        "observed_bRate",
        "participant_count_n",
        prediction_column,
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("Nested outer OOF prediction schema is incomplete")
    indices = np.asarray([int(row["row_index"]) for row in rows])
    if rows.__len__() != expected_rows or not np.array_equal(indices, np.arange(expected_rows)):
        raise ValueError("Outer OOF predictions must cover every row exactly once in row order")
    groups = np.asarray([row["structural_fingerprint"] for row in rows])
    folds = np.asarray([int(row["outer_fold"]) for row in rows])
    for group in np.unique(groups):
        if np.unique(folds[groups == group]).size != 1:
            raise ValueError("A structural group crosses outer OOF folds")
    target = np.asarray([float(row["observed_bRate"]) for row in rows])
    predictions = np.asarray([float(row[prediction_column]) for row in rows])
    participant_counts = np.asarray([float(row["participant_count_n"]) for row in rows])
    if not np.all(np.isfinite(predictions)) or np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("Outer OOF predictions must be finite and within [0, 1]")
    return {
        "row_index": indices,
        "groups": groups,
        "outer_fold": folds,
        "target": target,
        "predictions": predictions,
        "participant_counts": participant_counts,
    }
