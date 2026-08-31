from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from decisionlab.app.metadata import calculate_feature_reference, load_app_metadata


def _write_features(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["row_index", "problem", "a", "b"])
        writer.writeheader()
        writer.writerows(
            [
                {"row_index": 0, "problem": 10, "a": -1.0, "b": 2.0},
                {"row_index": 1, "problem": 11, "a": 3.0, "b": 4.0},
                {"row_index": 2, "problem": 12, "a": 2.0, "b": 8.0},
            ]
        )


def test_feature_reference_contains_ordered_training_ranges(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)

    reference = calculate_feature_reference(path, ["a", "b"])

    assert reference == {
        "a": {"min": -1.0, "median": 2.0, "max": 3.0},
        "b": {"min": 2.0, "median": 4.0, "max": 8.0},
    }


def test_feature_reference_rejects_misaligned_rows(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    _write_features(path)
    text = path.read_text(encoding="utf-8").replace("1,11", "3,11")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="row_index"):
        calculate_feature_reference(path, ["a", "b"])


def test_application_metadata_enforces_feature_order_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema": "decisionlab_application_metadata_v1",
                "feature_reference": {
                    "a": {"min": -1.0, "median": 0.0, "max": 2.0},
                    "b": {"min": 0.0, "median": 1.0, "max": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_app_metadata(path, ["a", "b"])
    assert list(loaded["feature_reference"]) == ["a", "b"]

    with pytest.raises(ValueError, match="feature order"):
        load_app_metadata(path, ["b", "a"])

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["feature_reference"]["a"] = {"min": 2.0, "median": 0.0, "max": 1.0}
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="not ordered"):
        load_app_metadata(path, ["a", "b"])
