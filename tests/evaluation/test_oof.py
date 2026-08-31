from __future__ import annotations

import csv
from pathlib import Path

import pytest

from decisionlab.evaluation.oof import read_complete_outer_oof_predictions


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_oof_reader_requires_complete_unique_row_coverage(tmp_path: Path) -> None:
    path = tmp_path / "oof.csv"
    rows = [
        {
            "row_index": index,
            "structural_fingerprint": f"g{index}",
            "outer_fold": index % 2,
            "observed_bRate": 0.5,
            "participant_count_n": 20,
            "selected_procedure": 0.4,
        }
        for index in range(4)
    ]
    _write(path, rows)

    values = read_complete_outer_oof_predictions(path, expected_rows=4)

    assert values["row_index"].tolist() == [0, 1, 2, 3]


def test_oof_reader_rejects_structural_group_across_folds(tmp_path: Path) -> None:
    path = tmp_path / "oof.csv"
    _write(
        path,
        [
            {
                "row_index": 0,
                "structural_fingerprint": "same",
                "outer_fold": 0,
                "observed_bRate": 0.5,
                "participant_count_n": 20,
                "selected_procedure": 0.4,
            },
            {
                "row_index": 1,
                "structural_fingerprint": "same",
                "outer_fold": 1,
                "observed_bRate": 0.5,
                "participant_count_n": 20,
                "selected_procedure": 0.4,
            },
        ],
    )

    with pytest.raises(ValueError, match="crosses outer OOF folds"):
        read_complete_outer_oof_predictions(path, expected_rows=2)
