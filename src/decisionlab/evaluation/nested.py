"""Persist and audit DecisionLab's nested structural-group CV assignments."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_and_validate, load_problems, load_selections
from decisionlab.evaluation.splitting import (
    InnerFoldAssignment,
    OuterFoldAssignment,
    audit_nested_fold_assignments,
    create_nested_fold_assignments,
    structural_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evaluation.json"
DEFAULT_OUTER_ASSIGNMENTS = PROJECT_ROOT / "data" / "processed" / "nested_outer_folds.csv"
DEFAULT_INNER_ASSIGNMENTS = PROJECT_ROOT / "data" / "processed" / "nested_inner_folds.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts" / "manifests" / "nested_cv_summary.json"


def load_nested_evaluation_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the strict nested-CV evaluation contract."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_name",
        "dataset_role",
        "outer_folds",
        "inner_folds",
        "outer_seed",
        "inner_seed",
        "bootstrap_repeats",
        "confidence_level",
        "primary_metric",
        "secondary_metrics",
        "weighted_sensitivity_metrics",
        "historical_partitions_role",
    }
    if set(config) != required:
        raise ValueError(f"Nested evaluation config must define exactly: {sorted(required)}")
    if config["protocol_name"] != "nested_grouped_cv_v1":
        raise ValueError("Unexpected nested evaluation protocol")
    if config["dataset_role"] != "development":
        raise ValueError("choices13k must be classified as development data")
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds")
    if type(config["outer_seed"]) is not int or type(config["inner_seed"]) is not int:
        raise ValueError("Nested CV seeds must be integers")
    if config["bootstrap_repeats"] < 100 or not 0.0 < config["confidence_level"] < 1.0:
        raise ValueError("Grouped-bootstrap settings are invalid")
    if config["primary_metric"] != "problem_group_equal_weighted_mae":
        raise ValueError("Primary metric must weight structural problems equally")
    if config["historical_partitions_role"] != "development_only_not_confirmatory":
        raise ValueError("Historical partitions cannot be classified as confirmatory")
    return config


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_outer_assignments(path: Path, expected_rows: int) -> list[OuterFoldAssignment]:
    """Read and validate row-aligned outer-fold assignments."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {"row_index", "problem", "structural_fingerprint", "outer_fold"}
    if len(rows) != expected_rows or not rows or set(rows[0]) != expected:
        raise ValueError("Outer assignment table is missing rows or required columns")
    result = [
        OuterFoldAssignment(
            row_index=int(row["row_index"]),
            problem=int(row["problem"]),
            structural_fingerprint=row["structural_fingerprint"],
            outer_fold=int(row["outer_fold"]),
        )
        for row in rows
    ]
    if [row.row_index for row in result] != list(range(expected_rows)):
        raise ValueError("Outer assignments are not aligned by row_index")
    return result


def read_inner_assignments(path: Path) -> list[InnerFoldAssignment]:
    """Read outer-specific inner validation-fold assignments."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {"outer_fold", "row_index", "structural_fingerprint", "inner_fold"}
    if not rows or set(rows[0]) != expected:
        raise ValueError("Inner assignment table is missing rows or required columns")
    return [
        InnerFoldAssignment(
            outer_fold=int(row["outer_fold"]),
            row_index=int(row["row_index"]),
            structural_fingerprint=row["structural_fingerprint"],
            inner_fold=int(row["inner_fold"]),
        )
        for row in rows
    ]


def build_nested_cv_assignments(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    outer_path: Path = DEFAULT_OUTER_ASSIGNMENTS,
    inner_path: Path = DEFAULT_INNER_ASSIGNMENTS,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    """Build deterministic nested folds from validated target-free group identities."""
    validation = load_and_validate(raw_dir=raw_dir, manifest_path=manifest_path)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    config = load_nested_evaluation_config(config_path)
    groups = np.asarray(
        [
            structural_fingerprint(record, problems[str(index)])
            for index, record in enumerate(selections)
        ]
    )
    problem_ids = np.asarray([record.problem for record in selections])
    outer, inner = create_nested_fold_assignments(
        problem_ids,
        groups,
        outer_folds=config["outer_folds"],
        inner_folds=config["inner_folds"],
        outer_seed=config["outer_seed"],
        inner_seed=config["inner_seed"],
    )
    audit = audit_nested_fold_assignments(
        outer,
        inner,
        outer_folds=config["outer_folds"],
        inner_folds=config["inner_folds"],
    )
    _write_csv(
        outer_path,
        ["row_index", "problem", "structural_fingerprint", "outer_fold"],
        [asdict(row) for row in outer],
    )
    _write_csv(
        inner_path,
        ["outer_fold", "row_index", "structural_fingerprint", "inner_fold"],
        [asdict(row) for row in inner],
    )
    summary = {
        "protocol_name": config["protocol_name"],
        "dataset_role": "development",
        "confirmatory_holdout": False,
        "target_values_used_to_create_folds": False,
        "source_commit": validation["source_commit"],
        "source_sha256": validation["sha256"],
        "config": config,
        "audit": audit,
        "outer_fold_rows": dict(sorted(Counter(row.outer_fold for row in outer).items())),
        "outer_fold_groups": {
            str(fold): len({row.structural_fingerprint for row in outer if row.outer_fold == fold})
            for fold in range(config["outer_folds"])
        },
        "outer_assignments": {
            "path": str(outer_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(outer_path),
        },
        "inner_assignments": {
            "path": str(inner_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(inner_path),
        },
        "historical_partitions": "development_only_not_confirmatory_and_not_used",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Create and audit persisted nested grouped-CV assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build_nested_cv_assignments(args.raw_dir, args.manifest, args.config)
    print(json.dumps(summary["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
