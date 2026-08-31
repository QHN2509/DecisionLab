"""Build the small, provenance-covered metadata bundle used by the Streamlit app."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.experiments.provenance import (
    finalize_run_provenance,
    start_run_provenance,
    validate_upstream_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SELECTION_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "nested_model_selection"
DEFAULT_MODEL_METRICS = MODEL_SELECTION_DIR / "metrics.json"
DEFAULT_MODEL_FEATURE_NAMES = MODEL_SELECTION_DIR / "feature_names.json"
DEFAULT_MODEL_PROVENANCE = MODEL_SELECTION_DIR / "provenance.json"
DEFAULT_ENGINEERED_FEATURES = (
    PROJECT_ROOT / "data" / "processed" / "choices13k_engineered_features.csv"
)
DEFAULT_APP_CONFIG = PROJECT_ROOT / "configs" / "app.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "application"
DEFAULT_METADATA = DEFAULT_OUTPUT_DIR / "metadata.json"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "provenance.json"


def calculate_feature_reference(
    path: Path, feature_names: list[str]
) -> dict[str, dict[str, float]]:
    """Calculate finite training-feature ranges from the provenance-verified feature table."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected_columns = {"row_index", "problem", *feature_names}
    if not rows or set(rows[0]) != expected_columns:
        raise ValueError("Engineered feature table does not match the model feature contract")
    if [int(row["row_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Engineered feature rows are not aligned by row_index")
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Engineered feature reference contains non-finite values")
    return {
        name: {
            "min": float(np.min(matrix[:, index])),
            "median": float(np.median(matrix[:, index])),
            "max": float(np.max(matrix[:, index])),
        }
        for index, name in enumerate(feature_names)
    }


def load_app_metadata(path: Path, feature_names: list[str]) -> dict[str, Any]:
    """Load and validate the tracked application metadata contract."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "decisionlab_application_metadata_v1":
        raise ValueError("Application metadata has an unsupported schema")
    reference = metadata.get("feature_reference")
    if not isinstance(reference, dict) or list(reference) != feature_names:
        raise ValueError("Application metadata feature order differs from the model")
    for name, values in reference.items():
        if not isinstance(values, dict) or set(values) != {"min", "median", "max"}:
            raise ValueError(f"Application range is incomplete for {name}")
        ordered = [values["min"], values["median"], values["max"]]
        if not all(isinstance(value, int | float) and math.isfinite(value) for value in ordered):
            raise ValueError(f"Application range is non-finite for {name}")
        if not ordered[0] <= ordered[1] <= ordered[2]:
            raise ValueError(f"Application range is not ordered for {name}")
    return metadata


def build_application_metadata(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    app_config_path: Path = DEFAULT_APP_CONFIG,
    output_path: Path = DEFAULT_METADATA,
    provenance_path: Path = DEFAULT_PROVENANCE,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Generate app metadata from official model inputs without fitting a model."""
    provenance = start_run_provenance(
        experiment_name="decisionlab_application_metadata_v1",
        config_paths=[app_config_path],
        configuration_values={
            "metadata_output": str(output_path),
            "model_experiment": str(MODEL_SELECTION_DIR),
            "allow_dirty": allow_dirty,
        },
        dataset_manifest_path=manifest_path,
        raw_dir=raw_dir,
        fold_specification_identifier="nested_grouped_cv_v1_application_metadata",
        entry_module=Path(__file__),
        allow_dirty=allow_dirty,
    )
    upstream_artifacts = [
        DEFAULT_MODEL_METRICS,
        DEFAULT_MODEL_FEATURE_NAMES,
        DEFAULT_ENGINEERED_FEATURES,
    ]
    model_provenance = validate_upstream_artifacts(
        DEFAULT_MODEL_PROVENANCE,
        upstream_artifacts,
    )
    metrics = json.loads(DEFAULT_MODEL_METRICS.read_text(encoding="utf-8"))
    feature_document = json.loads(DEFAULT_MODEL_FEATURE_NAMES.read_text(encoding="utf-8"))
    feature_names = feature_document["feature_names"]
    if feature_names != metrics["feature_names"]:
        raise ValueError("Model metrics and feature-name artifacts disagree")
    metadata = {
        "schema": "decisionlab_application_metadata_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_model_git_sha": model_provenance["git"]["commit_sha"],
        "model_metrics_sha256": sha256_file(DEFAULT_MODEL_METRICS),
        "model_feature_names_sha256": sha256_file(DEFAULT_MODEL_FEATURE_NAMES),
        "engineered_features_sha256": sha256_file(DEFAULT_ENGINEERED_FEATURES),
        "feature_count": len(feature_names),
        "feature_reference": calculate_feature_reference(
            DEFAULT_ENGINEERED_FEATURES,
            feature_names,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    finalize_run_provenance(
        provenance,
        fold_artifacts={},
        input_artifacts=[DEFAULT_MODEL_PROVENANCE, *upstream_artifacts],
        output_artifacts=[output_path],
        output_path=provenance_path,
    )
    return metadata


def main() -> None:
    """Generate the tracked metadata required for data-free app startup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a clearly marked non-official run from a dirty worktree.",
    )
    args = parser.parse_args()
    metadata = build_application_metadata(
        args.raw_dir,
        args.manifest,
        args.config,
        args.output,
        args.provenance,
        allow_dirty=args.allow_dirty,
    )
    print(f"Application metadata complete: features={metadata['feature_count']}")


if __name__ == "__main__":
    main()
