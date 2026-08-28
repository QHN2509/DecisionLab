"""Run the fixed validation-only DecisionLab baseline experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_selections
from decisionlab.evaluation.metrics import evaluate_brate_predictions
from decisionlab.evaluation.protocol import (
    DEFAULT_ASSIGNMENTS,
    build_evaluation_protocol,
)
from decisionlab.evaluation.protocol import (
    DEFAULT_CONFIG as DEFAULT_EVALUATION_CONFIG,
)
from decisionlab.features.behavioral import (
    DEFAULT_ENGINEERED_FEATURES,
    build_feature_tables,
)
from decisionlab.features.behavioral import (
    DEFAULT_SUMMARY as DEFAULT_FEATURE_SUMMARY,
)
from decisionlab.models.baselines import (
    BaselinePredictions,
    constant_mean_baseline,
    decision_tree_baseline,
    expected_value_heuristic,
    ridge_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baselines.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "baselines"
DEFAULT_METRICS = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "predictions_validation.csv"
DEFAULT_RIDGE_VALUES = DEFAULT_OUTPUT_DIR / "ridge_coefficients.csv"
DEFAULT_TREE_VALUES = DEFAULT_OUTPUT_DIR / "tree_feature_importances.csv"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "tables" / "baseline_comparison.csv"
DEFAULT_COMPARISON_MD = PROJECT_ROOT / "reports" / "tables" / "baseline_comparison.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "baselines.md"

MODEL_ORDER = (
    "constant_training_mean",
    "expected_value_hard_rule_oracle",
    "ridge_engineered_oracle",
    "shallow_decision_tree_engineered_oracle",
)

FORBIDDEN_FEATURES = {
    "row_index",
    "problem",
    "bRate",
    "bRate_std",
    "brate",
    "brate_std",
    "n",
    "Block",
    "block",
}


def load_baseline_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the fixed, untuned baseline specification."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "experiment_name",
        "evaluation_split",
        "feature_set",
        "random_seed",
        "ridge",
        "decision_tree",
        "expected_value_heuristic",
    }
    if set(config) != required:
        raise ValueError(f"Baseline config must define exactly: {sorted(required)}")
    if config["evaluation_split"] != "validation":
        raise ValueError("Baseline experiments may evaluate only the validation split")
    if config["feature_set"] != "engineered_design_oracle_v1":
        raise ValueError("Unexpected baseline feature set")
    if config["ridge"] != {
        "alpha": 1.0,
        "fit_intercept": True,
        "standardize": True,
        "prediction_postprocessing": "explicit_clip_0_1",
    }:
        raise ValueError("Ridge settings differ from the locked untuned specification")
    expected_tree = {
        "criterion": "squared_error",
        "max_depth": 6,
        "min_samples_leaf": 50,
        "prediction_postprocessing": "none",
    }
    if config["decision_tree"] != expected_tree:
        raise ValueError("Tree settings differ from the locked untuned specification")
    return config


def audited_feature_names(feature_summary: dict[str, Any]) -> list[str]:
    """Return engineered predictors only after explicit target-leakage checks."""
    names = list(feature_summary["engineered_feature_columns"])
    prohibited = sorted(set(names) & FORBIDDEN_FEATURES)
    if prohibited:
        raise ValueError(f"Prohibited feature columns: {prohibited}")
    audit = feature_summary["leakage_audit"]
    if audit["status"] != "PASS" or set(audit["features"]) != set(names):
        raise ValueError("Feature leakage audit is incomplete")
    failed = [
        name
        for name, values in audit["features"].items()
        if values["status"] != "PASS" or values["target_derived"]
    ]
    if failed:
        raise ValueError(f"Feature leakage audit failed: {failed}")
    return names


def _read_engineered_features(path: Path, feature_names: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected_columns = {"row_index", "problem", *feature_names}
    if not rows or set(rows[0]) != expected_columns:
        raise ValueError("Engineered feature table schema differs from the audited contract")
    for expected_index, row in enumerate(rows):
        if int(row["row_index"]) != expected_index:
            raise ValueError("Engineered feature rows are not aligned by row_index")
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Engineered feature matrix contains non-finite values")
    return matrix


def _read_assignments(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected_columns = {"row_index", "problem", "structural_fingerprint", "split"}
    if len(rows) != expected_rows or not rows or set(rows[0]) != expected_columns:
        raise ValueError("Split assignment table is missing rows or required columns")
    for expected_index, row in enumerate(rows):
        if int(row["row_index"]) != expected_index:
            raise ValueError("Split assignments are not aligned by row_index")
    return rows


def partition_indices(assignments: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation indices while deliberately excluding test rows."""
    train = np.asarray(
        [int(row["row_index"]) for row in assignments if row["split"] == "train"], dtype=int
    )
    validation = np.asarray(
        [int(row["row_index"]) for row in assignments if row["split"] == "validation"],
        dtype=int,
    )
    if train.size == 0 or validation.size == 0:
        raise ValueError("Training and validation partitions must both be nonempty")
    if set(train) & set(validation):
        raise ValueError("Training and validation row overlap detected")
    return train, validation


def run_fixed_baselines(
    training_features: np.ndarray,
    training_target: np.ndarray,
    validation_features: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> list[BaselinePredictions]:
    """Fit or apply the four fixed baseline definitions in prespecified order."""
    ev_index = feature_names.index("expected_value_difference_b_minus_a_oracle")
    models = [
        constant_mean_baseline(training_target, validation_features.shape[0]),
        expected_value_heuristic(
            validation_features[:, ev_index],
            tie_prediction=config["expected_value_heuristic"]["tie_prediction"],
        ),
        ridge_baseline(
            training_features,
            training_target,
            validation_features,
            feature_names,
            alpha=config["ridge"]["alpha"],
            fit_intercept=config["ridge"]["fit_intercept"],
        ),
        decision_tree_baseline(
            training_features,
            training_target,
            validation_features,
            feature_names,
            criterion=config["decision_tree"]["criterion"],
            max_depth=config["decision_tree"]["max_depth"],
            min_samples_leaf=config["decision_tree"]["min_samples_leaf"],
            random_seed=config["random_seed"],
        ),
    ]
    if tuple(model.name for model in models) != MODEL_ORDER:
        raise ValueError("Baseline order differs from the prespecified experiment")
    return models


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(temporary, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_feature_values(path: Path, values: dict[str, float], value_name: str) -> None:
    rows = [
        {"feature": feature, value_name: value}
        for feature, value in sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
    ]
    _write_csv(path, ["feature", value_name], rows)


def _comparison_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": name,
            "mae": metrics["models"][name]["metrics"]["unweighted"]["mae"],
            "rmse": metrics["models"][name]["metrics"]["unweighted"]["rmse"],
            "r2": metrics["models"][name]["metrics"]["unweighted"]["r2"],
            "weighted_mae": metrics["models"][name]["metrics"]["participant_count_weighted"]["mae"],
            "weighted_rmse": metrics["models"][name]["metrics"]["participant_count_weighted"][
                "rmse"
            ],
        }
        for name in MODEL_ORDER
    ]


def write_comparison_tables(metrics: dict[str, Any]) -> None:
    """Write compact CSV and Markdown tables from executed metric values."""
    rows = _comparison_rows(metrics)
    columns = ["model", "mae", "rmse", "r2", "weighted_mae", "weighted_rmse"]
    _write_csv(DEFAULT_COMPARISON_CSV, columns, rows)
    lines = [
        "| Baseline | MAE | RMSE | R² | n-weighted MAE | n-weighted RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['r2']:.4f} | {row['weighted_mae']:.4f} | "
            f"{row['weighted_rmse']:.4f} |"
        )
    DEFAULT_COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline_report(metrics: dict[str, Any], path: Path = DEFAULT_REPORT) -> None:
    """Document results and the specific lesson provided by each baseline."""
    values = metrics["models"]
    constant_mae = values["constant_training_mean"]["metrics"]["unweighted"]["mae"]

    def improvement(name: str) -> float:
        return constant_mae - values[name]["metrics"]["unweighted"]["mae"]

    lines = [
        "# Prediction baselines",
        "",
        (
            "These results were generated by `decisionlab-run-baselines` on the locked "
            f"**{metrics['evaluation_split']}** partition only. The test partition was not "
            "predicted or evaluated. No hyperparameter search was performed."
        ),
        "",
        "## Comparison",
        "",
        Path(DEFAULT_COMPARISON_MD).read_text(encoding="utf-8").rstrip(),
        "",
        "Unweighted MAE is the primary metric. Participant-count-weighted metrics are sensitivity "
        "analyses and do not replace problem-condition-level evaluation.",
        "",
        "## What each baseline teaches us",
        "",
        (
            "- **Constant training mean:** establishes how well prediction works with no problem "
            f"information. Its validation MAE is {constant_mae:.4f}."
        ),
        (
            "- **Hard expected-value rule:** tests the strict heuristic ‘choose the higher-EV "
            "gamble.’ It changes MAE relative to the constant by "
            f"{improvement('expected_value_hard_rule_oracle'):+.4f}. Predictions are deliberately "
            "extreme (0, 0.5, or 1), and B's probabilities are oracle information when ambiguous."
        ),
        (
            "- **Standardized ridge regression:** tests whether an additive combination of the 28 "
            "audited behavioral features improves prediction. It changes MAE relative to the "
            f"constant by {improvement('ridge_engineered_oracle'):+.4f}. Predictions outside "
            "`[0,1]` are explicitly clipped and the clip count is recorded."
        ),
        (
            "- **Shallow decision tree:** tests whether a small amount of nonlinearity and feature "
            "interaction helps without tuning an ensemble. It changes MAE relative to the constant "
            f"by {improvement('shallow_decision_tree_engineered_oracle'):+.4f}."
        ),
        "",
        (
            "A positive MAE change above means lower error than the constant baseline; a negative "
            "value means worse error. These are predictive comparisons, not causal or mechanistic "
            "effects."
        ),
        "",
        "## Feature and leakage scope",
        "",
        (
            f"The regression and tree use {metrics['feature_count']} engineered features. "
            f"{metrics['oracle_feature_count']} are explicitly audited as oracle under ambiguity. "
            "No feature contains `bRate`, `bRate_std`, `n`, `Block`, row identifiers, or "
            "problem IDs."
        ),
        "",
        "All models use the same grouped training and validation rows. Feedback variants of one "
        "problem cannot cross partitions. Training objectives are unweighted; `n` appears only in "
        "the prespecified weighted sensitivity metrics.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `artifacts/experiments/baselines/metrics.json`",
        "- `artifacts/experiments/baselines/predictions_validation.csv`",
        "- `artifacts/experiments/baselines/ridge_coefficients.csv`",
        "- `artifacts/experiments/baselines/tree_feature_importances.csv`",
        "- `reports/tables/baseline_comparison.csv`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baseline_experiment(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Execute and persist all four validation-only baseline experiments."""
    config = load_baseline_config(config_path)
    feature_summary = build_feature_tables(raw_dir=raw_dir, manifest_path=manifest_path)
    split_summary = build_evaluation_protocol(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        config_path=DEFAULT_EVALUATION_CONFIG,
    )
    feature_names = audited_feature_names(feature_summary)
    features = _read_engineered_features(DEFAULT_ENGINEERED_FEATURES, feature_names)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    assignments = _read_assignments(DEFAULT_ASSIGNMENTS, len(selections))
    if features.shape != (len(selections), len(feature_names)):
        raise ValueError("Feature matrix shape does not match validated selections")
    train_indices, validation_indices = partition_indices(assignments)

    training_target = np.asarray([selections[index].brate for index in train_indices])
    validation_target = np.asarray([selections[index].brate for index in validation_indices])
    validation_n = np.asarray([selections[index].n for index in validation_indices])
    models = run_fixed_baselines(
        features[train_indices],
        training_target,
        features[validation_indices],
        feature_names,
        config,
    )

    model_metrics: dict[str, Any] = {}
    for model in models:
        model_metrics[model.name] = {
            "metrics": evaluate_brate_predictions(
                validation_target,
                model.predictions,
                validation_n,
            ),
            "metadata": model.metadata,
            "prediction_summary": {
                "min": float(np.min(model.predictions)),
                "max": float(np.max(model.predictions)),
                "mean": float(np.mean(model.predictions)),
            },
        }

    oracle_features = [
        name
        for name, audit in feature_summary["leakage_audit"]["features"].items()
        if audit["availability"] == "oracle under ambiguity"
    ]
    metrics: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decisionlab_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "source_commit": feature_summary["source_commit"],
        "source_sha256": feature_summary["source_sha256"],
        "feature_build_summary_sha256": sha256_file(DEFAULT_FEATURE_SUMMARY),
        "split_assignments_sha256": split_summary["assignments_output"]["sha256"],
        "config": config,
        "evaluation_split": "validation",
        "training_rows": int(train_indices.size),
        "validation_rows": int(validation_indices.size),
        "test_rows_predicted": 0,
        "test_metrics_computed": False,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "oracle_feature_count": len(oracle_features),
        "oracle_feature_names": oracle_features,
        "target_leakage_audit": {
            "status": "PASS",
            "target_derived_features": [],
            "forbidden_feature_intersection": sorted(set(feature_names) & FORBIDDEN_FEATURES),
            "participant_count_used_as_predictor": False,
            "block_used_as_predictor": False,
            "identifiers_used_as_predictors": False,
        },
        "models": model_metrics,
    }

    prediction_columns = [
        "row_index",
        "problem",
        "structural_fingerprint",
        "observed_bRate",
        "participant_count_n",
        *MODEL_ORDER,
    ]
    prediction_rows: list[dict[str, Any]] = []
    for position, row_index in enumerate(validation_indices):
        assignment = assignments[int(row_index)]
        prediction_rows.append(
            {
                "row_index": int(row_index),
                "problem": selections[int(row_index)].problem,
                "structural_fingerprint": assignment["structural_fingerprint"],
                "observed_bRate": validation_target[position],
                "participant_count_n": validation_n[position],
                **{model.name: model.predictions[position] for model in models},
            }
        )
    _write_csv(DEFAULT_PREDICTIONS, prediction_columns, prediction_rows)
    _write_feature_values(
        DEFAULT_RIDGE_VALUES,
        next(model.feature_values for model in models if model.name == "ridge_engineered_oracle"),
        "standardized_coefficient",
    )
    _write_feature_values(
        DEFAULT_TREE_VALUES,
        next(
            model.feature_values
            for model in models
            if model.name == "shallow_decision_tree_engineered_oracle"
        ),
        "importance",
    )
    write_comparison_tables(metrics)
    write_baseline_report(metrics)
    metrics["outputs"] = {
        str(DEFAULT_PREDICTIONS.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_PREDICTIONS),
        str(DEFAULT_RIDGE_VALUES.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_RIDGE_VALUES),
        str(DEFAULT_TREE_VALUES.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_TREE_VALUES),
        str(DEFAULT_COMPARISON_CSV.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_COMPARISON_CSV),
        str(DEFAULT_COMPARISON_MD.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_COMPARISON_MD),
        str(DEFAULT_REPORT.relative_to(PROJECT_ROOT)): sha256_file(DEFAULT_REPORT),
    }
    DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """Run all fixed baseline models and save their validation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    metrics = run_baseline_experiment(args.raw_dir, args.manifest, args.config)
    print(
        f"Baselines complete on {metrics['evaluation_split']}: rows={metrics['validation_rows']:,}"
    )
    for name in MODEL_ORDER:
        values = metrics["models"][name]["metrics"]["unweighted"]
        print(f"{name}: MAE={values['mae']:.6f} RMSE={values['rmse']:.6f} R2={values['r2']:.6f}")


if __name__ == "__main__":
    main()
