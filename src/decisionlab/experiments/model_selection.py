"""Tune and select DecisionLab tree ensembles without touching the locked test set."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, ParameterGrid

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_selections
from decisionlab.evaluation.metrics import evaluate_brate_predictions, regression_metrics
from decisionlab.evaluation.protocol import DEFAULT_ASSIGNMENTS, build_evaluation_protocol
from decisionlab.evaluation.protocol import DEFAULT_CONFIG as DEFAULT_EVALUATION_CONFIG
from decisionlab.experiments.baselines import (
    FORBIDDEN_FEATURES,
    _read_assignments,
    _read_engineered_features,
    _write_csv,
    audited_feature_names,
    partition_indices,
)
from decisionlab.features.behavioral import DEFAULT_ENGINEERED_FEATURES, build_feature_tables
from decisionlab.features.behavioral import DEFAULT_SUMMARY as DEFAULT_FEATURE_SUMMARY
from decisionlab.models.ensembles import build_candidate_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_selection.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "model_selection"
DEFAULT_METRICS = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "predictions_validation.csv"
DEFAULT_TUNING = DEFAULT_OUTPUT_DIR / "tuning_results.csv"
DEFAULT_FEATURE_NAMES = DEFAULT_OUTPUT_DIR / "feature_names.json"
DEFAULT_EXPERIMENT_CONFIG = DEFAULT_OUTPUT_DIR / "experiment_config.json"
DEFAULT_SELECTED_MODEL = DEFAULT_OUTPUT_DIR / "selected_model.joblib"
DEFAULT_PREPROCESSOR = DEFAULT_OUTPUT_DIR / "preprocessing_pipeline.joblib"
DEFAULT_PIPELINE = DEFAULT_OUTPUT_DIR / "selected_pipeline.joblib"
DEFAULT_IMPORTANCES = DEFAULT_OUTPUT_DIR / "selected_feature_importances.csv"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "tables" / "model_comparison.csv"
DEFAULT_COMPARISON_MD = PROJECT_ROOT / "reports" / "tables" / "model_comparison.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "model_selection.md"

CANDIDATE_ORDER = ("random_forest", "gradient_boosting")


def load_model_selection_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and strictly validate the reproducible search and selection contract."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "experiment_name",
        "evaluation_split",
        "feature_set",
        "random_seed",
        "cross_validation",
        "selection",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }
    if set(config) != required:
        raise ValueError(f"Model-selection config must define exactly: {sorted(required)}")
    if config["evaluation_split"] != "validation":
        raise ValueError("Model selection may evaluate only the validation split")
    if config["feature_set"] != "engineered_design_oracle_v1":
        raise ValueError("Unexpected feature set")
    cv = config["cross_validation"]
    if cv != {"folds": 5, "shuffle_groups": True, "primary_metric": "mae"}:
        raise ValueError("Cross-validation settings differ from the locked protocol")
    if config["selection"]["validation_mae_tolerance"] < 0.0:
        raise ValueError("Validation MAE tolerance must be nonnegative")
    if config["xgboost"]["enabled"] is not False or not config["xgboost"]["reason"]:
        raise ValueError("The XGBoost inclusion decision must be explicit and documented")
    return config


def grouped_cv_splits(
    groups: np.ndarray, *, folds: int, random_seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic group folds and prove zero overlap in every fold."""
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("Groups must be a nonempty vector")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    dummy = np.zeros((groups.size, 1))
    splits = list(splitter.split(dummy, groups=groups))
    for train_indices, fold_indices in splits:
        if set(groups[train_indices]) & set(groups[fold_indices]):
            raise ValueError("Structural group overlap detected within cross-validation")
    return splits


def candidate_parameter_sets(config: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Expand a candidate's small declared grid and combine it with fixed settings."""
    specification = config[name]
    return [specification["fixed"] | values for values in ParameterGrid(specification["grid"])]


def evaluate_parameter_set(
    name: str,
    parameters: dict[str, Any],
    features: np.ndarray,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    random_seed: int,
) -> dict[str, Any]:
    """Evaluate one parameter set on the shared structural-group folds."""
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    for fit_indices, fold_indices in splits:
        pipeline = build_candidate_pipeline(name, parameters, random_seed=random_seed)
        pipeline.fit(features[fit_indices], target[fit_indices])
        predictions = pipeline.predict(features[fold_indices])
        values = regression_metrics(target[fold_indices], predictions)
        fold_mae.append(values["mae"])
        fold_rmse.append(values["rmse"])
    return {
        "model": name,
        "parameters": parameters,
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "mean_cv_mae": float(np.mean(fold_mae)),
        "std_cv_mae": float(np.std(fold_mae, ddof=1)),
        "mean_cv_rmse": float(np.mean(fold_rmse)),
        "std_cv_rmse": float(np.std(fold_rmse, ddof=1)),
    }


def choose_best_tuning_result(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Choose a candidate's setting by mean CV MAE, then stability and parameters."""
    matching = [result for result in results if result["model"] == name]
    if not matching:
        raise ValueError(f"No tuning results for {name}")
    return min(
        matching,
        key=lambda result: (
            result["mean_cv_mae"],
            result["std_cv_mae"],
            json.dumps(result["parameters"], sort_keys=True),
        ),
    )


def select_model(
    candidate_results: dict[str, dict[str, Any]], config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Apply the prespecified performance-complexity-stability selection rule."""
    tolerance = config["selection"]["validation_mae_tolerance"]
    best_validation_mae = min(
        result["validation_metrics"]["unweighted"]["mae"] for result in candidate_results.values()
    )
    eligible = [
        name
        for name, result in candidate_results.items()
        if result["validation_metrics"]["unweighted"]["mae"] <= best_validation_mae + tolerance
    ]
    selected = min(
        eligible,
        key=lambda name: (
            config[name]["complexity_rank"],
            candidate_results[name]["tuning"]["mean_cv_mae"],
            candidate_results[name]["tuning"]["std_cv_mae"],
            name,
        ),
    )
    rationale = {
        "rule": config["selection"]["rule"],
        "validation_mae_tolerance": tolerance,
        "best_validation_mae": best_validation_mae,
        "eligible_models": eligible,
        "selected_complexity_rank": config[selected]["complexity_rank"],
    }
    return selected, rationale


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as temp:
            temporary_path = Path(temp.name)
        joblib.dump(value, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(contents, encoding="utf-8")


def _write_outputs(
    metrics: dict[str, Any],
    tuning_results: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    selected_pipeline: Any,
    feature_names: list[str],
    config: dict[str, Any],
) -> None:
    tuning_rows = [
        {
            "model": result["model"],
            "parameters_json": json.dumps(result["parameters"], sort_keys=True),
            "mean_cv_mae": result["mean_cv_mae"],
            "std_cv_mae": result["std_cv_mae"],
            "mean_cv_rmse": result["mean_cv_rmse"],
            "std_cv_rmse": result["std_cv_rmse"],
            **{f"fold_{index + 1}_mae": value for index, value in enumerate(result["fold_mae"])},
        }
        for result in tuning_results
    ]
    fold_columns = [f"fold_{index + 1}_mae" for index in range(config["cross_validation"]["folds"])]
    _write_csv(
        DEFAULT_TUNING,
        [
            "model",
            "parameters_json",
            "mean_cv_mae",
            "std_cv_mae",
            "mean_cv_rmse",
            "std_cv_rmse",
            *fold_columns,
        ],
        tuning_rows,
    )
    _write_csv(
        DEFAULT_PREDICTIONS,
        [
            "row_index",
            "problem",
            "structural_fingerprint",
            "observed_bRate",
            "participant_count_n",
            *CANDIDATE_ORDER,
        ],
        validation_rows,
    )
    _write_json(
        DEFAULT_FEATURE_NAMES,
        {"feature_set": config["feature_set"], "feature_names": feature_names},
    )
    _write_json(DEFAULT_EXPERIMENT_CONFIG, config)
    _atomic_joblib_dump(selected_pipeline.named_steps["model"], DEFAULT_SELECTED_MODEL)
    _atomic_joblib_dump(selected_pipeline[:-1], DEFAULT_PREPROCESSOR)
    _atomic_joblib_dump(selected_pipeline, DEFAULT_PIPELINE)
    importances = selected_pipeline.named_steps["model"].feature_importances_
    importance_rows = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(feature_names, importances, strict=True), key=lambda item: item[1], reverse=True
        )
    ]
    _write_csv(DEFAULT_IMPORTANCES, ["feature", "importance"], importance_rows)
    write_comparison(metrics)
    write_report(metrics)


def write_comparison(metrics: dict[str, Any]) -> None:
    """Generate compact comparison tables from executed experiment results."""
    rows = []
    for name in CANDIDATE_ORDER:
        result = metrics["candidates"][name]
        values = result["validation_metrics"]
        rows.append(
            {
                "model": name,
                "selected": name == metrics["selected_model"],
                "validation_mae": values["unweighted"]["mae"],
                "validation_rmse": values["unweighted"]["rmse"],
                "validation_r2": values["unweighted"]["r2"],
                "weighted_mae": values["participant_count_weighted"]["mae"],
                "weighted_rmse": values["participant_count_weighted"]["rmse"],
                "mean_cv_mae": result["tuning"]["mean_cv_mae"],
                "std_cv_mae": result["tuning"]["std_cv_mae"],
                "complexity_rank": result["complexity_rank"],
            }
        )
    columns = list(rows[0])
    _write_csv(DEFAULT_COMPARISON_CSV, columns, rows)
    lines = [
        (
            "| Model | Selected | Validation MAE | RMSE | R² | n-weighted MAE | "
            "n-weighted RMSE | CV MAE ± SD | Complexity |"
        ),
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {'yes' if row['selected'] else 'no'} | "
            f"{row['validation_mae']:.4f} | {row['validation_rmse']:.4f} | "
            f"{row['validation_r2']:.4f} | {row['weighted_mae']:.4f} | "
            f"{row['weighted_rmse']:.4f} | "
            f"{row['mean_cv_mae']:.4f} ± {row['std_cv_mae']:.4f} | "
            f"{row['complexity_rank']} |"
        )
    DEFAULT_COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(metrics: dict[str, Any]) -> None:
    """Document the executed comparison and prespecified model choice."""
    selected = metrics["selected_model"]
    selected_result = metrics["candidates"][selected]
    lines = [
        "# Model comparison and selection",
        "",
        (
            "This report was generated from the grouped training/validation experiment. "
            "The locked test partition was neither predicted nor evaluated."
        ),
        "",
        "## Comparison",
        "",
        DEFAULT_COMPARISON_MD.read_text(encoding="utf-8").rstrip(),
        "",
        (
            "Unweighted MAE is primary. Participant-count-weighted metrics are "
            "sensitivity analyses only."
        ),
        "",
        "## Selection",
        "",
        (
            f"**Selected model: `{selected}`.** The best setting was first chosen within each "
            "family using mean five-fold grouped training MAE. On validation, models within "
            f"{metrics['selection_rationale']['validation_mae_tolerance']:.3f} MAE of the best "
            "were treated as practically tied; the lower-complexity model then won, followed "
            "by lower CV MAE and CV variability as tie-breakers."
        ),
        "",
        (
            "The selected model has validation MAE "
            f"{selected_result['validation_metrics']['unweighted']['mae']:.4f} and grouped-CV "
            f"MAE {selected_result['tuning']['mean_cv_mae']:.4f} ± "
            f"{selected_result['tuning']['std_cv_mae']:.4f}. This balances held-out "
            "performance, fold stability, and complexity rather than automatically preferring "
            "the most flexible candidate."
        ),
        "",
        "## Interpretability and complexity",
        "",
    ]
    for name in CANDIDATE_ORDER:
        result = metrics["candidates"][name]
        lines.append(
            f"- **{name}:** {result['interpretability']} Complexity rank: "
            f"{result['complexity_rank']}."
        )
    lines.extend(
        [
            "",
            (
                "XGBoost was not added: the configuration records that its large optional "
                "dependency and overlap with sklearn gradient boosting were not justified for "
                "this modest comparison. This can be revisited if these ensembles leave a "
                "material performance gap."
            ),
            "",
            (
                "Feature importance values describe predictive use by the fitted ensemble, not "
                "causal effects. The feature set includes design-oracle probabilities under "
                "ambiguity, so performance does not represent a strictly participant-visible "
                "deployment setting."
            ),
            "",
            "## Saved artifacts",
            "",
            "- `artifacts/experiments/model_selection/selected_pipeline.joblib`",
            "- `artifacts/experiments/model_selection/selected_model.joblib`",
            "- `artifacts/experiments/model_selection/preprocessing_pipeline.joblib`",
            "- `artifacts/experiments/model_selection/metrics.json`",
            "- `artifacts/experiments/model_selection/tuning_results.csv`",
            "- `artifacts/experiments/model_selection/feature_names.json`",
            "- `artifacts/experiments/model_selection/experiment_config.json`",
        ]
    )
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model_selection_experiment(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Tune candidates on grouped training folds and select using validation only."""
    config = load_model_selection_config(config_path)
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
    train_indices, validation_indices = partition_indices(assignments)
    training_features = features[train_indices]
    training_target = np.asarray([selections[index].brate for index in train_indices])
    validation_features = features[validation_indices]
    validation_target = np.asarray([selections[index].brate for index in validation_indices])
    validation_n = np.asarray([selections[index].n for index in validation_indices])
    training_groups = np.asarray(
        [assignments[index]["structural_fingerprint"] for index in train_indices]
    )
    splits = grouped_cv_splits(
        training_groups,
        folds=config["cross_validation"]["folds"],
        random_seed=config["random_seed"],
    )

    tuning_results: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    fitted_pipelines: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in CANDIDATE_ORDER:
        for parameters in candidate_parameter_sets(config, name):
            tuning_results.append(
                evaluate_parameter_set(
                    name,
                    parameters,
                    training_features,
                    training_target,
                    splits,
                    random_seed=config["random_seed"],
                )
            )
        best = choose_best_tuning_result(tuning_results, name)
        pipeline = build_candidate_pipeline(
            name, best["parameters"], random_seed=config["random_seed"]
        )
        pipeline.fit(training_features, training_target)
        model_predictions = pipeline.predict(validation_features)
        transformed_validation = pipeline.named_steps["preprocess"].transform(validation_features)
        raw_predictions = pipeline.named_steps["model"].predict_raw(transformed_validation)
        if np.any((model_predictions < 0.0) | (model_predictions > 1.0)):
            raise ValueError(f"{name} generated predictions outside [0, 1]")
        predictions[name] = model_predictions
        fitted_pipelines[name] = pipeline
        candidates[name] = {
            "best_parameters": best["parameters"],
            "tuning": {
                key: value for key, value in best.items() if key not in {"model", "parameters"}
            },
            "validation_metrics": evaluate_brate_predictions(
                validation_target, model_predictions, validation_n
            ),
            "prediction_summary": {
                "min": float(np.min(model_predictions)),
                "max": float(np.max(model_predictions)),
                "mean": float(np.mean(model_predictions)),
                "raw_min": float(np.min(raw_predictions)),
                "raw_max": float(np.max(raw_predictions)),
                "clipped_prediction_count": int(
                    np.sum((raw_predictions < 0.0) | (raw_predictions > 1.0))
                ),
                "postprocessing": "explicit_clip_0_1",
            },
            "complexity_rank": config[name]["complexity_rank"],
            "interpretability": config[name]["interpretability"],
        }

    selected, rationale = select_model(candidates, config)
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
        "experiment_config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            "src/decisionlab/experiments/model_selection.py": sha256_file(Path(__file__)),
            "src/decisionlab/models/ensembles.py": sha256_file(
                PROJECT_ROOT / "src" / "decisionlab" / "models" / "ensembles.py"
            ),
            "src/decisionlab/evaluation/metrics.py": sha256_file(
                PROJECT_ROOT / "src" / "decisionlab" / "evaluation" / "metrics.py"
            ),
        },
        "feature_build_summary_sha256": sha256_file(DEFAULT_FEATURE_SUMMARY),
        "split_assignments_sha256": split_summary["assignments_output"]["sha256"],
        "config": config,
        "evaluation_split": "validation",
        "training_rows": int(train_indices.size),
        "training_structural_groups": int(np.unique(training_groups).size),
        "validation_rows": int(validation_indices.size),
        "test_rows_predicted": 0,
        "test_metrics_computed": False,
        "cross_validation_group_overlap_count": 0,
        "shared_cv_folds": True,
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
        "xgboost": config["xgboost"],
        "candidates": candidates,
        "selected_model": selected,
        "selection_rationale": rationale,
    }
    validation_rows = []
    for position, row_index in enumerate(validation_indices):
        validation_rows.append(
            {
                "row_index": int(row_index),
                "problem": selections[int(row_index)].problem,
                "structural_fingerprint": assignments[int(row_index)]["structural_fingerprint"],
                "observed_bRate": validation_target[position],
                "participant_count_n": validation_n[position],
                **{name: predictions[name][position] for name in CANDIDATE_ORDER},
            }
        )
    _write_outputs(
        metrics,
        tuning_results,
        validation_rows,
        fitted_pipelines[selected],
        feature_names,
        config,
    )
    output_paths = [
        DEFAULT_PREDICTIONS,
        DEFAULT_TUNING,
        DEFAULT_FEATURE_NAMES,
        DEFAULT_EXPERIMENT_CONFIG,
        DEFAULT_SELECTED_MODEL,
        DEFAULT_PREPROCESSOR,
        DEFAULT_PIPELINE,
        DEFAULT_IMPORTANCES,
        DEFAULT_COMPARISON_CSV,
        DEFAULT_COMPARISON_MD,
        DEFAULT_REPORT,
    ]
    metrics["outputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in output_paths
    }
    _write_json(DEFAULT_METRICS, metrics)
    return metrics


def main() -> None:
    """Run grouped tuning, validation comparison, and model selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    metrics = run_model_selection_experiment(args.raw_dir, args.manifest, args.config)
    print(f"Selected model: {metrics['selected_model']}")
    for name in CANDIDATE_ORDER:
        values = metrics["candidates"][name]["validation_metrics"]["unweighted"]
        cv = metrics["candidates"][name]["tuning"]
        print(
            f"{name}: validation MAE={values['mae']:.6f}, "
            f"CV MAE={cv['mean_cv_mae']:.6f} ± {cv['std_cv_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
