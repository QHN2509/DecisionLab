"""Nested grouped-CV model tuning, family selection, and outer OOF evaluation."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import ParameterGrid

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_selections
from decisionlab.evaluation.metrics import (
    evaluate_brate_predictions,
    problem_group_mae_interval,
    problem_group_regression_metrics,
)
from decisionlab.evaluation.nested import (
    DEFAULT_CONFIG as DEFAULT_EVALUATION_CONFIG,
)
from decisionlab.evaluation.nested import (
    DEFAULT_INNER_ASSIGNMENTS,
    DEFAULT_OUTER_ASSIGNMENTS,
    audit_nested_fold_assignments,
    build_nested_cv_assignments,
    load_nested_evaluation_config,
    read_inner_assignments,
    read_outer_assignments,
)
from decisionlab.evaluation.nested import (
    DEFAULT_SUMMARY as DEFAULT_FOLD_SUMMARY,
)
from decisionlab.evaluation.splitting import grouped_fold_indices
from decisionlab.experiments.baselines import (
    FORBIDDEN_FEATURES,
    _read_engineered_features,
    _write_csv,
    audited_feature_names,
)
from decisionlab.experiments.provenance import (
    finalize_run_provenance,
    start_run_provenance,
)
from decisionlab.features.behavioral import (
    DEFAULT_DOCUMENTATION as DEFAULT_FEATURE_DOCUMENTATION,
)
from decisionlab.features.behavioral import (
    DEFAULT_ENGINEERED_FEATURES,
    DEFAULT_RAW_FEATURES,
    build_feature_tables,
)
from decisionlab.features.behavioral import (
    DEFAULT_SUMMARY as DEFAULT_FEATURE_SUMMARY,
)
from decisionlab.models.ensembles import build_candidate_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_selection.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "nested_model_selection"
DEFAULT_METRICS = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "outer_oof_predictions.csv"
DEFAULT_TUNING = DEFAULT_OUTPUT_DIR / "inner_tuning_results.csv"
DEFAULT_SELECTIONS = DEFAULT_OUTPUT_DIR / "outer_fold_selections.json"
DEFAULT_FEATURE_NAMES = DEFAULT_OUTPUT_DIR / "feature_names.json"
DEFAULT_EXPERIMENT_CONFIG = DEFAULT_OUTPUT_DIR / "experiment_config.json"
DEFAULT_PIPELINE = DEFAULT_OUTPUT_DIR / "production_pipeline.joblib"
DEFAULT_OUTER_PIPELINE_DIR = DEFAULT_OUTPUT_DIR / "outer_fold_pipelines"
DEFAULT_COMPARISON = PROJECT_ROOT / "reports" / "tables" / "nested_model_comparison.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "nested_model_selection.md"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "provenance.json"
CANDIDATE_ORDER = ("random_forest", "gradient_boosting")


def load_model_selection_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the frozen nested search and selection contract."""
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_name",
        "evaluation_design",
        "feature_set",
        "random_seed",
        "nested_cross_validation",
        "selection",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }
    if set(config) != required:
        raise ValueError(f"Model-selection config must define exactly: {sorted(required)}")
    if config["evaluation_design"] != "nested_grouped_cv_v1":
        raise ValueError("Model selection requires nested grouped CV")
    if config["feature_set"] != "engineered_design_oracle_v1":
        raise ValueError("Unexpected feature set")
    cv = config["nested_cross_validation"]
    if cv["primary_metric"] != "problem_group_equal_weighted_mae":
        raise ValueError("Nested CV must use equal-problem MAE as its primary metric")
    if cv["outer_folds"] < 2 or cv["inner_folds"] < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds")
    if config["selection"]["inner_oof_mae_tolerance"] < 0.0:
        raise ValueError("Inner OOF MAE tolerance must be nonnegative")
    if config["xgboost"]["enabled"] is not False or not config["xgboost"]["reason"]:
        raise ValueError("The XGBoost decision must be explicit")
    return config


def grouped_cv_splits(groups: np.ndarray, *, folds: int, random_seed: int):
    """Compatibility wrapper for the canonical grouped-fold implementation."""
    return grouped_fold_indices(groups, folds=folds, random_seed=random_seed)


def audited_oracle_feature_names(
    feature_summary: dict[str, Any], feature_names: list[str]
) -> list[str]:
    """Return the audited predictors that use oracle information under ambiguity."""
    audit = feature_summary["leakage_audit"]["features"]
    return [
        name for name in feature_names if audit[name]["availability"] == "oracle under ambiguity"
    ]


def candidate_parameter_sets(config: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Expand one frozen modest parameter grid."""
    specification = config[name]
    return [specification["fixed"] | values for values in ParameterGrid(specification["grid"])]


def evaluate_parameter_set(
    name: str,
    parameters: dict[str, Any],
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    random_seed: int,
) -> dict[str, Any]:
    """Produce inner OOF scores, fitting a fresh full pipeline in every fold."""
    predictions = np.full(target.size, np.nan)
    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    for fold, (fit_indices, validation_indices) in enumerate(splits):
        if set(groups[fit_indices]) & set(groups[validation_indices]):
            raise ValueError("Structural group leaked within inner CV")
        pipeline = build_candidate_pipeline(name, parameters, random_seed=random_seed + fold)
        pipeline.fit(features[fit_indices], target[fit_indices])
        fold_predictions = pipeline.predict(features[validation_indices])
        predictions[validation_indices] = fold_predictions
        values = problem_group_regression_metrics(
            target[validation_indices], fold_predictions, groups[validation_indices]
        )
        fold_mae.append(values["mae"])
        fold_rmse.append(values["rmse"])
    if np.any(np.isnan(predictions)):
        raise ValueError("Inner OOF predictions do not cover every outer-training row")
    overall = problem_group_regression_metrics(target, predictions, groups)
    return {
        "model": name,
        "parameters": parameters,
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "mean_cv_mae": overall["mae"],
        "std_cv_mae": float(np.std(fold_mae, ddof=1)),
        "mean_cv_rmse": overall["rmse"],
        "std_cv_rmse": float(np.std(fold_rmse, ddof=1)),
    }


def choose_best_tuning_result(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Choose family parameters by pooled inner OOF MAE and deterministic ties."""
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


def select_model(family_results: dict[str, dict[str, Any]], config: dict[str, Any]):
    """Select a family from inner OOF evidence only; outer scores are not accepted."""
    tolerance = config["selection"]["inner_oof_mae_tolerance"]
    best_inner_mae = min(result["mean_cv_mae"] for result in family_results.values())
    eligible = [
        name
        for name, result in family_results.items()
        if result["mean_cv_mae"] <= best_inner_mae + tolerance
    ]
    selected = min(
        eligible,
        key=lambda name: (
            config[name]["complexity_rank"],
            family_results[name]["mean_cv_mae"],
            family_results[name]["std_cv_mae"],
            name,
        ),
    )
    return selected, {
        "rule": config["selection"]["rule"],
        "inner_oof_mae_tolerance": tolerance,
        "best_inner_oof_mae": best_inner_mae,
        "eligible_models": eligible,
        "selected_complexity_rank": config[selected]["complexity_rank"],
    }


def _inner_splits_from_assignments(
    outer_train: np.ndarray,
    outer_fold: int,
    inner_assignments: list[Any],
    inner_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    local_by_global = {int(index): position for position, index in enumerate(outer_train)}
    fold_by_row = {
        row.row_index: row.inner_fold for row in inner_assignments if row.outer_fold == outer_fold
    }
    if set(fold_by_row) != set(outer_train.tolist()):
        raise ValueError("Inner assignments differ from the current outer training rows")
    splits = []
    for inner_fold in range(inner_folds):
        validation = np.asarray(
            [local_by_global[index] for index, fold in fold_by_row.items() if fold == inner_fold],
            dtype=int,
        )
        held_out = set(validation.tolist())
        fit = np.asarray(
            [position for position in range(outer_train.size) if position not in held_out],
            dtype=int,
        )
        splits.append((fit, validation))
    return splits


def run_nested_cv(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    participant_counts: np.ndarray,
    outer_fold_by_row: np.ndarray,
    inner_assignments: list[Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run nested selection and return exactly one outer OOF prediction per row."""
    if not (
        features.shape[0]
        == target.size
        == groups.size
        == participant_counts.size
        == outer_fold_by_row.size
    ):
        raise ValueError("Nested-CV inputs are not row aligned")
    cv = config["nested_cross_validation"]
    selected_oof = np.full(target.size, np.nan)
    family_oof = {name: np.full(target.size, np.nan) for name in CANDIDATE_ORDER}
    tuning_results: list[dict[str, Any]] = []
    fold_selections: list[dict[str, Any]] = []
    fitted_outer_pipelines: dict[int, Any] = {}
    for outer_fold in range(cv["outer_folds"]):
        outer_test = np.flatnonzero(outer_fold_by_row == outer_fold)
        outer_train = np.flatnonzero(outer_fold_by_row != outer_fold)
        if set(groups[outer_train]) & set(groups[outer_test]):
            raise ValueError("Structural group leaked across an outer boundary")
        inner_splits = _inner_splits_from_assignments(
            outer_train, outer_fold, inner_assignments, cv["inner_folds"]
        )
        family_best = {}
        for name in CANDIDATE_ORDER:
            family_tuning = []
            for parameters in candidate_parameter_sets(config, name):
                result = evaluate_parameter_set(
                    name,
                    parameters,
                    features[outer_train],
                    target[outer_train],
                    groups[outer_train],
                    inner_splits,
                    random_seed=config["random_seed"] + outer_fold * 100,
                )
                result["outer_fold"] = outer_fold
                family_tuning.append(result)
                tuning_results.append(result)
            best = choose_best_tuning_result(family_tuning, name)
            family_best[name] = best
            pipeline = build_candidate_pipeline(
                name, best["parameters"], random_seed=config["random_seed"] + outer_fold
            )
            pipeline.fit(features[outer_train], target[outer_train])
            family_oof[name][outer_test] = pipeline.predict(features[outer_test])
        selected, rationale = select_model(family_best, config)
        selected_oof[outer_test] = family_oof[selected][outer_test]
        selected_pipeline = build_candidate_pipeline(
            selected,
            family_best[selected]["parameters"],
            random_seed=config["random_seed"] + outer_fold,
        )
        selected_pipeline.fit(features[outer_train], target[outer_train])
        fitted_outer_pipelines[outer_fold] = selected_pipeline
        fold_selections.append(
            {
                "outer_fold": outer_fold,
                "training_rows": int(outer_train.size),
                "test_rows": int(outer_test.size),
                "training_groups": int(np.unique(groups[outer_train]).size),
                "test_groups": int(np.unique(groups[outer_test]).size),
                "selected_model": selected,
                "selected_parameters": family_best[selected]["parameters"],
                "inner_results": {
                    name: {
                        key: value
                        for key, value in family_best[name].items()
                        if key not in {"fold_mae", "fold_rmse"}
                    }
                    for name in CANDIDATE_ORDER
                },
                "selection_rationale": rationale,
            }
        )
    if np.any(np.isnan(selected_oof)) or any(np.any(np.isnan(x)) for x in family_oof.values()):
        raise ValueError("Outer OOF predictions do not cover every development row")
    if np.any((selected_oof < 0.0) | (selected_oof > 1.0)):
        raise ValueError("Selected-procedure OOF predictions fall outside [0, 1]")
    return {
        "selected_oof_predictions": selected_oof,
        "family_oof_predictions": family_oof,
        "selected_procedure_metrics": evaluate_brate_predictions(
            target, selected_oof, groups, participant_counts
        ),
        "family_metrics": {
            name: evaluate_brate_predictions(target, values, groups, participant_counts)
            for name, values in family_oof.items()
        },
        "fold_selections": fold_selections,
        "tuning_results": tuning_results,
        "fitted_outer_pipelines": fitted_outer_pipelines,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_nested_results(metrics: dict[str, Any]) -> None:
    """Generate the canonical table and report only from executed nested metrics."""
    rows = [
        {
            "procedure": "inner_selected_procedure",
            "problem_group_mae_primary": metrics["primary_metrics"]["mae"],
            "condition_row_mae_secondary": metrics["secondary_metrics"]["condition_row_unweighted"][
                "mae"
            ],
            "participant_count_weighted_mae_secondary": metrics["secondary_metrics"][
                "participant_count_weighted"
            ]["mae"],
        }
    ]
    for name in CANDIDATE_ORDER:
        values = metrics["family_metrics"][name]
        rows.append(
            {
                "procedure": f"independently_inner_tuned_{name}",
                "problem_group_mae_primary": values["problem_group_equal_weighted"]["mae"],
                "condition_row_mae_secondary": values["condition_row_unweighted"]["mae"],
                "participant_count_weighted_mae_secondary": values["participant_count_weighted"][
                    "mae"
                ],
            }
        )
    _write_csv(DEFAULT_COMPARISON, list(rows[0]), rows)
    interval = metrics["primary_mae_group_bootstrap_interval"]
    lines = [
        "# Nested grouped-CV model selection",
        "",
        (
            "All choices13k rows are development data. This report uses complete outer "
            "out-of-fold predictions from a 5 × 5 nested structural-group CV procedure; it "
            "does not claim an untouched confirmatory holdout."
        ),
        "",
        "## Headline",
        "",
        (
            "The inner-selected procedure achieved equal-problem outer OOF MAE "
            f"**{metrics['primary_metrics']['mae']:.4f}** "
            f"({interval['confidence_level']:.0%} structural-group bootstrap interval "
            f"{interval['lower']:.4f}–{interval['upper']:.4f})."
        ),
        "",
        "Condition-row and participant-count-weighted metrics in the machine-readable table "
        "are secondary.",
        "",
        "## Isolation",
        "",
        (
            "Hyperparameters and model family were selected only from inner OOF predictions "
            "within each outer training portion. Every outer test group was predicted once by "
            "a pipeline fit without that group."
        ),
        "",
        "The production model was selected and fit separately on all development rows and has "
        "no direct in-sample performance claim.",
    ]
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model_selection_experiment(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Execute the official nested experiment; this is intentionally expensive."""
    config = load_model_selection_config(config_path)
    provenance = start_run_provenance(
        experiment_name=config["experiment_name"],
        config_paths=[config_path, DEFAULT_EVALUATION_CONFIG],
        configuration_values={
            "raw_dir": str(raw_dir),
            "dataset_manifest_path": str(manifest_path),
            "allow_dirty": allow_dirty,
        },
        dataset_manifest_path=manifest_path,
        raw_dir=raw_dir,
        fold_specification_identifier=config["evaluation_design"],
        entry_module=Path(__file__),
        allow_dirty=allow_dirty,
    )
    evaluation = load_nested_evaluation_config(DEFAULT_EVALUATION_CONFIG)
    cv = config["nested_cross_validation"]
    for key in ("outer_folds", "inner_folds", "outer_seed", "inner_seed"):
        if cv[key] != evaluation[key]:
            raise ValueError(f"Model-selection and evaluation configs disagree on {key}")
    feature_summary = build_feature_tables(raw_dir=raw_dir, manifest_path=manifest_path)
    fold_summary = build_nested_cv_assignments(raw_dir, manifest_path)
    feature_names = audited_feature_names(feature_summary)
    oracle_feature_names = audited_oracle_feature_names(feature_summary, feature_names)
    features = _read_engineered_features(DEFAULT_ENGINEERED_FEATURES, feature_names)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    outer = read_outer_assignments(DEFAULT_OUTER_ASSIGNMENTS, len(selections))
    inner = read_inner_assignments(DEFAULT_INNER_ASSIGNMENTS)
    audit_nested_fold_assignments(
        outer, inner, outer_folds=cv["outer_folds"], inner_folds=cv["inner_folds"]
    )
    target = np.asarray([row.brate for row in selections])
    groups = np.asarray([row.structural_fingerprint for row in outer])
    participant_counts = np.asarray([row.n for row in selections])
    outer_fold_by_row = np.asarray([row.outer_fold for row in outer])
    result = run_nested_cv(
        features, target, groups, participant_counts, outer_fold_by_row, inner, config
    )
    prediction_rows = [
        {
            "row_index": index,
            "problem": selections[index].problem,
            "structural_fingerprint": groups[index],
            "outer_fold": int(outer_fold_by_row[index]),
            "observed_bRate": target[index],
            "participant_count_n": participant_counts[index],
            "selected_procedure": result["selected_oof_predictions"][index],
            **{name: result["family_oof_predictions"][name][index] for name in CANDIDATE_ORDER},
        }
        for index in range(target.size)
    ]
    _write_csv(DEFAULT_PREDICTIONS, list(prediction_rows[0]), prediction_rows)
    tuning_rows = [
        {
            "outer_fold": row["outer_fold"],
            "model": row["model"],
            "parameters_json": json.dumps(row["parameters"], sort_keys=True),
            "inner_oof_problem_mae": row["mean_cv_mae"],
            "inner_fold_mae_std": row["std_cv_mae"],
        }
        for row in result["tuning_results"]
    ]
    _write_csv(DEFAULT_TUNING, list(tuning_rows[0]), tuning_rows)
    _write_json(DEFAULT_SELECTIONS, result["fold_selections"])
    _write_json(
        DEFAULT_FEATURE_NAMES,
        {"feature_set": config["feature_set"], "feature_names": feature_names},
    )
    _write_json(DEFAULT_EXPERIMENT_CONFIG, config)
    DEFAULT_OUTER_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    outer_pipeline_hashes = {}
    for fold, pipeline in result["fitted_outer_pipelines"].items():
        path = DEFAULT_OUTER_PIPELINE_DIR / f"fold_{fold}.joblib"
        joblib.dump(pipeline, path)
        outer_pipeline_hashes[str(fold)] = sha256_file(path)

    # Deployment fitting is separate and makes no performance claim.
    full_splits = grouped_cv_splits(
        groups, folds=cv["inner_folds"], random_seed=cv["inner_seed"] + cv["outer_folds"]
    )
    full_results = []
    for name in CANDIDATE_ORDER:
        for parameters in candidate_parameter_sets(config, name):
            full_results.append(
                evaluate_parameter_set(
                    name,
                    parameters,
                    features,
                    target,
                    groups,
                    full_splits,
                    random_seed=config["random_seed"] + 10_000,
                )
            )
    full_best = {name: choose_best_tuning_result(full_results, name) for name in CANDIDATE_ORDER}
    production_name, production_rationale = select_model(full_best, config)
    production_pipeline = build_candidate_pipeline(
        production_name,
        full_best[production_name]["parameters"],
        random_seed=config["random_seed"],
    )
    production_pipeline.fit(features, target)
    DEFAULT_PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_pipeline, DEFAULT_PIPELINE)
    metrics = {
        "experiment_name": config["experiment_name"],
        "evaluation_design": "nested_grouped_cv_v1",
        "dataset_role": "development",
        "confirmatory_holdout": False,
        "headline_source": "complete_outer_out_of_fold_predictions",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decisionlab_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "source_commit": feature_summary["source_commit"],
        "source_sha256": feature_summary["source_sha256"],
        "config_sha256": sha256_file(config_path),
        "fold_assignment_sha256": {
            "outer": fold_summary["outer_assignments"]["sha256"],
            "inner": fold_summary["inner_assignments"]["sha256"],
        },
        "rows": int(target.size),
        "structural_groups": int(np.unique(groups).size),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "oracle_feature_count": len(oracle_feature_names),
        "oracle_feature_names": oracle_feature_names,
        "oof_rows_predicted": int(np.isfinite(result["selected_oof_predictions"]).sum()),
        "outer_group_overlap_count": 0,
        "outer_test_access_during_selection": False,
        "primary_metrics": result["selected_procedure_metrics"]["problem_group_equal_weighted"],
        "primary_mae_group_bootstrap_interval": problem_group_mae_interval(
            target,
            result["selected_oof_predictions"],
            groups,
            repeats=evaluation["bootstrap_repeats"],
            confidence_level=evaluation["confidence_level"],
            random_seed=evaluation["outer_seed"],
        ),
        "secondary_metrics": {
            key: value
            for key, value in result["selected_procedure_metrics"].items()
            if key != "problem_group_equal_weighted"
        },
        "family_metrics": result["family_metrics"],
        "outer_fold_selections": result["fold_selections"],
        "outer_pipeline_sha256": outer_pipeline_hashes,
        "production_model": {
            "name": production_name,
            "parameters": full_best[production_name]["parameters"],
            "selection_rationale": production_rationale,
            "performance_claim": None,
        },
        "historical_artifacts_status": "stale_not_used",
        "run_eligibility": provenance["run_eligibility"],
        "provenance_path": str(DEFAULT_PROVENANCE.relative_to(PROJECT_ROOT)),
        "target_leakage_audit": {
            "forbidden_feature_intersection": sorted(set(feature_names) & FORBIDDEN_FEATURES),
            "participant_count_used_as_predictor": False,
        },
    }
    _write_json(DEFAULT_METRICS, metrics)
    write_nested_results(metrics)
    finalize_run_provenance(
        provenance,
        fold_artifacts={
            "outer_assignments": DEFAULT_OUTER_ASSIGNMENTS,
            "inner_assignments": DEFAULT_INNER_ASSIGNMENTS,
            "fold_summary": DEFAULT_FOLD_SUMMARY,
        },
        output_artifacts=[
            DEFAULT_METRICS,
            DEFAULT_PREDICTIONS,
            DEFAULT_TUNING,
            DEFAULT_SELECTIONS,
            DEFAULT_FEATURE_NAMES,
            DEFAULT_EXPERIMENT_CONFIG,
            DEFAULT_PIPELINE,
            DEFAULT_COMPARISON,
            DEFAULT_REPORT,
            DEFAULT_RAW_FEATURES,
            DEFAULT_ENGINEERED_FEATURES,
            DEFAULT_FEATURE_SUMMARY,
            DEFAULT_FEATURE_DOCUMENTATION,
            *sorted(DEFAULT_OUTER_PIPELINE_DIR.glob("fold_*.joblib")),
        ],
        output_path=DEFAULT_PROVENANCE,
    )
    return metrics


def main() -> None:
    """Run the official nested grouped-CV experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a clearly marked non-official run from a dirty worktree.",
    )
    args = parser.parse_args()
    metrics = run_model_selection_experiment(
        args.raw_dir, args.manifest, args.config, allow_dirty=args.allow_dirty
    )
    print(f"Outer OOF equal-problem MAE: {metrics['primary_metrics']['mae']:.6f}")


if __name__ == "__main__":
    main()
