"""Interpret the selected DecisionLab model without making causal claims."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decisionlab import __version__
from decisionlab.analysis.grouped_permutation import (
    DOMAIN_PERTURBATION_FAMILIES,
    FEATURE_PERTURBATION_FAMILIES,
    engineer_input_matrix,
    grouped_permutation_importance_rows,
    production_feature_inputs,
)
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_and_validate, load_problems, load_selections
from decisionlab.evaluation.metrics import problem_group_regression_metrics, regression_metrics
from decisionlab.evaluation.nested import (
    DEFAULT_INNER_ASSIGNMENTS,
    DEFAULT_OUTER_ASSIGNMENTS,
)
from decisionlab.evaluation.nested import DEFAULT_SUMMARY as DEFAULT_FOLD_SUMMARY
from decisionlab.evaluation.oof import read_complete_outer_oof_predictions
from decisionlab.experiments.baselines import (
    _read_engineered_features,
    _write_csv,
)
from decisionlab.experiments.provenance import (
    finalize_run_provenance,
    start_run_provenance,
    validate_upstream_artifacts,
)
from decisionlab.features.behavioral import DEFAULT_ENGINEERED_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "behavioral_analysis.json"
MODEL_SELECTION_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "nested_model_selection"
DEFAULT_MODEL_METRICS = MODEL_SELECTION_DIR / "metrics.json"
DEFAULT_PIPELINE_DIR = MODEL_SELECTION_DIR / "outer_fold_pipelines"
DEFAULT_MODEL_PREDICTIONS = MODEL_SELECTION_DIR / "outer_oof_predictions.csv"
DEFAULT_FEATURE_NAMES = MODEL_SELECTION_DIR / "feature_names.json"
DEFAULT_MODEL_PROVENANCE = MODEL_SELECTION_DIR / "provenance.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "behavioral"
DEFAULT_STATISTICS = DEFAULT_OUTPUT_DIR / "statistics.json"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "provenance.json"
DEFAULT_FEATURE_IMPORTANCE = DEFAULT_OUTPUT_DIR / "permutation_importance_features.csv"
DEFAULT_DOMAIN_IMPORTANCE = DEFAULT_OUTPUT_DIR / "permutation_importance_domains.csv"
DEFAULT_RELATIONSHIPS = DEFAULT_OUTPUT_DIR / "prediction_relationships.csv"
DEFAULT_SLICES = DEFAULT_OUTPUT_DIR / "error_slices.csv"
DEFAULT_SENSITIVITY = DEFAULT_OUTPUT_DIR / "condition_sensitivity.csv"
DEFAULT_NORMATIVE_CASES = DEFAULT_OUTPUT_DIR / "normative_cases.csv"
DEFAULT_NORMATIVE_EXAMPLES = DEFAULT_OUTPUT_DIR / "normative_examples.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "behavioral_analysis.md"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_IMPORTANCE_FIGURE = FIGURE_DIR / "behavioral_permutation_importance.png"
DEFAULT_RELATIONSHIP_FIGURE = FIGURE_DIR / "behavioral_prediction_relationships.png"
DEFAULT_SENSITIVITY_FIGURE = FIGURE_DIR / "behavioral_condition_sensitivity.png"
DEFAULT_ERROR_FIGURE = FIGURE_DIR / "behavioral_error_slices.png"
DEFAULT_NORMATIVE_FIGURE = FIGURE_DIR / "normative_vs_observed.png"
DEFAULT_NORMATIVE_EXAMPLES_FIGURE = FIGURE_DIR / "normative_case_examples.png"

RELATIONSHIP_FEATURES = {
    "Expected-value difference B − A (oracle)": (
        "expected_value_difference_b_minus_a_oracle",
        "expected_value",
    ),
    "Payoff SD difference B − A (oracle)": (
        "payoff_std_difference_b_minus_a_oracle",
        "payoff_and_risk",
    ),
    "Best-payoff probability difference B − A (oracle)": (
        "best_payoff_probability_difference_b_minus_a_oracle",
        "probability_structure",
    ),
    "Loss-probability difference B − A (oracle)": (
        "loss_probability_difference_b_minus_a_oracle",
        "probability_structure",
    ),
}


def load_behavioral_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the fixed behavioral-analysis specification."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "analysis_name",
        "prediction_source",
        "selected_model_experiment",
        "random_seed",
        "permutation_repeats",
        "permutation_scheme",
        "group_bootstrap_repeats",
        "permutation_confidence_level",
        "relationship_quantile_bins",
        "expected_value_near_tie_threshold",
        "participant_count_split",
        "normative_benchmark",
        "shap",
    }
    if set(config) != required:
        raise ValueError(f"Behavioral config must define exactly: {sorted(required)}")
    if config["prediction_source"] != "nested_outer_oof":
        raise ValueError("Behavioral generalization analysis requires nested outer OOF predictions")
    if config["permutation_repeats"] < 2 or config["relationship_quantile_bins"] < 3:
        raise ValueError("Behavioral analysis requires repeated permutations and at least 3 bins")
    if config["permutation_scheme"] != ("same_outer_fold_structural_group_primitive_recompute_v1"):
        raise ValueError("Behavioral analysis requires the grouped coherent permutation scheme")
    if config["group_bootstrap_repeats"] < 2:
        raise ValueError("Behavioral analysis requires repeated structural-group bootstraps")
    if not 0.0 < config["permutation_confidence_level"] < 1.0:
        raise ValueError("Permutation confidence level must be in (0, 1)")
    if config["expected_value_near_tie_threshold"] <= 0.0:
        raise ValueError("Expected-value near-tie threshold must be positive")
    if config["participant_count_split"] != "development_median":
        raise ValueError("Participant-count slices must use the development-data median")
    normative = config["normative_benchmark"]
    normative_required = {
        "strong_expected_value_difference",
        "human_favor_b_threshold",
        "human_favor_a_threshold",
        "highly_divided_half_width",
        "successful_prediction_max_absolute_error",
        "failed_prediction_min_absolute_error",
        "examples_per_category",
    }
    if set(normative) != normative_required:
        raise ValueError("Normative-benchmark settings are incomplete")
    if not 0.5 < normative["human_favor_b_threshold"] <= 1.0:
        raise ValueError("Human-favor-B threshold must exceed 0.5")
    if not 0.0 <= normative["human_favor_a_threshold"] < 0.5:
        raise ValueError("Human-favor-A threshold must be below 0.5")
    if normative["strong_expected_value_difference"] <= 0.0:
        raise ValueError("Strong expected-value threshold must be positive")
    if config["shap"]["enabled"] is not False or not config["shap"]["reason"]:
        raise ValueError("The SHAP decision must be explicit and documented")
    return config


def _problem_group_mae(
    observed: np.ndarray, predicted: np.ndarray, structural_groups: np.ndarray
) -> float:
    return problem_group_regression_metrics(
        observed, predicted, structural_groups, include_r2=False
    )["mae"]


def quantile_relationship_rows(
    values: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    feature: str,
    domain: str,
    bins: int,
) -> list[dict[str, Any]]:
    """Summarize observed and predicted bRate across validation feature quantiles."""
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        raise ValueError(f"Feature {feature} has insufficient unique values for quantile bins")
    memberships = np.digitize(values, edges[1:-1], right=True)
    rows = []
    for bin_index in range(edges.size - 1):
        mask = memberships == bin_index
        if not np.any(mask):
            continue
        rows.append(
            {
                "feature": feature,
                "domain": domain,
                "bin": bin_index + 1,
                "rows": int(np.sum(mask)),
                "feature_min": float(np.min(values[mask])),
                "feature_max": float(np.max(values[mask])),
                "feature_mean": float(np.mean(values[mask])),
                "observed_brate_mean": float(np.mean(target[mask])),
                "predicted_brate_mean": float(np.mean(predicted[mask])),
                "condition_row_mae": regression_metrics(
                    target[mask], predicted[mask], include_r2=False
                )["mae"],
            }
        )
    return rows


def _slice_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    structural_groups: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    primary = problem_group_regression_metrics(
        target[mask], predicted[mask], structural_groups[mask], include_r2=False
    )
    condition_row = regression_metrics(target[mask], predicted[mask], include_r2=False)
    return {
        "rows": int(np.sum(mask)),
        "structural_groups": int(np.unique(structural_groups[mask]).size),
        "problem_group_mae": primary["mae"],
        "problem_group_rmse": primary["rmse"],
        "problem_group_mean_bias": primary["mean_bias"],
        "condition_row_mae": condition_row["mae"],
        "condition_row_rmse": condition_row["rmse"],
        "condition_row_mean_bias": condition_row["mean_bias"],
    }


def error_slice_rows(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray,
    predicted: np.ndarray,
    structural_groups: np.ndarray,
    participant_counts: np.ndarray,
    *,
    ev_threshold: float,
    participant_threshold: float,
) -> list[dict[str, Any]]:
    """Compute validation errors for prespecified behavioral and measurement slices."""
    position = {name: index for index, name in enumerate(feature_names)}
    feedback = features[:, position["feedback_indicator"]].astype(bool)
    ambiguity = features[:, position["ambiguity_indicator"]].astype(bool)
    ev_difference = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    slices = {
        "feedback": {"no_feedback": ~feedback, "feedback": feedback},
        "ambiguity": {"known_probabilities": ~ambiguity, "ambiguous_b": ambiguity},
        "expected_value_regime": {
            "b_lower_ev": ev_difference < -ev_threshold,
            "near_equal_ev": np.abs(ev_difference) <= ev_threshold,
            "b_higher_ev": ev_difference > ev_threshold,
        },
        "participant_count": {
            f"n_at_or_below_{participant_threshold:g}": participant_counts <= participant_threshold,
            f"n_above_{participant_threshold:g}": participant_counts > participant_threshold,
        },
    }
    rows = []
    for dimension, levels in slices.items():
        for level, mask in levels.items():
            if not np.any(mask):
                raise ValueError(f"Empty error slice: {dimension}/{level}")
            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    **_slice_metrics(target, predicted, structural_groups, mask),
                }
            )
    return rows


def condition_sensitivity_rows(
    predict: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Estimate sensitivity only for conditions with coherent feature-space interventions.

    Lottery shape is intentionally excluded: changing its one-hot encoding without rebuilding
    Gamble B's primitive outcome distribution would create an internally inconsistent scenario.
    """
    position = {name: index for index, name in enumerate(feature_names)}
    ev = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    rows: list[dict[str, Any]] = []

    for condition, interaction in (
        ("feedback_indicator", "expected_value_difference_oracle_x_feedback"),
        ("ambiguity_indicator", "expected_value_difference_oracle_x_ambiguity"),
    ):
        predictions = []
        for value in (0, 1):
            changed = features.copy()
            changed[:, position[condition]] = value
            changed[:, position[interaction]] = ev * value
            predictions.append(predict(changed))
            rows.append(
                {
                    "dimension": condition.removesuffix("_indicator"),
                    "level": str(value),
                    "reference_level": "0",
                    "mean_prediction": float(np.mean(predictions[-1])),
                    "mean_difference_from_reference": 0.0,
                }
            )
        rows[-1]["mean_difference_from_reference"] = float(np.mean(predictions[1] - predictions[0]))

    categorical_sets = {
        "correlation": {
            "negative": "correlation_negative",
            "zero": "correlation_zero",
            "positive": "correlation_positive",
        },
    }
    for dimension, levels in categorical_sets.items():
        columns = [position[name] for name in levels.values()]
        predictions_by_level: dict[str, np.ndarray] = {}
        reference = next(iter(levels))
        for level, feature in levels.items():
            changed = features.copy()
            changed[:, columns] = 0.0
            changed[:, position[feature]] = 1.0
            predictions_by_level[level] = predict(changed)
        for level, values in predictions_by_level.items():
            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "reference_level": reference,
                    "mean_prediction": float(np.mean(values)),
                    "mean_difference_from_reference": float(
                        np.mean(values - predictions_by_level[reference])
                    ),
                }
            )
    return rows


def normative_case_rows(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray,
    predicted: np.ndarray,
    row_indices: np.ndarray,
    selections: list[Any],
    problems: dict[str, Any],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify validation cases relative to a configured expected-value benchmark."""

    def format_gamble(description: dict[str, Any], name: str) -> str:
        return "; ".join(
            f"p={probability:g}→{payoff:g}" for probability, payoff in description[name]
        )

    position = {name: index for index, name in enumerate(feature_names)}
    ev_a = features[:, position["expected_value_a"]]
    ev_b = features[:, position["expected_value_b_oracle"]]
    ev_difference = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    strong = settings["strong_expected_value_difference"]
    favor_b = settings["human_favor_b_threshold"]
    favor_a = settings["human_favor_a_threshold"]
    divided_width = settings["highly_divided_half_width"]
    success_error = settings["successful_prediction_max_absolute_error"]
    failure_error = settings["failed_prediction_min_absolute_error"]
    rows = []
    for position_index, row_index in enumerate(row_indices):
        record = selections[int(row_index)]
        problem_description = problems[str(int(row_index))]

        observed = float(target[position_index])
        prediction = float(predicted[position_index])
        difference = float(ev_difference[position_index])
        deviation_type = "none"
        if difference <= -strong and observed >= favor_b:
            deviation_type = "strong_ev_a_humans_favor_b"
        elif difference >= strong and observed <= favor_a:
            deviation_type = "strong_ev_b_humans_favor_a"
        is_deviation = deviation_type != "none"
        prediction_matches_human_side = (observed - 0.5) * (prediction - 0.5) > 0.0
        absolute_error = abs(prediction - observed)
        success = is_deviation and prediction_matches_human_side and absolute_error <= success_error
        failure = is_deviation and (
            not prediction_matches_human_side or absolute_error >= failure_error
        )
        rows.append(
            {
                "row_index": int(row_index),
                "problem": record.problem,
                "feedback": record.feedback,
                "ambiguity": record.amb,
                "participant_count_n": record.n,
                "ev_a": float(ev_a[position_index]),
                "ev_b_oracle": float(ev_b[position_index]),
                "ev_difference_b_minus_a_oracle": difference,
                "ev_benchmark_favors": "B"
                if difference > 0.0
                else "A"
                if difference < 0.0
                else "tie",
                "observed_brate": observed,
                "ml_predicted_brate": prediction,
                "absolute_error": absolute_error,
                "human_aggregate_favors": "B"
                if observed > 0.5
                else "A"
                if observed < 0.5
                else "tie",
                "deviation_type": deviation_type,
                "highly_divided": abs(observed - 0.5) <= divided_width,
                "ml_successfully_predicts_deviation": success,
                "ml_fails_to_predict_deviation": failure,
                "ha": record.ha,
                "pha": record.pha,
                "la": record.la,
                "hb": record.hb,
                "phb": record.phb,
                "lb": record.lb,
                "lottery_shape_b": record.lot_shape_b,
                "correlation": record.corr,
                "gamble_a": format_gamble(problem_description, "A"),
                "gamble_b": format_gamble(problem_description, "B"),
            }
        )
    return rows


def select_normative_examples(
    rows: list[dict[str, Any]], *, examples_per_category: int
) -> list[dict[str, Any]]:
    """Select deterministic, interpretable examples for each requested case type."""
    categories: dict[
        str, tuple[Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], Any]]
    ] = {
        "strong_ev_a_humans_favor_b": (
            lambda row: row["deviation_type"] == "strong_ev_a_humans_favor_b",
            lambda row: (-abs(row["ev_difference_b_minus_a_oracle"]), -row["observed_brate"]),
        ),
        "strong_ev_b_humans_favor_a": (
            lambda row: row["deviation_type"] == "strong_ev_b_humans_favor_a",
            lambda row: (-abs(row["ev_difference_b_minus_a_oracle"]), row["observed_brate"]),
        ),
        "highly_divided": (
            lambda row: row["highly_divided"],
            lambda row: (abs(row["observed_brate"] - 0.5), -row["participant_count_n"]),
        ),
        "ml_successfully_predicts_deviation": (
            lambda row: row["ml_successfully_predicts_deviation"],
            lambda row: row["absolute_error"],
        ),
        "ml_fails_to_predict_deviation": (
            lambda row: row["ml_fails_to_predict_deviation"],
            lambda row: -row["absolute_error"],
        ),
    }
    examples = []
    for category, (condition, order) in categories.items():
        matching = sorted((row for row in rows if condition(row)), key=order)
        for rank, row in enumerate(matching[:examples_per_category], start=1):
            examples.append({"example_category": category, "example_rank": rank, **row})
    return examples


def summarize_normative_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count benchmark deviations, divided choices, and model outcomes."""
    categories = {
        "strong_ev_a_humans_favor_b": lambda row: (
            row["deviation_type"] == "strong_ev_a_humans_favor_b"
        ),
        "strong_ev_b_humans_favor_a": lambda row: (
            row["deviation_type"] == "strong_ev_b_humans_favor_a"
        ),
        "highly_divided": lambda row: row["highly_divided"],
        "ml_successfully_predicts_deviation": lambda row: row["ml_successfully_predicts_deviation"],
        "ml_fails_to_predict_deviation": lambda row: row["ml_fails_to_predict_deviation"],
    }
    return {name: sum(condition(row) for row in rows) for name, condition in categories.items()}


def _plot_importance(domain_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, rows, title, limit in (
        (axes[0], domain_rows, "Coherent domain reliance", len(domain_rows)),
        (axes[1], feature_rows, "Coherent input-family reliance", len(feature_rows)),
    ):
        selected = list(reversed(rows[:limit]))
        labels = [row["name"].replace("_", " ") for row in selected]
        values = [row["mean_mae_increase"] for row in selected]
        errors = np.asarray(
            [
                [row["mean_mae_increase"] - row["group_bootstrap_ci_lower"] for row in selected],
                [row["group_bootstrap_ci_upper"] - row["mean_mae_increase"] for row in selected],
            ]
        )
        axis.barh(labels, values, xerr=errors, color="#4472C4", alpha=0.85)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Equal-problem outer OOF MAE increase")
        axis.set_title(title)
    figure.suptitle("Grouped, dependency-preserving permutation importance")
    figure.tight_layout()
    figure.savefig(DEFAULT_IMPORTANCE_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_relationships(rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (title, (feature, _)) in zip(axes.flat, RELATIONSHIP_FEATURES.items(), strict=True):
        selected = [row for row in rows if row["feature"] == feature]
        x_values = [row["feature_mean"] for row in selected]
        axis.plot(
            x_values,
            [row["predicted_brate_mean"] for row in selected],
            "o-",
            label="model prediction",
        )
        axis.plot(
            x_values,
            [row["observed_brate_mean"] for row in selected],
            "s--",
            label="observed bRate",
            alpha=0.8,
        )
        axis.set_title(title)
        axis.set_xlabel("Validation-bin feature mean")
        axis.set_ylabel("Mean bRate")
        axis.set_ylim(0.0, 1.0)
        axis.legend(fontsize=8)
    figure.suptitle("Observed predictive relationships across validation quantiles")
    figure.tight_layout()
    figure.savefig(DEFAULT_RELATIONSHIP_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_sensitivity(rows: list[dict[str, Any]]) -> None:
    dimensions = ("feedback", "ambiguity", "correlation")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, dimension in zip(axes, dimensions, strict=True):
        selected = [row for row in rows if row["dimension"] == dimension]
        axis.bar(
            [row["level"].replace("_", "\n") for row in selected],
            [row["mean_prediction"] for row in selected],
            color="#70AD47",
        )
        axis.set_ylim(0.35, 0.65)
        axis.set_ylabel("Mean prediction")
        axis.set_title(f"Model sensitivity: {dimension.replace('_', ' ')}")
    figure.suptitle("PDP-like sensitivity for coherent condition interventions")
    figure.tight_layout()
    figure.savefig(DEFAULT_SENSITIVITY_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_error_slices(rows: list[dict[str, Any]]) -> None:
    dimensions = ("feedback", "ambiguity", "expected_value_regime", "participant_count")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, dimension in zip(axes.flat, dimensions, strict=True):
        selected = [row for row in rows if row["dimension"] == dimension]
        labels = [f"{row['level'].replace('_', ' ')}\n(n={row['rows']})" for row in selected]
        axis.bar(labels, [row["problem_group_mae"] for row in selected], color="#ED7D31")
        axis.set_ylim(0.0, max(row["problem_group_mae"] for row in rows) * 1.2)
        axis.set_ylabel("Equal-problem outer OOF MAE")
        axis.set_title(dimension.replace("_", " "))
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("Selected-model error by prespecified behavioral slices")
    figure.tight_layout()
    figure.savefig(DEFAULT_ERROR_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_normative_comparison(rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """Plot observed and predicted behavior against the simple EV benchmark."""
    ev = np.asarray([row["ev_difference_b_minus_a_oracle"] for row in rows])
    observed = np.asarray([row["observed_brate"] for row in rows])
    predicted = np.asarray([row["ml_predicted_brate"] for row in rows])
    a_over_b = np.asarray([row["deviation_type"] == "strong_ev_a_humans_favor_b" for row in rows])
    b_over_a = np.asarray([row["deviation_type"] == "strong_ev_b_humans_favor_a" for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(ev, observed, s=11, alpha=0.18, color="#777777", label="all validation")
    axes[0].scatter(
        ev[a_over_b], observed[a_over_b], s=34, color="#C00000", label="EV favors A, humans B"
    )
    axes[0].scatter(
        ev[b_over_a], observed[b_over_a], s=34, color="#7030A0", label="EV favors B, humans A"
    )
    axes[0].axhline(0.5, color="black", linewidth=0.8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Oracle expected-value difference B − A")
    axes[0].set_ylabel("Observed aggregate B-choice rate")
    axes[0].set_title("Deviations from the simple EV benchmark")
    axes[0].legend(fontsize=8)

    axes[1].scatter(observed, predicted, s=11, alpha=0.18, color="#777777")
    axes[1].scatter(observed[a_over_b], predicted[a_over_b], s=34, color="#C00000")
    axes[1].scatter(observed[b_over_a], predicted[b_over_a], s=34, color="#7030A0")
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1)
    axes[1].axvline(settings["human_favor_a_threshold"], color="#999999", linestyle=":")
    axes[1].axvline(settings["human_favor_b_threshold"], color="#999999", linestyle=":")
    axes[1].set_xlabel("Observed aggregate B-choice rate")
    axes[1].set_ylabel("ML-predicted B-choice rate")
    axes[1].set_title("Prediction of benchmark-deviation cases")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    figure.suptitle("Expected-value benchmark, observed behavior, and ML prediction")
    figure.tight_layout()
    figure.savefig(DEFAULT_NORMATIVE_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_normative_examples(rows: list[dict[str, Any]]) -> None:
    """Show observed and predicted rates for selected interpretable examples."""
    panels = (
        (
            "Benchmark-deviation examples",
            {"strong_ev_a_humans_favor_b", "strong_ev_b_humans_favor_a"},
        ),
        (
            "ML success and failure examples",
            {"ml_successfully_predicts_deviation", "ml_fails_to_predict_deviation"},
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, (title, categories) in zip(axes, panels, strict=True):
        selected = [row for row in rows if row["example_category"] in categories]
        prefixes = {
            "strong_ev_a_humans_favor_b": "A→B",
            "strong_ev_b_humans_favor_a": "B→A",
            "ml_successfully_predicts_deviation": "success",
            "ml_fails_to_predict_deviation": "failure",
        }
        labels = [
            f"{prefixes[row['example_category']]}  P{row['problem']}  "
            f"ΔEV={row['ev_difference_b_minus_a_oracle']:+.1f}"
            for row in selected
        ]
        positions = np.arange(len(selected))
        observed = np.asarray([row["observed_brate"] for row in selected])
        predicted = np.asarray([row["ml_predicted_brate"] for row in selected])
        for y_value, start, end in zip(positions, observed, predicted, strict=True):
            axis.plot([start, end], [y_value, y_value], color="#999999", linewidth=1.2)
        axis.scatter(observed, positions, label="observed", color="#ED7D31", marker="s", zorder=3)
        axis.scatter(
            predicted, positions, label="ML prediction", color="#4472C4", marker="o", zorder=3
        )
        axis.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Aggregate B-choice rate")
        axis.set_title(title)
        axis.legend(fontsize=8)
    figure.suptitle("Selected cases: observed behavior versus ML prediction")
    figure.tight_layout()
    figure.savefig(DEFAULT_NORMATIVE_EXAMPLES_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty analysis table: {path}")
    _write_csv(path, list(rows[0]), rows)


def _lookup(rows: list[dict[str, Any]], dimension: str, level: str) -> dict[str, Any]:
    return next(row for row in rows if row["dimension"] == dimension and row["level"] == level)


def write_behavioral_report(statistics: dict[str, Any]) -> None:
    """Generate behavioral interpretation directly from executed statistics."""
    domain_rows = statistics["permutation_importance"]["domains"]
    feature_rows = statistics["permutation_importance"]["features"]
    slices = statistics["error_slices"]
    sensitivity = statistics["condition_sensitivity"]
    top_domains = ", ".join(
        f"{row['name'].replace('_', ' ')} ({row['mean_mae_increase']:.4f}; "
        f"{row['group_bootstrap_ci_lower']:.4f}–{row['group_bootstrap_ci_upper']:.4f})"
        for row in domain_rows[:3]
    )
    top_features = ", ".join(
        f"`{row['name']}` ({row['mean_mae_increase']:.4f}; "
        f"{row['group_bootstrap_ci_lower']:.4f}–{row['group_bootstrap_ci_upper']:.4f})"
        for row in feature_rows[:5]
    )
    feedback_delta = _lookup(sensitivity, "feedback", "1")["mean_difference_from_reference"]
    ambiguity_delta = _lookup(sensitivity, "ambiguity", "1")["mean_difference_from_reference"]
    relationships = statistics["prediction_relationships"]

    def endpoints(feature: str) -> tuple[float, float]:
        selected = [row for row in relationships if row["feature"] == feature]
        return selected[0]["predicted_brate_mean"], selected[-1]["predicted_brate_mean"]

    ev_low, ev_high = endpoints("expected_value_difference_b_minus_a_oracle")
    risk_low, risk_high = endpoints("payoff_std_difference_b_minus_a_oracle")
    best_probability_low, best_probability_high = endpoints(
        "best_payoff_probability_difference_b_minus_a_oracle"
    )
    loss_probability_low, loss_probability_high = endpoints(
        "loss_probability_difference_b_minus_a_oracle"
    )
    near_equal_error = _lookup(slices, "expected_value_regime", "near_equal_ev")[
        "problem_group_mae"
    ]
    lower_ev_error = _lookup(slices, "expected_value_regime", "b_lower_ev")["problem_group_mae"]
    higher_ev_error = _lookup(slices, "expected_value_regime", "b_higher_ev")["problem_group_mae"]
    normative = statistics["normative_benchmark"]
    normative_counts = normative["counts"]
    normative_settings = statistics["config"]["normative_benchmark"]
    report_examples = [row for row in normative["examples"] if row["example_rank"] <= 2]
    lines = [
        "# Behavioral analysis of selected-model predictions",
        "",
        (
            "This report interprets complete nested outer out-of-fold predictions. "
            "It describes predictive associations and model sensitivity, not causal effects or "
            "participant-level psychological mechanisms. No confirmatory holdout is claimed."
        ),
        "",
        "## Main findings",
        "",
        (
            f"- Grouped coherent domain perturbations ranked as {top_domains}. Parenthesized "
            "values are equal-problem outer OOF MAE increases and 95% structural-group "
            "bootstrap intervals. These overlapping domains describe model reliance, not "
            "isolated feature effects."
        ),
        (
            f"- Coherent primitive/dependency-family perturbations ranked as {top_features}. "
            "Every dependent engineered feature was rebuilt through the production feature "
            "pipeline; these are family-level reliance estimates, not individual-feature "
            "effects."
        ),
        (
            "- Coherently switching feedback on while updating its EV interaction changed the "
            f"mean prediction by {feedback_delta:+.4f}. Switching ambiguity on with its "
            f"interaction changed it by {ambiguity_delta:+.4f}. These are model-sensitivity "
            "contrasts over synthetic feature settings, not treatment effects."
        ),
        (
            "- Validation-bin plots show how predictions and observations co-vary with "
            "expected-value, probability, payoff, and risk differences. They are observational "
            "predictive relationships and retain the oracle limitation under ambiguity."
        ),
        (
            "- Across the lowest-to-highest validation quantile bins, mean prediction changed "
            f"from {ev_low:.3f} to {ev_high:.3f} for B-minus-A expected value and from "
            f"{loss_probability_low:.3f} to {loss_probability_high:.3f} for relative loss "
            "probability. Thus, higher B expected value was associated with more predicted B "
            "choice, while higher relative B loss probability was associated with less."
        ),
        (
            "- The corresponding endpoint changes were "
            f"{risk_low:.3f} to {risk_high:.3f} for relative payoff dispersion and "
            f"{best_probability_low:.3f} to {best_probability_high:.3f} for relative "
            "best-payoff probability. Both curves are non-monotone, so endpoint contrasts "
            "should not be interpreted as constant slopes."
        ),
        (
            f"- Near-equal-EV problems had MAE {near_equal_error:.4f}, versus "
            f"{lower_ev_error:.4f} when B had lower EV and {higher_ev_error:.4f} when B had "
            "higher EV. This is an error concentration, not evidence about causal difficulty."
        ),
        "",
        "## Normative benchmark versus observed behavior",
        "",
        (
            "Expected-value maximization is used here only as a simple normative benchmark. "
            f"‘Strongly favors’ means `|EV_B − EV_A| ≥ "
            f"{normative_settings['strong_expected_value_difference']:g}` payoff units; "
            f"aggregate humans favor B at `bRate ≥ "
            f"{normative_settings['human_favor_b_threshold']:.2f}` and A at `bRate ≤ "
            f"{normative_settings['human_favor_a_threshold']:.2f}`. These labels describe "
            "benchmark deviations, not irrational decisions."
        ),
        "",
        (
            f"- EV strongly favored A while aggregate humans favored B in "
            f"{normative_counts['strong_ev_a_humans_favor_b']} outer OOF rows."
        ),
        (
            f"- EV strongly favored B while aggregate humans favored A in "
            f"{normative_counts['strong_ev_b_humans_favor_a']} outer OOF rows."
        ),
        (
            f"- {normative_counts['highly_divided']} rows were highly divided, defined as "
            f"`|bRate − 0.5| ≤ {normative_settings['highly_divided_half_width']:.2f}`."
        ),
        (
            f"- The ML model successfully predicted "
            f"{normative_counts['ml_successfully_predicts_deviation']} strong deviations, "
            "meaning it predicted the observed side of 0.5 with absolute error at most "
            f"{normative_settings['successful_prediction_max_absolute_error']:.2f}. It failed "
            f"on {normative_counts['ml_fails_to_predict_deviation']}, meaning it predicted the "
            "opposite side or had absolute error at least "
            f"{normative_settings['failed_prediction_min_absolute_error']:.2f}. Intermediate "
            "cases satisfy neither label."
        ),
        "",
        "### Selected examples",
        "",
        "| Category | Problem | Gamble A | Gamble B | ΔEV B−A | Observed | ML prediction | Error |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report_examples:
        lines.append(
            f"| {row['example_category'].replace('_', ' ')} | {row['problem']} | "
            f"{row['gamble_a']} | {row['gamble_b']} | "
            f"{row['ev_difference_b_minus_a_oracle']:+.2f} | "
            f"{row['observed_brate']:.3f} | {row['ml_predicted_brate']:.3f} | "
            f"{row['absolute_error']:.3f} |"
        )
    lines.extend(
        [
            "",
            "[Normative comparison](figures/normative_vs_observed.png) and "
            "[case examples](figures/normative_case_examples.png) visualize these results. "
            "The complete case and example tables are saved as machine-readable CSV files.",
            "",
            "## Error analysis",
            "",
            "| Slice | Rows | Groups | Problem-group MAE | Condition-row MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in slices:
        lines.append(
            f"| {row['dimension'].replace('_', ' ')}: {row['level'].replace('_', ' ')} | "
            f"{row['rows']:,} | {row['structural_groups']:,} | "
            f"{row['problem_group_mae']:.4f} | {row['condition_row_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Participant-count groups use the complete development-data median (`n = "
                f"{statistics['participant_count_threshold']:g}`), not a validation-optimized "
                "cutoff. Expected-value regimes use the prespecified ±"
                f"{statistics['expected_value_near_tie_threshold']:g} payoff-unit near-tie band."
            ),
            "",
            "## Figures",
            "",
            "- [Permutation importance](figures/behavioral_permutation_importance.png)",
            "- [Predictive relationships](figures/behavioral_prediction_relationships.png)",
            "- [Condition sensitivity](figures/behavioral_condition_sensitivity.png)",
            "- [Error slices](figures/behavioral_error_slices.png)",
            "",
            "## Interpretation limits",
            "",
            (
                "- Permutation importance measures loss of predictive accuracy, not causal "
                "importance. Families overlap and must not be added or interpreted as mutually "
                "exclusive effects."
            ),
            (
                "- Donor structural groups are drawn only from the same outer fold. Complete "
                "groups move together, engineered dependencies are recomputed, and uncertainty "
                "resamples whole structural groups."
            ),
            (
                "- The feedback block perturbation preserves paired rows. Its estimate may be "
                "driven mainly by singleton groups because paired feedback/no-feedback blocks "
                "contain the same condition pattern."
            ),
            (
                "- The condition-sensitivity chart is PDP-like. It updates feedback and "
                "ambiguity interactions and the correlation one-hot family coherently, but "
                "the resulting settings are still synthetic and may be sparsely represented "
                "in the data. Lottery shape is excluded because changing its encoding without "
                "rebuilding Gamble B's outcome distribution would be internally inconsistent."
            ),
            (
                "- Validation-bin relationships combine model behavior with the observed "
                "feature distribution; they do not isolate a variable while holding every "
                "confounder fixed."
            ),
            (
                "- Ten features use design-oracle probabilities under ambiguity. Accordingly, "
                "ambiguity-related interpretation does not describe a strictly "
                "participant-visible prediction setting."
            ),
            f"- SHAP was not run: {statistics['config']['shap']['reason']}",
            (
                "- Error differences across slices may reflect sample composition and target "
                "measurement noise. They are diagnostics, not evidence that a condition causes "
                "higher error."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "All plotted values are stored under `artifacts/analysis/behavioral/`. The "
                "analysis verifies every outer-fold pipeline hash, exact feature order, "
                "complete OOF coverage, and structural-group isolation before producing "
                "outputs."
            ),
        ]
    )
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_behavioral_analysis(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run foldwise interpretation over complete nested outer OOF predictions."""
    provenance = start_run_provenance(
        experiment_name="behavioral_analysis_nested_outer_oof",
        config_paths=[config_path],
        configuration_values={
            "model_experiment": str(MODEL_SELECTION_DIR),
            "allow_dirty": allow_dirty,
        },
        dataset_manifest_path=manifest_path,
        raw_dir=raw_dir,
        fold_specification_identifier="nested_grouped_cv_v1",
        entry_module=Path(__file__),
        allow_dirty=allow_dirty,
    )
    upstream_artifacts = [
        DEFAULT_MODEL_METRICS,
        DEFAULT_MODEL_PREDICTIONS,
        DEFAULT_FEATURE_NAMES,
        DEFAULT_ENGINEERED_FEATURES,
        *sorted(DEFAULT_PIPELINE_DIR.glob("fold_*.joblib")),
    ]
    validate_upstream_artifacts(DEFAULT_MODEL_PROVENANCE, upstream_artifacts)
    config = load_behavioral_config(config_path)
    validation_summary = load_and_validate(raw_dir, manifest_path)
    model_metrics = json.loads(DEFAULT_MODEL_METRICS.read_text(encoding="utf-8"))
    feature_document = json.loads(DEFAULT_FEATURE_NAMES.read_text(encoding="utf-8"))
    feature_names = feature_document["feature_names"]
    if model_metrics["experiment_name"] != config["selected_model_experiment"]:
        raise ValueError("Behavioral config does not match the selected-model experiment")
    if model_metrics["evaluation_design"] != "nested_grouped_cv_v1":
        raise ValueError("Behavioral analysis requires nested grouped CV")
    if model_metrics["headline_source"] != "complete_outer_out_of_fold_predictions":
        raise ValueError("Behavioral analysis requires complete outer OOF predictions")
    features = _read_engineered_features(DEFAULT_ENGINEERED_FEATURES, feature_names)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    production_inputs = production_feature_inputs(selections, problems)
    recomputed_features = engineer_input_matrix(production_inputs, feature_names)
    if not np.array_equal(features, recomputed_features):
        maximum_difference = float(np.max(np.abs(features - recomputed_features)))
        raise ValueError(
            "Persisted model features differ from production feature recomputation; "
            f"maximum absolute difference={maximum_difference:.12g}"
        )
    oof = read_complete_outer_oof_predictions(
        DEFAULT_MODEL_PREDICTIONS, expected_rows=len(selections)
    )
    validation_indices = oof["row_index"]
    validation_features = features
    validation_target = oof["target"]
    validation_n = oof["participant_counts"]
    validation_groups = oof["groups"]
    outer_fold = oof["outer_fold"]
    pipelines = {}
    for fold, expected_hash in model_metrics["outer_pipeline_sha256"].items():
        path = DEFAULT_PIPELINE_DIR / f"fold_{fold}.joblib"
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Outer fold {fold} pipeline hash differs from metadata")
        pipelines[int(fold)] = joblib.load(path)

    def foldwise_predict(values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape[0], np.nan)
        for fold, pipeline in pipelines.items():
            mask = outer_fold == fold
            result[mask] = pipeline.predict(values[mask])
        if np.any(np.isnan(result)):
            raise ValueError("Foldwise prediction did not cover every outer OOF row")
        return result

    predictions = oof["predictions"]
    if not np.allclose(predictions, foldwise_predict(validation_features), atol=1e-12):
        raise ValueError("Saved OOF predictions differ from the outer-fold pipelines")

    feature_importance = grouped_permutation_importance_rows(
        foldwise_predict,
        production_inputs,
        validation_features,
        validation_target,
        validation_groups,
        outer_fold,
        feature_names,
        FEATURE_PERTURBATION_FAMILIES,
        repeats=config["permutation_repeats"],
        group_bootstrap_repeats=config["group_bootstrap_repeats"],
        confidence_level=config["permutation_confidence_level"],
        random_seed=config["random_seed"],
    )
    domain_importance = grouped_permutation_importance_rows(
        foldwise_predict,
        production_inputs,
        validation_features,
        validation_target,
        validation_groups,
        outer_fold,
        feature_names,
        DOMAIN_PERTURBATION_FAMILIES,
        repeats=config["permutation_repeats"],
        group_bootstrap_repeats=config["group_bootstrap_repeats"],
        confidence_level=config["permutation_confidence_level"],
        random_seed=config["random_seed"] + 10_000,
    )
    position = {name: index for index, name in enumerate(feature_names)}
    relationships = []
    for _, (feature, domain) in RELATIONSHIP_FEATURES.items():
        relationships.extend(
            quantile_relationship_rows(
                validation_features[:, position[feature]],
                validation_target,
                predictions,
                feature=feature,
                domain=domain,
                bins=config["relationship_quantile_bins"],
            )
        )
    participant_threshold = float(np.median(validation_n))
    slices = error_slice_rows(
        validation_features,
        feature_names,
        validation_target,
        predictions,
        validation_groups,
        validation_n,
        ev_threshold=config["expected_value_near_tie_threshold"],
        participant_threshold=participant_threshold,
    )
    sensitivity = condition_sensitivity_rows(foldwise_predict, validation_features, feature_names)
    normative_cases = normative_case_rows(
        validation_features,
        feature_names,
        validation_target,
        predictions,
        validation_indices,
        selections,
        problems,
        config["normative_benchmark"],
    )
    normative_examples = select_normative_examples(
        normative_cases,
        examples_per_category=config["normative_benchmark"]["examples_per_category"],
    )
    normative_counts = summarize_normative_cases(normative_cases)
    statistics: dict[str, Any] = {
        "analysis_name": config["analysis_name"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decisionlab_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "data_validation_status": "PASS",
        "source_commit": validation_summary["source_commit"],
        "source_sha256": validation_summary["sha256"],
        "selected_model": "inner_selected_procedure",
        "outer_pipeline_sha256": model_metrics["outer_pipeline_sha256"],
        "model_metrics_sha256": sha256_file(DEFAULT_MODEL_METRICS),
        "feature_names_sha256": sha256_file(DEFAULT_FEATURE_NAMES),
        "fold_assignments_sha256": model_metrics["fold_assignment_sha256"],
        "feature_count": len(feature_names),
        "oracle_feature_count": model_metrics["oracle_feature_count"],
        "analysis_split": "complete_nested_outer_oof",
        "validation_rows": int(validation_indices.size),
        "validation_structural_groups": len(set(validation_groups)),
        "confirmatory_holdout": False,
        "participant_count_threshold": participant_threshold,
        "participant_count_threshold_source": "development_data_median",
        "expected_value_near_tie_threshold": config["expected_value_near_tie_threshold"],
        "permutation_importance": {
            "metric": "equal_structural_problem_group_outer_oof_mae_increase",
            "scheme": config["permutation_scheme"],
            "uncertainty": "outer_fold_stratified_structural_group_bootstrap",
            "feature_table_scope": "coherent_primitive_dependency_families",
            "domain_table_scope": "overlapping_coherent_domains",
            "production_feature_recomputation_match": True,
            "features": feature_importance,
            "domains": domain_importance,
        },
        "prediction_relationships": relationships,
        "condition_sensitivity": sensitivity,
        "error_slices": slices,
        "normative_benchmark": {
            "counts": normative_counts,
            "examples": normative_examples,
            "classification_scope": "complete outer OOF rows",
        },
        "shap": config["shap"],
        "claim_scope": "predictive associations and model sensitivity; no causal claims",
    }
    _write_rows(DEFAULT_FEATURE_IMPORTANCE, feature_importance)
    _write_rows(DEFAULT_DOMAIN_IMPORTANCE, domain_importance)
    _write_rows(DEFAULT_RELATIONSHIPS, relationships)
    _write_rows(DEFAULT_SLICES, slices)
    _write_rows(DEFAULT_SENSITIVITY, sensitivity)
    _write_rows(DEFAULT_NORMATIVE_CASES, normative_cases)
    _write_rows(DEFAULT_NORMATIVE_EXAMPLES, normative_examples)
    _plot_importance(domain_importance, feature_importance)
    _plot_relationships(relationships)
    _plot_sensitivity(sensitivity)
    _plot_error_slices(slices)
    _plot_normative_comparison(normative_cases, config["normative_benchmark"])
    _plot_normative_examples(normative_examples)
    write_behavioral_report(statistics)
    output_paths = (
        DEFAULT_FEATURE_IMPORTANCE,
        DEFAULT_DOMAIN_IMPORTANCE,
        DEFAULT_RELATIONSHIPS,
        DEFAULT_SLICES,
        DEFAULT_SENSITIVITY,
        DEFAULT_NORMATIVE_CASES,
        DEFAULT_NORMATIVE_EXAMPLES,
        DEFAULT_IMPORTANCE_FIGURE,
        DEFAULT_RELATIONSHIP_FIGURE,
        DEFAULT_SENSITIVITY_FIGURE,
        DEFAULT_ERROR_FIGURE,
        DEFAULT_NORMATIVE_FIGURE,
        DEFAULT_NORMATIVE_EXAMPLES_FIGURE,
        DEFAULT_REPORT,
    )
    statistics["outputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in output_paths
    }
    statistics["implementation_sha256"] = sha256_file(Path(__file__))
    DEFAULT_STATISTICS.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_STATISTICS.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    finalize_run_provenance(
        provenance,
        fold_artifacts={
            "outer_assignments": DEFAULT_OUTER_ASSIGNMENTS,
            "inner_assignments": DEFAULT_INNER_ASSIGNMENTS,
            "fold_summary": DEFAULT_FOLD_SUMMARY,
        },
        input_artifacts=[
            DEFAULT_MODEL_PROVENANCE,
            *upstream_artifacts,
        ],
        output_artifacts=[DEFAULT_STATISTICS, *output_paths],
        output_path=DEFAULT_PROVENANCE,
    )
    return statistics


def main() -> None:
    """Run behavioral interpretation for the persisted selected model."""
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
    statistics = run_behavioral_analysis(
        args.raw_dir, args.manifest, args.config, allow_dirty=args.allow_dirty
    )
    print(f"Behavioral analysis complete: {statistics['validation_rows']:,} outer OOF rows")
    for row in statistics["permutation_importance"]["domains"]:
        print(f"{row['name']}: MAE increase={row['mean_mae_increase']:.6f}")


if __name__ == "__main__":
    main()
