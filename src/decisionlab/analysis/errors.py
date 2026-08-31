"""Systematic validation-error analysis for the selected DecisionLab model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import load_and_validate, load_problems, load_selections
from decisionlab.evaluation.metrics import (
    evaluate_brate_predictions,
    problem_group_regression_metrics,
    regression_metrics,
)
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "error_analysis.json"
MODEL_SELECTION_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "nested_model_selection"
DEFAULT_MODEL_METRICS = MODEL_SELECTION_DIR / "metrics.json"
DEFAULT_MODEL_PREDICTIONS = MODEL_SELECTION_DIR / "outer_oof_predictions.csv"
DEFAULT_FEATURE_NAMES = MODEL_SELECTION_DIR / "feature_names.json"
DEFAULT_MODEL_PROVENANCE = MODEL_SELECTION_DIR / "provenance.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "errors"
DEFAULT_STATISTICS = DEFAULT_OUTPUT_DIR / "statistics.json"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "provenance.json"
DEFAULT_LARGEST_ERRORS = DEFAULT_OUTPUT_DIR / "largest_errors.csv"
DEFAULT_REGIMES = DEFAULT_OUTPUT_DIR / "regime_metrics.csv"
DEFAULT_PARTICIPANT_COUNTS = DEFAULT_OUTPUT_DIR / "participant_count_metrics.csv"
DEFAULT_REPRESENTATIVE_CASES = DEFAULT_OUTPUT_DIR / "representative_failures.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "error_analysis.md"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_DIAGNOSTIC_FIGURE = FIGURE_DIR / "model_error_diagnostics.png"
DEFAULT_REGIME_FIGURE = FIGURE_DIR / "model_error_by_regime.png"
DEFAULT_PARTICIPANT_FIGURE = FIGURE_DIR / "model_error_by_participant_count.png"
DEFAULT_FAILURE_FIGURE = FIGURE_DIR / "representative_model_failures.png"


def load_error_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the prespecified error-analysis contract."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    required = {
        "analysis_name",
        "prediction_source",
        "selected_model_experiment",
        "random_seed",
        "cluster_bootstrap_repeats",
        "confidence_level",
        "largest_errors_to_save",
        "representative_cases_per_category",
        "near_equal_ev_absolute_difference",
        "extreme_ev_development_absolute_quantile",
        "approximately_half_half_width",
        "strong_consensus_half_width",
    }
    if set(config) != required:
        raise ValueError(f"Error-analysis config must define exactly: {sorted(required)}")
    if config["prediction_source"] != "nested_outer_oof":
        raise ValueError("Generalization error analysis requires nested outer OOF predictions")
    if config["cluster_bootstrap_repeats"] < 100:
        raise ValueError("Grouped bootstrap requires at least 100 repeats")
    if not 0.0 < config["confidence_level"] < 1.0:
        raise ValueError("Confidence level must be in (0, 1)")
    if not 0.5 < config["extreme_ev_development_absolute_quantile"] < 1.0:
        raise ValueError("Extreme-EV development quantile must exceed 0.5 and be below 1")
    if not 0.0 < config["approximately_half_half_width"] < 0.5:
        raise ValueError("50/50 half-width must be in (0, 0.5)")
    return config


def _read_saved_predictions(
    path: Path, selected_model: str, expected_indices: np.ndarray
) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or selected_model not in rows[0]:
        raise ValueError("Selected-model predictions are missing from the saved validation table")
    indices = np.asarray([int(row["row_index"]) for row in rows])
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Saved prediction rows differ from the grouped validation assignment")
    predictions = np.asarray([float(row[selected_model]) for row in rows])
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("Saved predictions fall outside [0, 1]")
    return predictions


def clustered_error_interval(
    target: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    """Bootstrap MAE and RMSE by resampling structural groups with replacement."""
    if not (target.size == predicted.size == groups.size) or target.size == 0:
        raise ValueError("Cluster-bootstrap inputs must be aligned and nonempty")
    unique_groups, group_positions = np.unique(groups, return_inverse=True)
    absolute = np.abs(predicted - target)
    squared = np.square(predicted - target)
    counts = np.bincount(group_positions).astype(float)
    absolute_sums = np.bincount(group_positions, weights=absolute)
    squared_sums = np.bincount(group_positions, weights=squared)
    group_absolute_means = absolute_sums / counts
    group_squared_means = squared_sums / counts
    rng = np.random.default_rng(random_seed)
    mae_values = np.empty(repeats)
    rmse_values = np.empty(repeats)
    for repeat in range(repeats):
        sampled = rng.integers(0, unique_groups.size, size=unique_groups.size)
        mae_values[repeat] = float(np.mean(group_absolute_means[sampled]))
        rmse_values[repeat] = math.sqrt(float(np.mean(group_squared_means[sampled])))
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mae_ci_lower": float(np.quantile(mae_values, tail)),
        "mae_ci_upper": float(np.quantile(mae_values, 1.0 - tail)),
        "rmse_ci_lower": float(np.quantile(rmse_values, tail)),
        "rmse_ci_upper": float(np.quantile(rmse_values, 1.0 - tail)),
    }


def regime_masks(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray,
    participant_counts: np.ndarray,
    *,
    extreme_ev_threshold: float,
    near_ev_threshold: float,
    half_half_width: float,
    consensus_width: float,
    participant_median: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Create prespecified, interpretable validation regimes."""
    position = {name: index for index, name in enumerate(feature_names)}
    feedback = features[:, position["feedback_indicator"]].astype(bool)
    ambiguity = features[:, position["ambiguity_indicator"]].astype(bool)
    ev = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    distance_from_half = np.abs(target - 0.5)
    shape_names = {
        "single_outcome": "lottery_shape_b_undefined",
        "symmetric": "lottery_shape_b_symmetric",
        "right_skewed": "lottery_shape_b_right_skewed",
        "left_skewed": "lottery_shape_b_left_skewed",
    }
    return {
        "feedback": {"no_feedback": ~feedback, "feedback": feedback},
        "ambiguity": {"known_probabilities": ~ambiguity, "ambiguous_b": ambiguity},
        "expected_value_regime": {
            "extreme_a_advantage": ev <= -extreme_ev_threshold,
            "moderate_a_advantage": (ev > -extreme_ev_threshold) & (ev < -near_ev_threshold),
            "near_equal_ev": np.abs(ev) <= near_ev_threshold,
            "moderate_b_advantage": (ev > near_ev_threshold) & (ev < extreme_ev_threshold),
            "extreme_b_advantage": ev >= extreme_ev_threshold,
        },
        "aggregate_division": {
            "approximately_50_50": distance_from_half <= half_half_width,
            "leaning": (distance_from_half > half_half_width)
            & (distance_from_half < consensus_width),
            "strong_consensus": distance_from_half >= consensus_width,
        },
        "participant_count": {
            f"n_at_or_below_{participant_median:g}": participant_counts <= participant_median,
            f"n_above_{participant_median:g}": participant_counts > participant_median,
        },
        "lottery_shape_b": {
            level: features[:, position[feature]].astype(bool)
            for level, feature in shape_names.items()
        },
    }


def calculate_regime_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    masks: dict[str, dict[str, np.ndarray]],
    *,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Calculate error and structural-group bootstrap intervals for every regime."""
    rows = []
    regime_index = 0
    for dimension, levels in masks.items():
        for level, mask in levels.items():
            if not np.any(mask):
                raise ValueError(f"Empty error regime: {dimension}/{level}")
            primary = problem_group_regression_metrics(
                target[mask], predicted[mask], groups[mask], include_r2=False
            )
            condition_row = regression_metrics(target[mask], predicted[mask], include_r2=False)
            interval = clustered_error_interval(
                target[mask],
                predicted[mask],
                groups[mask],
                repeats=repeats,
                confidence_level=confidence_level,
                random_seed=random_seed + regime_index,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "rows": int(np.sum(mask)),
                    "structural_groups": int(np.unique(groups[mask]).size),
                    "problem_group_mae": primary["mae"],
                    "problem_group_rmse": primary["rmse"],
                    "problem_group_mean_bias": primary["mean_bias"],
                    "condition_row_mae": condition_row["mae"],
                    "condition_row_rmse": condition_row["rmse"],
                    "condition_row_mean_bias": condition_row["mean_bias"],
                    **interval,
                }
            )
            regime_index += 1
    return rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def participant_count_analysis(
    participant_counts: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Summarize errors by exact n and calculate row-level error correlations."""
    absolute_error = np.abs(predicted - target)
    rows = []
    for offset, count in enumerate(sorted(np.unique(participant_counts))):
        mask = participant_counts == count
        primary = problem_group_regression_metrics(
            target[mask], predicted[mask], groups[mask], include_r2=False
        )
        condition_row = regression_metrics(target[mask], predicted[mask], include_r2=False)
        rows.append(
            {
                "participant_count_n": int(count),
                "rows": int(np.sum(mask)),
                "structural_groups": int(np.unique(groups[mask]).size),
                "problem_group_mae": primary["mae"],
                "problem_group_rmse": primary["rmse"],
                "problem_group_mean_bias": primary["mean_bias"],
                "condition_row_mae": condition_row["mae"],
                "condition_row_rmse": condition_row["rmse"],
                "condition_row_mean_bias": condition_row["mean_bias"],
                **clustered_error_interval(
                    target[mask],
                    predicted[mask],
                    groups[mask],
                    repeats=repeats,
                    confidence_level=confidence_level,
                    random_seed=random_seed + offset,
                ),
            }
        )
    return rows, {
        "pearson_n_vs_absolute_error": float(np.corrcoef(participant_counts, absolute_error)[0, 1]),
        "spearman_n_vs_absolute_error": float(
            np.corrcoef(_average_ranks(participant_counts), _average_ranks(absolute_error))[0, 1]
        ),
    }


def _format_gamble(problem: dict[str, Any], name: str) -> str:
    return "; ".join(f"p={probability:g}→{payoff:g}" for probability, payoff in problem[name])


def _possible_modeling_considerations(
    *,
    record: Any,
    target: float,
    ev_difference: float,
    extreme_ev_threshold: float,
    near_ev_threshold: float,
    participant_median: float,
) -> str:
    considerations = []
    if record.amb:
        considerations.append(
            "Ambiguous B: the model uses design-oracle probabilities that participants did not see"
        )
    if record.feedback:
        considerations.append(
            "Feedback condition: a binary indicator cannot represent realized experience histories"
        )
    if abs(ev_difference) >= extreme_ev_threshold:
        considerations.append("Expected-value difference lies in the training-distribution tail")
    if abs(ev_difference) <= near_ev_threshold:
        considerations.append(
            "Expected values are close, so prediction depends on other represented structure"
        )
    if (ev_difference > 0.0 and target < 0.5) or (ev_difference < 0.0 and target > 0.5):
        considerations.append("Aggregate choice is on the opposite side of the simple EV benchmark")
    if abs(target - 0.5) <= 0.05:
        considerations.append("Aggregate choices are nearly evenly divided")
    if record.n <= participant_median:
        considerations.append(
            "The aggregate target is based on a relatively small participant count"
        )
    if not considerations:
        considerations.append(
            "The case may depend on nonlinear feature combinations not isolated here"
        )
    return "; ".join(considerations)


def build_case_rows(
    row_indices: np.ndarray,
    selections: list[Any],
    problems: dict[str, Any],
    features: np.ndarray,
    feature_names: list[str],
    predicted: np.ndarray,
    *,
    extreme_ev_threshold: float,
    near_ev_threshold: float,
    participant_median: float,
) -> list[dict[str, Any]]:
    """Reconstruct validation problems and attach prediction-error context."""
    position = {name: index for index, name in enumerate(feature_names)}
    ev_a = features[:, position["expected_value_a"]]
    ev_b = features[:, position["expected_value_b_oracle"]]
    ev_difference = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    rows = []
    for local_index, row_index in enumerate(row_indices):
        record = selections[int(row_index)]
        actual = record.brate
        prediction = float(predicted[local_index])
        difference = float(ev_difference[local_index])
        rows.append(
            {
                "row_index": int(row_index),
                "problem": record.problem,
                "feedback": record.feedback,
                "ambiguity": record.amb,
                "participant_count_n": record.n,
                "gamble_a": _format_gamble(problems[str(int(row_index))], "A"),
                "gamble_b": _format_gamble(problems[str(int(row_index))], "B"),
                "expected_value_a": float(ev_a[local_index]),
                "expected_value_b_oracle": float(ev_b[local_index]),
                "expected_value_difference_b_minus_a_oracle": difference,
                "expected_value_benchmark": (
                    "B" if difference > 0.0 else "A" if difference < 0.0 else "tie"
                ),
                "actual_brate": actual,
                "predicted_brate": prediction,
                "residual_predicted_minus_actual": prediction - actual,
                "absolute_error": abs(prediction - actual),
                "possible_modeling_considerations": _possible_modeling_considerations(
                    record=record,
                    target=actual,
                    ev_difference=difference,
                    extreme_ev_threshold=extreme_ev_threshold,
                    near_ev_threshold=near_ev_threshold,
                    participant_median=participant_median,
                ),
            }
        )
    return rows


def select_representative_failures(
    cases: list[dict[str, Any]],
    *,
    per_category: int,
    extreme_ev_threshold: float,
    half_half_width: float,
) -> list[dict[str, Any]]:
    """Select highest-error cases within each requested behavioral category."""
    categories = {
        "largest_overall": lambda row: True,
        "ambiguity": lambda row: row["ambiguity"],
        "feedback": lambda row: row["feedback"],
        "extreme_expected_value": lambda row: (
            abs(row["expected_value_difference_b_minus_a_oracle"]) >= extreme_ev_threshold
        ),
        "approximately_50_50": lambda row: abs(row["actual_brate"] - 0.5) <= half_half_width,
    }
    selected = []
    for category, condition in categories.items():
        matching = sorted(
            (row for row in cases if condition(row)),
            key=lambda row: (-row["absolute_error"], row["row_index"]),
        )
        for rank, row in enumerate(matching[:per_category], start=1):
            selected.append({"case_category": category, "category_rank": rank, **row})
    return selected


def _plot_diagnostics(target: np.ndarray, predicted: np.ndarray) -> None:
    residual = predicted - target
    absolute = np.abs(residual)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = axes[0].scatter(target, predicted, c=absolute, cmap="magma", s=16, alpha=0.65)
    axes[0].plot([0.0, 1.0], [0.0, 1.0], "--", color="black", linewidth=1)
    axes[0].set(xlabel="Actual bRate", ylabel="Predicted bRate", title="Actual vs predicted")
    figure.colorbar(scatter, ax=axes[0], label="Absolute error")
    axes[1].scatter(predicted, residual, s=14, alpha=0.35, color="#4472C4")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Predicted bRate",
        ylabel="Residual (predicted − actual)",
        title="Residual pattern",
    )
    figure.suptitle("Selected-model validation errors")
    figure.tight_layout()
    figure.savefig(DEFAULT_DIAGNOSTIC_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_regimes(rows: list[dict[str, Any]]) -> None:
    dimensions = ("feedback", "ambiguity", "expected_value_regime", "aggregate_division")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, dimension in zip(axes.flat, dimensions, strict=True):
        selected = [row for row in rows if row["dimension"] == dimension]
        values = np.asarray([row["problem_group_mae"] for row in selected])
        lower = values - np.asarray([row["mae_ci_lower"] for row in selected])
        upper = np.asarray([row["mae_ci_upper"] for row in selected]) - values
        positions = np.arange(len(selected))
        labels = [f"{row['level'].replace('_', ' ')} (n={row['rows']})" for row in selected]
        axis.barh(positions, values, color="#ED7D31", alpha=0.9)
        axis.errorbar(
            values,
            positions,
            xerr=[lower, upper],
            fmt="none",
            color="black",
        )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(dimension.replace("_", " "))
        axis.set_xlabel("Equal-problem outer OOF MAE")
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle("Error by behavioral regime (95% structural-group bootstrap intervals)")
    figure.tight_layout()
    figure.savefig(DEFAULT_REGIME_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_participant_counts(rows: list[dict[str, Any]]) -> None:
    counts = np.asarray([row["participant_count_n"] for row in rows])
    mae = np.asarray([row["problem_group_mae"] for row in rows])
    lower = mae - np.asarray([row["mae_ci_lower"] for row in rows])
    upper = np.asarray([row["mae_ci_upper"] for row in rows]) - mae
    sample_rows = np.asarray([row["rows"] for row in rows])
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        counts,
        mae,
        yerr=[lower, upper],
        marker="o",
        linestyle="none",
        color="#4472C4",
        capsize=3,
    )
    axis.set_xlabel("Participant count n")
    axis.set_ylabel("Equal-problem outer OOF MAE")
    axis.set_title("Error and validation support by participant count")
    support_axis = axis.twinx()
    support_axis.bar(counts, sample_rows, alpha=0.16, color="#777777")
    support_axis.set_ylabel("Validation rows")
    figure.tight_layout()
    figure.savefig(DEFAULT_PARTICIPANT_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_failures(rows: list[dict[str, Any]]) -> None:
    selected = rows[:15]
    positions = np.arange(len(selected))
    actual = np.asarray([row["actual_brate"] for row in selected])
    predicted = np.asarray([row["predicted_brate"] for row in selected])
    labels = [
        f"P{row['problem']} · ΔEV={row['expected_value_difference_b_minus_a_oracle']:+.1f}"
        for row in selected
    ]
    figure, axis = plt.subplots(figsize=(10, 7))
    for y_value, start, end in zip(positions, actual, predicted, strict=True):
        axis.plot([start, end], [y_value, y_value], color="#999999", linewidth=1.2)
    axis.scatter(actual, positions, marker="s", color="#ED7D31", label="actual bRate", zorder=3)
    axis.scatter(predicted, positions, marker="o", color="#4472C4", label="prediction", zorder=3)
    axis.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Aggregate B-choice rate")
    axis.set_title("Largest absolute validation errors")
    axis.legend()
    figure.tight_layout()
    figure.savefig(DEFAULT_FAILURE_FIGURE, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty error-analysis table: {path}")
    _write_csv(path, list(rows[0]), rows)


def write_error_report(statistics: dict[str, Any]) -> None:
    """Generate the systematic error report from executed analysis artifacts."""
    overall = statistics["overall_metrics"]
    overall_primary = overall["problem_group_equal_weighted"]
    overall_condition_row = overall["condition_row_unweighted"]
    counts = statistics["special_case_counts"]
    correlations = statistics["participant_count_relationship"]
    config = statistics["config"]
    largest = statistics["largest_errors"][:10]
    regimes = statistics["regime_metrics"]
    representatives = [
        row for row in statistics["representative_failures"] if row["category_rank"] == 1
    ]
    highest_regimes = sorted(regimes, key=lambda row: row["problem_group_mae"], reverse=True)[:5]
    consistently_higher = [
        row
        for row in regimes
        if row["problem_group_mae"] > overall_primary["mae"]
        and row["problem_group_rmse"] > overall_primary["rmse"]
    ]
    lines = [
        "# Systematic model error analysis",
        "",
        (
            "This report analyzes complete nested outer out-of-fold predictions. "
            "No partition is described as a confirmatory holdout. Differences are descriptive "
            "predictive "
            "patterns, not causal effects or explanations of participant behavior."
        ),
        "",
        "## Summary",
        "",
        (
            f"Primary equal-structural-group outer OOF MAE was "
            f"**{overall_primary['mae']:.4f}** across "
            f"{statistics['validation_structural_groups']:,} groups. Secondary condition-row "
            f"MAE was **{overall_condition_row['mae']:.4f}** across "
            f"{statistics['validation_rows']:,} rows."
        ),
        (
            f"The absolute EV threshold for an extreme case was "
            f"**{statistics['extreme_ev_threshold']:.3f}**, calculated as the configured "
            f"{config['extreme_ev_development_absolute_quantile']:.0%} quantile using "
            "development predictors."
        ),
        "",
        "The five regimes with the largest descriptive MAE were: "
        + ", ".join(
            f"{row['dimension'].replace('_', ' ')}/{row['level'].replace('_', ' ')} "
            f"({row['problem_group_mae']:.4f})"
            for row in highest_regimes
        )
        + ". Overlapping bootstrap intervals mean small differences should not be treated as "
        "clear population differences.",
        "",
        (
            "Regimes above the overall equal-problem error on both MAE and RMSE were: "
            + ", ".join(
                f"{row['dimension'].replace('_', ' ')}/{row['level'].replace('_', ' ')}"
                for row in consistently_higher
            )
            + ". These overlapping slices are descriptive diagnostics, not independent "
            "effects."
        ),
        "",
        "## Largest absolute errors",
        "",
        (
            "| Problem | Conditions | Gamble A | Gamble B | Actual | Predicted | "
            "Abs. error | EV benchmark | ΔEV B−A |"
        ),
        "|---:|---|---|---|---:|---:|---:|:---:|---:|",
    ]
    for row in largest:
        conditions = ", ".join(
            [
                "feedback" if row["feedback"] else "no feedback",
                "ambiguous" if row["ambiguity"] else "known",
            ]
        )
        lines.append(
            f"| {row['problem']} | {conditions} | {row['gamble_a']} | {row['gamble_b']} | "
            f"{row['actual_brate']:.3f} | {row['predicted_brate']:.3f} | "
            f"{row['absolute_error']:.3f} | {row['expected_value_benchmark']} | "
            f"{row['expected_value_difference_b_minus_a_oracle']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral regimes",
            "",
            (
                "Intervals below resample structural groups and give every sampled group equal "
                "total weight within each behavioral slice."
            ),
            "",
            "| Dimension | Regime | Rows | Groups | Problem-group MAE | 95% MAE interval | "
            "Condition-row MAE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regimes:
        lines.append(
            f"| {row['dimension'].replace('_', ' ')} | {row['level'].replace('_', ' ')} | "
            f"{row['rows']:,} | {row['structural_groups']:,} | "
            f"{row['problem_group_mae']:.4f} | "
            f"[{row['mae_ci_lower']:.4f}, {row['mae_ci_upper']:.4f}] | "
            f"{row['condition_row_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Requested special cases",
            "",
            (
                f"- Ambiguity: {counts['ambiguity_rows']:,} rows; feedback: "
                f"{counts['feedback_rows']:,} rows."
            ),
            (
                f"- Extreme EV difference: {counts['extreme_ev_rows']:,} rows using "
                "the training-derived threshold above."
            ),
            (
                f"- Approximately 50/50 human choice: "
                f"{counts['approximately_50_50_rows']:,} rows with "
                f"`|bRate − 0.5| ≤ {config['approximately_half_half_width']:.2f}`."
            ),
            (
                f"- Participant count had Pearson correlation "
                f"{correlations['pearson_n_vs_absolute_error']:+.3f} and Spearman correlation "
                f"{correlations['spearman_n_vs_absolute_error']:+.3f} with absolute error. "
                "These row-level associations are descriptive and the available n range is narrow."
            ),
            "",
            "## Representative failures and possible modeling considerations",
            "",
            (
                "The considerations below are generated from observed case properties. "
                "They are plausible limitations to investigate, not causal explanations."
            ),
            "",
        ]
    )
    for row in representatives:
        lines.extend(
            [
                f"### {row['case_category'].replace('_', ' ').title()}: problem {row['problem']}",
                "",
                f"- Conditions: {'feedback' if row['feedback'] else 'no feedback'}; "
                f"{'ambiguous B' if row['ambiguity'] else 'known probabilities'}; "
                f"n = {row['participant_count_n']}.",
                f"- Gamble A: {row['gamble_a']} (EV = {row['expected_value_a']:.3f}).",
                f"- Gamble B: {row['gamble_b']} "
                f"(oracle EV = {row['expected_value_b_oracle']:.3f}).",
                f"- Observed bRate = **{row['actual_brate']:.3f}**; predicted bRate = "
                f"**{row['predicted_brate']:.3f}**; absolute error = "
                f"**{row['absolute_error']:.3f}**.",
                f"- Simple EV benchmark: **{row['expected_value_benchmark']}** "
                f"(B − A = {row['expected_value_difference_b_minus_a_oracle']:+.3f}).",
                f"- Possible modeling considerations: {row['possible_modeling_considerations']}.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This post-selection analysis uses complete nested outer out-of-fold predictions. "
            "It provides exploratory generalization diagnostics, not independent confirmatory "
            "evidence.",
            "- Regimes overlap, so their error differences are not independent effects. The "
            "bootstrap intervals quantify group-resampling variation, not causal uncertainty.",
            "- The extreme-EV threshold uses raw payoff units and is therefore scale-dependent.",
            "- Some exact participant-count levels have few rows; their estimates and intervals "
            "are correspondingly unstable.",
            "- bRate is an aggregate estimate. Its sampling variability can contribute to "
            "observed prediction error, especially for smaller n, without identifying why a "
            "specific group chose as it did.",
            "",
            "## Figures and artifacts",
            "",
            "- [Residual diagnostics](figures/model_error_diagnostics.png)",
            "- [Error by behavioral regime](figures/model_error_by_regime.png)",
            "- [Error by participant count](figures/model_error_by_participant_count.png)",
            "- [Largest failures](figures/representative_model_failures.png)",
            "",
            (
                "Complete machine-readable tables are stored in "
                "`artifacts/analysis/errors/`. The report uses outer OOF errors from the "
                "nested selection procedure and is not untouched confirmatory evidence."
            ),
        ]
    )
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Execute systematic error analysis from complete nested outer OOF predictions."""
    provenance = start_run_provenance(
        experiment_name="error_analysis_nested_outer_oof",
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
    ]
    validate_upstream_artifacts(DEFAULT_MODEL_PROVENANCE, upstream_artifacts)
    config = load_error_config(config_path)
    validation_summary = load_and_validate(raw_dir, manifest_path)
    model_metrics = json.loads(DEFAULT_MODEL_METRICS.read_text(encoding="utf-8"))
    feature_document = json.loads(DEFAULT_FEATURE_NAMES.read_text(encoding="utf-8"))
    if model_metrics["experiment_name"] != config["selected_model_experiment"]:
        raise ValueError("Error config refers to a different model-selection experiment")
    if model_metrics["evaluation_design"] != "nested_grouped_cv_v1":
        raise ValueError("Error analysis requires a nested grouped-CV experiment")
    if model_metrics["headline_source"] != "complete_outer_out_of_fold_predictions":
        raise ValueError("Error analysis requires complete outer OOF predictions")

    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    feature_names = feature_document["feature_names"]
    features = _read_engineered_features(DEFAULT_ENGINEERED_FEATURES, feature_names)
    oof = read_complete_outer_oof_predictions(
        DEFAULT_MODEL_PREDICTIONS,
        expected_rows=len(selections),
    )
    selected_model = "inner_selected_procedure"
    validation_indices = oof["row_index"]
    predictions = oof["predictions"]
    target = oof["target"]
    participant_counts = oof["participant_counts"]
    groups = oof["groups"]
    validation_features = features
    position = {name: index for index, name in enumerate(feature_names)}
    training_ev = features[:, position["expected_value_difference_b_minus_a_oracle"]]
    extreme_ev_threshold = float(
        np.quantile(np.abs(training_ev), config["extreme_ev_development_absolute_quantile"])
    )
    participant_median = float(np.median([row.n for row in selections]))
    masks = regime_masks(
        validation_features,
        feature_names,
        target,
        participant_counts,
        extreme_ev_threshold=extreme_ev_threshold,
        near_ev_threshold=config["near_equal_ev_absolute_difference"],
        half_half_width=config["approximately_half_half_width"],
        consensus_width=config["strong_consensus_half_width"],
        participant_median=participant_median,
    )
    regime_rows = calculate_regime_metrics(
        target,
        predictions,
        groups,
        masks,
        repeats=config["cluster_bootstrap_repeats"],
        confidence_level=config["confidence_level"],
        random_seed=config["random_seed"],
    )
    participant_rows, participant_relationship = participant_count_analysis(
        participant_counts,
        target,
        predictions,
        groups,
        repeats=config["cluster_bootstrap_repeats"],
        confidence_level=config["confidence_level"],
        random_seed=config["random_seed"] + 20_000,
    )
    cases = build_case_rows(
        validation_indices,
        selections,
        problems,
        validation_features,
        feature_names,
        predictions,
        extreme_ev_threshold=extreme_ev_threshold,
        near_ev_threshold=config["near_equal_ev_absolute_difference"],
        participant_median=participant_median,
    )
    largest = sorted(cases, key=lambda row: (-row["absolute_error"], row["row_index"]))[
        : config["largest_errors_to_save"]
    ]
    representatives = select_representative_failures(
        cases,
        per_category=config["representative_cases_per_category"],
        extreme_ev_threshold=extreme_ev_threshold,
        half_half_width=config["approximately_half_half_width"],
    )
    overall = evaluate_brate_predictions(target, predictions, groups, participant_counts)
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
        "selected_model": selected_model,
        "model_metrics_sha256": sha256_file(DEFAULT_MODEL_METRICS),
        "model_predictions_sha256": sha256_file(DEFAULT_MODEL_PREDICTIONS),
        "fold_assignments_sha256": model_metrics["fold_assignment_sha256"],
        "analysis_split": "complete_nested_outer_oof",
        "validation_rows": int(validation_indices.size),
        "validation_structural_groups": int(np.unique(groups).size),
        "test_rows_predicted": 0,
        "test_metrics_computed": False,
        "overall_metrics": overall,
        "extreme_ev_threshold": extreme_ev_threshold,
        "extreme_ev_threshold_source": "development-data absolute EV-difference quantile",
        "participant_count_median": participant_median,
        "participant_count_median_source": "development rows",
        "participant_count_relationship": participant_relationship,
        "special_case_counts": {
            "ambiguity_rows": int(np.sum(masks["ambiguity"]["ambiguous_b"])),
            "feedback_rows": int(np.sum(masks["feedback"]["feedback"])),
            "extreme_ev_rows": int(
                np.sum(
                    np.abs(
                        validation_features[
                            :, position["expected_value_difference_b_minus_a_oracle"]
                        ]
                    )
                    >= extreme_ev_threshold
                )
            ),
            "approximately_50_50_rows": int(
                np.sum(masks["aggregate_division"]["approximately_50_50"])
            ),
        },
        "regime_metrics": regime_rows,
        "participant_count_metrics": participant_rows,
        "largest_errors": largest,
        "representative_failures": representatives,
        "claim_scope": "descriptive predictive error patterns; no causal claims",
    }
    _write_rows(DEFAULT_LARGEST_ERRORS, largest)
    _write_rows(DEFAULT_REGIMES, regime_rows)
    _write_rows(DEFAULT_PARTICIPANT_COUNTS, participant_rows)
    _write_rows(DEFAULT_REPRESENTATIVE_CASES, representatives)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _plot_diagnostics(target, predictions)
    _plot_regimes(regime_rows)
    _plot_participant_counts(participant_rows)
    _plot_failures(largest)
    write_error_report(statistics)
    output_paths = (
        DEFAULT_LARGEST_ERRORS,
        DEFAULT_REGIMES,
        DEFAULT_PARTICIPANT_COUNTS,
        DEFAULT_REPRESENTATIVE_CASES,
        DEFAULT_DIAGNOSTIC_FIGURE,
        DEFAULT_REGIME_FIGURE,
        DEFAULT_PARTICIPANT_FIGURE,
        DEFAULT_FAILURE_FIGURE,
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
    """Run the selected-model systematic error analysis."""
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
    statistics = run_error_analysis(
        args.raw_dir, args.manifest, args.config, allow_dirty=args.allow_dirty
    )
    print(
        f"Error analysis complete: {statistics['validation_rows']:,} outer OOF rows; "
        "problem-group MAE="
        f"{statistics['overall_metrics']['problem_group_equal_weighted']['mae']:.6f}"
    )
    print(
        "Largest absolute error: "
        f"{statistics['largest_errors'][0]['absolute_error']:.6f} "
        f"(Problem {statistics['largest_errors'][0]['problem']})"
    )


if __name__ == "__main__":
    main()
