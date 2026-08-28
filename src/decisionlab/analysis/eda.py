"""Generate reproducible, question-driven exploratory analysis for choices13k."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST
from decisionlab.data.validation import (
    SelectionRecord,
    load_and_validate,
    load_problems,
    load_selections,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLOT_CACHE = Path(tempfile.gettempdir()) / "decisionlab-matplotlib-cache"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eda.json"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "figures"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "eda_summary.md"
DEFAULT_STATISTICS = PROJECT_ROOT / "artifacts" / "manifests" / "eda_statistics.json"

COLORS = {
    "blue": "#2878B5",
    "orange": "#E07A3F",
    "green": "#3A923A",
    "purple": "#7A5195",
    "gray": "#667085",
}

LOTTERY_SHAPE_LABELS = {
    0: "Undefined\n(one outcome)",
    1: "Symmetric",
    2: "Right-skewed",
    3: "Left-skewed",
}


def load_eda_config(path: Path = DEFAULT_CONFIG) -> dict[str, int]:
    """Load and validate deterministic plotting settings."""
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    expected = {"random_seed", "quantile_bins", "figure_dpi"}
    if set(config) != expected or not all(isinstance(config[key], int) for key in expected):
        raise ValueError(f"EDA config must define integer settings: {sorted(expected)}")
    if config["quantile_bins"] < 4 or config["figure_dpi"] < 72:
        raise ValueError("EDA bin count or figure DPI is too small")
    return config


def _option_moments(outcomes: list[list[float]]) -> tuple[float, float]:
    probabilities = np.asarray([outcome[0] for outcome in outcomes], dtype=float)
    payoffs = np.asarray([outcome[1] for outcome in outcomes], dtype=float)
    mean = float(np.sum(probabilities * payoffs))
    variance = float(np.sum(probabilities * np.square(payoffs - mean)))
    return mean, variance


def build_analysis_arrays(
    selections: list[SelectionRecord], problems: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Convert validated records into analysis arrays and derive gamble moments."""
    arrays: dict[str, np.ndarray] = {
        "problem": np.asarray([record.problem for record in selections], dtype=int),
        "feedback": np.asarray([record.feedback for record in selections], dtype=bool),
        "n": np.asarray([record.n for record in selections], dtype=int),
        "block": np.asarray([record.block for record in selections], dtype=int),
        "ha": np.asarray([record.ha for record in selections], dtype=float),
        "pha": np.asarray([record.pha for record in selections], dtype=float),
        "la": np.asarray([record.la for record in selections], dtype=float),
        "hb": np.asarray([record.hb for record in selections], dtype=float),
        "phb": np.asarray([record.phb for record in selections], dtype=float),
        "lb": np.asarray([record.lb for record in selections], dtype=float),
        "lot_shape_b": np.asarray([record.lot_shape_b for record in selections], dtype=int),
        "lot_num_b": np.asarray([record.lot_num_b for record in selections], dtype=int),
        "amb": np.asarray([record.amb for record in selections], dtype=bool),
        "corr": np.asarray([record.corr for record in selections], dtype=int),
        "brate": np.asarray([record.brate for record in selections], dtype=float),
        "brate_std": np.asarray([record.brate_std for record in selections], dtype=float),
    }
    ev_a: list[float] = []
    ev_b: list[float] = []
    variance_a: list[float] = []
    variance_b: list[float] = []
    for index in range(len(selections)):
        mean_a, var_a = _option_moments(problems[str(index)]["A"])
        mean_b, var_b = _option_moments(problems[str(index)]["B"])
        ev_a.append(mean_a)
        ev_b.append(mean_b)
        variance_a.append(var_a)
        variance_b.append(var_b)
    arrays["ev_a"] = np.asarray(ev_a)
    arrays["ev_b"] = np.asarray(ev_b)
    arrays["ev_diff_b_minus_a"] = arrays["ev_b"] - arrays["ev_a"]
    arrays["variance_a"] = np.asarray(variance_a)
    arrays["variance_b"] = np.asarray(variance_b)
    arrays["variance_diff_b_minus_a"] = arrays["variance_b"] - arrays["variance_a"]
    arrays["estimated_brate_se"] = arrays["brate_std"] / np.sqrt(arrays["n"])
    return arrays


def _describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size < 2 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _group_brate(
    arrays: dict[str, np.ndarray], key: str, labels: dict[Any, str]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for value, label in labels.items():
        selected = arrays["brate"][arrays[key] == value]
        result[label] = {
            "count": int(selected.size),
            "mean_bRate": _mean_or_nan(selected),
            "median_bRate": float(np.median(selected)) if selected.size else math.nan,
        }
    return result


def _paired_feedback_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    by_problem: dict[int, dict[bool, float]] = defaultdict(dict)
    for problem, feedback, brate in zip(
        arrays["problem"], arrays["feedback"], arrays["brate"], strict=True
    ):
        by_problem[int(problem)][bool(feedback)] = float(brate)
    paired = [values for values in by_problem.values() if set(values) == {False, True}]
    no_feedback = np.asarray([values[False] for values in paired])
    feedback = np.asarray([values[True] for values in paired])
    differences = feedback - no_feedback
    return {
        "pair_count": int(differences.size),
        "no_feedback_mean_bRate": float(np.mean(no_feedback)),
        "feedback_mean_bRate": float(np.mean(feedback)),
        "difference_feedback_minus_no_feedback": _describe(differences),
        "proportion_positive": float(np.mean(differences > 0)),
        "proportion_negative": float(np.mean(differences < 0)),
        "proportion_zero": float(np.mean(differences == 0)),
        "paired_bRate_correlation": _pearson(no_feedback, feedback),
    }


def compute_eda_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute descriptive statistics used by both plots and the report."""
    brate = arrays["brate"]
    ev_diff = arrays["ev_diff_b_minus_a"]
    known = ~arrays["amb"]
    ambiguous = arrays["amb"]

    ev_relationship: dict[str, Any] = {}
    for label, mask in {
        "all": np.ones(brate.size, dtype=bool),
        "known": known,
        "ambiguous": ambiguous,
    }.items():
        group_ev = ev_diff[mask]
        group_brate = brate[mask]
        ev_relationship[label] = {
            "count": int(mask.sum()),
            "pearson_ev_difference_vs_bRate": _pearson(group_ev, group_brate),
            "mean_bRate_when_ev_b_higher": _mean_or_nan(group_brate[group_ev > 0]),
            "mean_bRate_when_ev_a_higher": _mean_or_nan(group_brate[group_ev < 0]),
            "mean_bRate_when_ev_equal": (
                float(np.mean(group_brate[group_ev == 0])) if np.any(group_ev == 0) else math.nan
            ),
        }

    primitive_relationships = {
        name: _pearson(values, brate)
        for name, values in {
            "Ha": arrays["ha"],
            "pHa": arrays["pha"],
            "La": arrays["la"],
            "Hb": arrays["hb"],
            "pHb": arrays["phb"],
            "Lb": arrays["lb"],
            "EV_A": arrays["ev_a"],
            "EV_B": arrays["ev_b"],
            "variance_A": arrays["variance_a"],
            "variance_B": arrays["variance_b"],
            "variance_difference_B_minus_A": arrays["variance_diff_b_minus_a"],
        }.items()
    }
    return {
        "target_bRate": {
            **_describe(brate),
            "zero_count": int(np.sum(brate == 0)),
            "one_count": int(np.sum(brate == 1)),
        },
        "sample_size_n": {
            **_describe(arrays["n"]),
            "unique_values": int(np.unique(arrays["n"]).size),
            "pearson_n_vs_bRate": _pearson(arrays["n"], brate),
            "estimated_bRate_se": _describe(arrays["estimated_brate_se"]),
        },
        "payoff_probability_ranges": {
            name: _describe(arrays[key])
            for name, key in {
                "Ha": "ha",
                "pHa": "pha",
                "La": "la",
                "Hb": "hb",
                "pHb": "phb",
                "Lb": "lb",
                "EV_A": "ev_a",
                "EV_B": "ev_b",
                "EV_difference_B_minus_A": "ev_diff_b_minus_a",
                "variance_A": "variance_a",
                "variance_B": "variance_b",
            }.items()
        },
        "bRate_by_feedback": _group_brate(
            arrays, "feedback", {False: "No feedback", True: "Feedback"}
        ),
        "bRate_by_ambiguity": _group_brate(
            arrays, "amb", {False: "Known probabilities", True: "Ambiguous B"}
        ),
        "bRate_by_correlation": _group_brate(
            arrays, "corr", {-1: "Negative", 0: "Zero", 1: "Positive"}
        ),
        "bRate_by_lottery_shape": _group_brate(
            arrays,
            "lot_shape_b",
            {
                0: "Undefined (one outcome)",
                1: "Symmetric",
                2: "Right-skewed",
                3: "Left-skewed",
            },
        ),
        "paired_feedback": _paired_feedback_statistics(arrays),
        "expected_value_relationship": ev_relationship,
        "primitive_pearson_correlations_with_bRate": primitive_relationships,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.2,
            "font.size": 10,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))


def plot_target_and_sample_quality(arrays: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    """Show target coverage, sample-size imbalance, and approximate target precision."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(arrays["brate"], bins=np.linspace(0, 1, 31), color=COLORS["blue"], alpha=0.85)
    axes[0].axvline(np.mean(arrays["brate"]), color=COLORS["orange"], linestyle="--")
    axes[0].set(
        title="Is bRate broadly distributed?", xlabel="Aggregate Gamble B rate", ylabel="Rows"
    )

    n_values, n_counts = np.unique(arrays["n"], return_counts=True)
    axes[1].bar(n_values, n_counts, color=COLORS["gray"], width=0.8)
    axes[1].set(
        title="How uneven is row sample size?", xlabel="Participants in row (n)", ylabel="Rows"
    )

    median_se = [
        np.median(arrays["estimated_brate_se"][arrays["n"] == value]) for value in n_values
    ]
    axes[2].plot(n_values, median_se, marker="o", color=COLORS["green"])
    axes[2].set(
        title="Does estimated target precision vary with n?",
        xlabel="Participants in row (n)",
        ylabel="Median bRate_std / sqrt(n)",
    )
    fig.suptitle("Target coverage and measurement considerations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output, dpi)


def _condition_panel(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    key: str,
    values: list[Any],
    labels: list[str],
    title: str,
) -> None:
    means: list[float] = []
    errors: list[float] = []
    counts: list[int] = []
    for value in values:
        selected = arrays["brate"][arrays[key] == value]
        mean, error = _mean_ci(selected)
        means.append(mean)
        errors.append(error)
        counts.append(int(selected.size))
    positions = np.arange(len(values))
    ax.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="o",
        markersize=7,
        capsize=4,
        color=COLORS["blue"],
    )
    for position, mean, count in zip(positions, means, counts, strict=True):
        ax.text(position, mean + 0.035, f"rows={count:,}", ha="center", fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("Mean bRate ± 95% row-level CI")
    ax.set_title(title)


def plot_condition_associations(arrays: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    """Compare descriptive bRate associations across prespecified experimental factors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _condition_panel(
        axes[0, 0], arrays, "feedback", [False, True], ["No feedback", "Feedback"], "Feedback"
    )
    _condition_panel(
        axes[0, 1],
        arrays,
        "amb",
        [False, True],
        ["Known", "Ambiguous B"],
        "Probability information",
    )
    _condition_panel(
        axes[1, 0],
        arrays,
        "corr",
        [-1, 0, 1],
        ["Negative", "Zero", "Positive"],
        "Payoff correlation",
    )
    _condition_panel(
        axes[1, 1],
        arrays,
        "lot_shape_b",
        [0, 1, 2, 3],
        [LOTTERY_SHAPE_LABELS[index] for index in range(4)],
        "Gamble B sublottery shape",
    )
    fig.suptitle(
        "Which experimental conditions are associated with aggregate choice?",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Intervals summarize variation across rows; contrasts are descriptive, not causal.",
        ha="center",
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, output, dpi)


def _paired_values(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    by_problem: dict[int, dict[bool, float]] = defaultdict(dict)
    for problem, feedback, brate in zip(
        arrays["problem"], arrays["feedback"], arrays["brate"], strict=True
    ):
        by_problem[int(problem)][bool(feedback)] = float(brate)
    paired = [values for values in by_problem.values() if set(values) == {False, True}]
    return (
        np.asarray([values[False] for values in paired]),
        np.asarray([values[True] for values in paired]),
    )


def plot_paired_feedback(arrays: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    """Contrast feedback conditions only where the underlying problem is matched."""
    no_feedback, feedback = _paired_values(arrays)
    differences = feedback - no_feedback
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(
        no_feedback, feedback, s=13, alpha=0.25, color=COLORS["blue"], edgecolors="none"
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"])
    axes[0].set(
        title=f"Are paired choice rates stable? (pairs={differences.size:,})",
        xlabel="bRate without feedback",
        ylabel="bRate with feedback",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    axes[1].hist(differences, bins=np.linspace(-1, 1, 41), color=COLORS["orange"], alpha=0.85)
    axes[1].axvline(0, color=COLORS["gray"], linestyle="--")
    axes[1].axvline(np.mean(differences), color="black", linestyle="-")
    axes[1].set(
        title="How does bRate change within matched problems?",
        xlabel="Feedback bRate − no-feedback bRate",
        ylabel="Problem pairs",
    )
    axes[1].text(
        0.03,
        0.95,
        f"Mean={np.mean(differences):.3f}\nMedian={np.median(differences):.3f}",
        transform=axes[1].transAxes,
        va="top",
    )
    fig.suptitle(
        "Matched descriptive comparison of feedback conditions", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, output, dpi)


def _binned_means(
    x_values: np.ndarray, y_values: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        unique = np.unique(x_values)
        return unique, np.asarray([np.mean(y_values[x_values == value]) for value in unique])
    bin_ids = np.clip(np.digitize(x_values, edges[1:-1]), 0, edges.size - 2)
    x_means = np.asarray([np.mean(x_values[bin_ids == index]) for index in range(edges.size - 1)])
    y_means = np.asarray([np.mean(y_values[bin_ids == index]) for index in range(edges.size - 1)])
    return x_means, y_means


def plot_expected_value_relationship(
    arrays: dict[str, np.ndarray], output: Path, dpi: int, bins: int
) -> None:
    """Show how aggregate choice tracks expected-value differences by information condition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, ambiguous, title in zip(
        axes,
        [False, True],
        ["Known probabilities", "Ambiguous B (oracle EV)"],
        strict=True,
    ):
        for feedback, label, color in [
            (False, "No feedback", COLORS["orange"]),
            (True, "Feedback", COLORS["blue"]),
        ]:
            mask = (arrays["amb"] == ambiguous) & (arrays["feedback"] == feedback)
            x_means, y_means = _binned_means(
                arrays["ev_diff_b_minus_a"][mask], arrays["brate"][mask], bins
            )
            ax.plot(x_means, y_means, marker="o", linewidth=2, label=label, color=color)
        ax.axvline(0, color=COLORS["gray"], linestyle="--")
        ax.axhline(0.5, color=COLORS["gray"], linestyle=":")
        ax.set(
            title=title,
            xlabel="Expected value B − expected value A",
            ylabel="Mean bRate in quantile bin",
        )
        ax.legend(frameon=False)
    fig.suptitle(
        "Does aggregate choice track expected-value advantage?",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "For ambiguous B, EV uses hidden design probabilities and was not participant-visible.",
        ha="center",
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, output, dpi)


def plot_primitive_relationships(
    arrays: dict[str, np.ndarray], output: Path, dpi: int, bins: int
) -> None:
    """Reveal marginal nonlinearities between primitive gamble fields and bRate."""
    fields = [
        ("ha", "Ha: high outcome A"),
        ("la", "La: low outcome A"),
        ("hb", "Hb: mean of B sublottery"),
        ("lb", "Lb: non-sublottery outcome B"),
        ("pha", "pHa: probability of Ha"),
        ("phb", "pHb: probability of B sublottery"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axes.flat, fields, strict=True):
        x_means, y_means = _binned_means(arrays[key], arrays["brate"], bins)
        ax.plot(x_means, y_means, marker="o", color=COLORS["purple"])
        ax.axhline(np.mean(arrays["brate"]), color=COLORS["gray"], linestyle=":")
        ax.set(title=title, xlabel="Feature value (binned where needed)", ylabel="Mean bRate")
    fig.suptitle(
        "Are primitive payoff and probability relationships nonlinear?",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Marginal associations mix other problem characteristics and are not causal effects.",
        ha="center",
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, output, dpi)


def generate_figures(
    arrays: dict[str, np.ndarray], figures_dir: Path, config: dict[str, int]
) -> list[Path]:
    """Generate the prespecified compact EDA figure set."""
    _style()
    figures = [
        figures_dir / "target_and_sample_quality.png",
        figures_dir / "condition_associations.png",
        figures_dir / "paired_feedback_comparison.png",
        figures_dir / "expected_value_relationship.png",
        figures_dir / "primitive_feature_relationships.png",
    ]
    plot_target_and_sample_quality(arrays, figures[0], config["figure_dpi"])
    plot_condition_associations(arrays, figures[1], config["figure_dpi"])
    plot_paired_feedback(arrays, figures[2], config["figure_dpi"])
    plot_expected_value_relationship(
        arrays, figures[3], config["figure_dpi"], config["quantile_bins"]
    )
    plot_primitive_relationships(arrays, figures[4], config["figure_dpi"], config["quantile_bins"])
    return figures


def _format_group_table(groups: dict[str, dict[str, float | int]]) -> str:
    lines = ["| Group | Rows | Mean bRate | Median bRate |", "|---|---:|---:|---:|"]
    for label, values in groups.items():
        lines.append(
            f"| {label} | {values['count']:,} | {values['mean_bRate']:.3f} | "
            f"{values['median_bRate']:.3f} |"
        )
    return "\n".join(lines)


def write_report(statistics: dict[str, Any], figures: list[Path], output: Path) -> None:
    """Write the required EDA summary entirely from executed analysis results."""
    target = statistics["target_bRate"]
    samples = statistics["sample_size_n"]
    paired = statistics["paired_feedback"]
    paired_diff = paired["difference_feedback_minus_no_feedback"]
    ev = statistics["expected_value_relationship"]
    payoff_ranges = statistics["payoff_probability_ranges"]
    primitive = statistics["primitive_pearson_correlations_with_bRate"]
    relative_figures = [path.relative_to(output.parent) for path in figures]
    report = f"""# DecisionLab exploratory data analysis

Generated by `decisionlab-eda` from the checksum-validated choices13k source at commit `{statistics["source_commit"]}`. All values below were computed by the executed run recorded in `artifacts/manifests/eda_statistics.json`. They are descriptive associations, not causal estimates.

## 1. Important findings

- `bRate` covers the full `[0, 1]` interval. Its mean is {target["mean"]:.3f}, median is {target["median"]:.3f}, standard deviation is {target["std"]:.3f}, and interquartile range is [{target["q25"]:.3f}, {target["q75"]:.3f}]. There are {target["zero_count"]} zeros and {target["one_count"]} ones.
- Row sample size is concentrated near the minimum: median `n` is {samples["median"]:.0f}, with range {samples["min"]:.0f}–{samples["max"]:.0f}. The approximate row-level standard error `bRate_std / sqrt(n)` has median {samples["estimated_bRate_se"]["median"]:.3f}. This indicates non-negligible target measurement uncertainty.
- Across all rows, the Pearson association between oracle EV difference (`EV_B - EV_A`) and `bRate` is {ev["all"]["pearson_ev_difference_vs_bRate"]:.3f}. It is {ev["known"]["pearson_ev_difference_vs_bRate"]:.3f} when probabilities are known and {ev["ambiguous"]["pearson_ev_difference_vs_bRate"]:.3f} for ambiguous B. The latter uses hidden design probabilities and does not represent information available to participants.
- Among the {paired["pair_count"]:,} matched problems observed with and without feedback, mean `bRate` changes from {paired["no_feedback_mean_bRate"]:.3f} to {paired["feedback_mean_bRate"]:.3f}. The within-problem feedback-minus-no-feedback difference has mean {paired_diff["mean"]:.3f}, median {paired_diff["median"]:.3f}, and IQR [{paired_diff["q25"]:.3f}, {paired_diff["q75"]:.3f}]. Positive changes occur in {paired["proportion_positive"]:.1%} of pairs and negative changes in {paired["proportion_negative"]:.1%}. This matched association still should not be called a causal feedback effect.
- Primitive features have different marginal relationships with choice. The strongest absolute Pearson association among the six primitive payoff/probability fields is `{max(["Ha", "pHa", "La", "Hb", "pHb", "Lb"], key=lambda name: abs(primitive[name]))}` at {max(abs(primitive[name]) for name in ["Ha", "pHa", "La", "Hb", "pHb", "Lb"]):.3f}; the binned plots show why nonlinear models or transformations may eventually be useful.
- Payoffs span materially different ranges: `Ha` [{payoff_ranges["Ha"]["min"]:.0f}, {payoff_ranges["Ha"]["max"]:.0f}], `La` [{payoff_ranges["La"]["min"]:.0f}, {payoff_ranges["La"]["max"]:.0f}], `Hb` [{payoff_ranges["Hb"]["min"]:.0f}, {payoff_ranges["Hb"]["max"]:.0f}], and `Lb` [{payoff_ranges["Lb"]["min"]:.0f}, {payoff_ranges["Lb"]["max"]:.0f}]. Both probability fields range from {payoff_ranges["pHa"]["min"]:.2f} to {payoff_ranges["pHa"]["max"]:.2f} and take a limited discrete set of values.

### Condition summaries

Feedback:

{_format_group_table(statistics["bRate_by_feedback"])}

Ambiguity:

{_format_group_table(statistics["bRate_by_ambiguity"])}

Correlation:

{_format_group_table(statistics["bRate_by_correlation"])}

Gamble B sublottery shape:

{_format_group_table(statistics["bRate_by_lottery_shape"])}

### Question-driven figures

1. [Target coverage and sample quality]({relative_figures[0]}) — Are the target and its estimated precision evenly distributed?
2. [Condition associations]({relative_figures[1]}) — How does mean aggregate choice vary with feedback, ambiguity, correlation, and lottery shape?
3. [Matched feedback comparison]({relative_figures[2]}) — How different are feedback and no-feedback observations for the same problem?
4. [Expected-value relationship]({relative_figures[3]}) — Does aggregate choice track the EV advantage of Gamble B, and does that pattern differ by information condition?
5. [Primitive feature relationships]({relative_figures[4]}) — Are marginal payoff and probability relationships visibly nonlinear?

## 2. Potential modeling issues

- The target is bounded and includes exact endpoints, so methods requiring a response strictly inside `(0, 1)` need special handling.
- `bRate` is an aggregate estimate with varying `n` and participant-level dispersion. Primary evaluation should remain unweighted by problem, with `n`-weighted or uncertainty-aware results labeled as sensitivity analyses.
- `bRate_std` is computed from the same responses as `bRate`; it is useful for measurement diagnostics but must not be a predictor.
- Feedback rows ({statistics["bRate_by_feedback"]["Feedback"]["count"]:,}) greatly outnumber no-feedback rows ({statistics["bRate_by_feedback"]["No feedback"]["count"]:,}), and most problems do not have both conditions. Unmatched marginal feedback differences mix feedback status with problem composition.
- Expected-value and primitive-feature plots show associations and likely interactions rather than isolated mechanisms. Linear marginal effects may miss thresholding, saturation, and condition-specific patterns.
- `LotShapeB = 0` means the sublottery has one outcome; it is a structural state, not an ordered level below the other shapes. Correlation and shape should likewise be categorical.
- Under ambiguity, latent probabilities and oracle expected values are analyst-visible design information but were not participant-visible. Visible-feature and oracle-feature model specifications must remain separate.

## 3. Possible leakage risks

- The 1,562 paired feedback/no-feedback rows share an underlying problem. Future partitions must group by `Problem` and verified structural fingerprint.
- `Block = 1` identifies no-feedback rows, while blocks 2–5 identify feedback rows. Using both fields creates a deterministic shortcut and requires an explicit prediction-time rationale.
- `bRate_std` is target-derived and prohibited as a feature. `n` is post-collection measurement metadata and is not a default feature.
- The JSON is keyed by CSV row index rather than `Problem`; joining on the wrong identifier would silently misalign lotteries.
- Probabilities hidden from participants under ambiguity can leak oracle information into a model described as participant-visible.
- This EDA inspected the full dataset before a locked test partition existed. Questions stated in this report can be preregistered before splitting, but newly discovered patterns should not later be presented as untouched confirmatory test findings.

## 4. Questions worth testing later

- How much does a calibrated expected-value baseline improve over the training-set mean, especially when probabilities are known?
- Do variance, downside exposure, and lottery shape explain residual behavior beyond EV difference?
- Are feedback associations robust in the matched-problem subset and under grouped resampling?
- Does ambiguity change the relationship between oracle EV advantage and aggregate choice?
- Do payoff/probability transformations improve out-of-sample calibration without sacrificing interpretability?
- How sensitive are conclusions to problem-level versus `n`-weighted evaluation?
- Which findings survive a leakage-safe grouped split and prespecified subgroup definitions?
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def run_eda(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    figures_dir: Path = DEFAULT_FIGURES,
    report_path: Path = DEFAULT_REPORT,
    statistics_path: Path = DEFAULT_STATISTICS,
) -> dict[str, Any]:
    """Validate data, execute EDA, and persist figures, statistics, and report."""
    validation = load_and_validate(raw_dir, manifest_path)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    config = load_eda_config(config_path)
    arrays = build_analysis_arrays(selections, problems)
    statistics = compute_eda_statistics(arrays)
    statistics.update(
        {
            "dataset": validation["dataset"],
            "source_commit": validation["source_commit"],
            "source_sha256": validation["sha256"],
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "decisionlab_version": __version__,
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "config": config,
        }
    )
    figures = generate_figures(arrays, figures_dir, config)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(statistics, figures, report_path)
    return statistics


def main() -> None:
    """Run the reproducible DecisionLab EDA pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    args = parser.parse_args()
    statistics = run_eda(
        args.raw_dir,
        args.manifest,
        args.config,
        args.figures_dir,
        args.report,
        args.statistics,
    )
    print(
        f"EDA complete: rows={statistics['target_bRate']['count']:,}; "
        f"figures={args.figures_dir}; report={args.report}; statistics={args.statistics}"
    )


if __name__ == "__main__":
    main()
