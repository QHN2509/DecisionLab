"""Build documented, leakage-audited behavioral features from validated choices13k data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from decisionlab import __version__
from decisionlab.data.fetch import DEFAULT_DESTINATION, DEFAULT_MANIFEST, sha256_file
from decisionlab.data.validation import (
    SelectionRecord,
    load_and_validate,
    load_problems,
    load_selections,
)
from decisionlab.experiments.provenance import (
    finalize_run_provenance,
    start_run_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW_FEATURES = DEFAULT_OUTPUT_DIR / "choices13k_raw_features.csv"
DEFAULT_ENGINEERED_FEATURES = DEFAULT_OUTPUT_DIR / "choices13k_engineered_features.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts" / "manifests" / "feature_build_summary.json"
DEFAULT_DOCUMENTATION = PROJECT_ROOT / "docs" / "features.md"
DEFAULT_PROVENANCE = PROJECT_ROOT / "artifacts" / "manifests" / "feature_build_provenance.json"

Availability = Literal[
    "participant-visible",
    "oracle under ambiguity",
    "experimental condition",
    "categorical encoding",
]


@dataclass(frozen=True, slots=True)
class RawFeatureRow:
    """Validated source predictors, separated from identifiers and outcomes."""

    feedback: bool
    ha: int
    pha: float
    la: int
    hb: int
    phb: float
    lb: int
    lot_shape_b: int
    lot_num_b: int
    amb: bool
    corr: int


@dataclass(frozen=True, slots=True)
class ProblemFeatureInput:
    """Predictor-only full gamble distributions from the problem JSON."""

    gamble_a: list[list[float]]
    gamble_b: list[list[float]]


@dataclass(frozen=True, slots=True)
class ScenarioFeatureInput:
    """User-constructible risky-choice inputs supported by the trained feature contract."""

    high_payoff_a: float
    high_probability_a: float
    low_payoff_a: float
    sublottery_mean_b: float
    sublottery_probability_b: float
    low_payoff_b: float
    lottery_shape_b: int
    lottery_outcomes_b: int
    ambiguity: bool
    feedback: bool
    correlation: int


@dataclass(frozen=True, slots=True)
class EngineeredFeatureRow:
    """One row of interpretable, non-target-derived behavioral features."""

    expected_value_a: float
    expected_value_b_oracle: float
    expected_value_difference_b_minus_a_oracle: float
    payoff_range_a: float
    payoff_range_b: float
    payoff_range_difference_b_minus_a: float
    maximum_payoff_difference_b_minus_a: float
    minimum_payoff_difference_b_minus_a: float
    payoff_std_a: float
    payoff_std_b_oracle: float
    payoff_std_difference_b_minus_a_oracle: float
    best_payoff_probability_a: float
    best_payoff_probability_b_oracle: float
    best_payoff_probability_difference_b_minus_a_oracle: float
    loss_probability_a: float
    loss_probability_b_oracle: float
    loss_probability_difference_b_minus_a_oracle: float
    ambiguity_indicator: int
    feedback_indicator: int
    lottery_shape_b_undefined: int
    lottery_shape_b_symmetric: int
    lottery_shape_b_right_skewed: int
    lottery_shape_b_left_skewed: int
    correlation_negative: int
    correlation_zero: int
    correlation_positive: int
    expected_value_difference_oracle_x_feedback: float
    expected_value_difference_oracle_x_ambiguity: float


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Mathematical and leakage contract for one engineered feature."""

    name: str
    formula: str
    sources: tuple[str, ...]
    interpretation: str
    assumptions: str
    limitations: str
    availability: Availability
    leakage_audit: str
    target_derived: bool = False


def _definition(
    name: str,
    formula: str,
    sources: tuple[str, ...],
    interpretation: str,
    assumptions: str,
    limitations: str,
    availability: Availability,
    leakage_audit: str,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        formula=formula,
        sources=sources,
        interpretation=interpretation,
        assumptions=assumptions,
        limitations=limitations,
        availability=availability,
        leakage_audit=leakage_audit,
    )


FEATURE_DEFINITIONS = (
    _definition(
        "expected_value_a",
        "pHa × Ha + (1 − pHa) × La",
        ("pHa", "Ha", "La"),
        "Probability-weighted mean payoff of Gamble A.",
        "A probabilities and payoffs are participant-visible.",
        "Expected value does not represent risk preferences or nonlinear utility.",
        "participant-visible",
        "Uses only pre-choice Gamble A parameters; no outcome or target fields.",
    ),
    _definition(
        "expected_value_b_oracle",
        "pHb × Hb + (1 − pHb) × Lb",
        ("pHb", "Hb", "Lb"),
        "Probability-weighted mean payoff of the complete Gamble B distribution.",
        "Hb is the mean of the embedded B sublottery, as defined upstream.",
        "B probabilities are hidden when Amb=True, so this is then an oracle feature.",
        "oracle under ambiguity",
        "Pre-choice design data only, but unavailable to participants under ambiguity.",
    ),
    _definition(
        "expected_value_difference_b_minus_a_oracle",
        "expected_value_b_oracle − expected_value_a",
        ("expected_value_b_oracle", "expected_value_a"),
        "Objective expected-value advantage of B over A.",
        "Both gamble means are comparable in the payoff units of the task.",
        "Inherits the oracle status of expected_value_b_oracle under ambiguity.",
        "oracle under ambiguity",
        "Derived only from audited expected values; unavailable under ambiguous B.",
    ),
    _definition(
        "payoff_range_a",
        "max(A payoffs) − min(A payoffs)",
        ("c13k_problems.json:A",),
        "Outcome spread of Gamble A without probability weighting.",
        "Displayed A outcomes define the relevant payoff support.",
        "Range ignores outcome probabilities and distribution shape.",
        "participant-visible",
        "Uses only the displayed payoff support; no response information.",
    ),
    _definition(
        "payoff_range_b",
        "max(B payoffs) − min(B payoffs)",
        ("c13k_problems.json:B",),
        "Outcome spread of Gamble B without probability weighting.",
        "B payoff values remain visible when their probabilities are ambiguous.",
        "Range ignores probabilities and may be dominated by rare outcomes.",
        "participant-visible",
        "Uses only the displayed payoff support; no hidden probability or response data.",
    ),
    _definition(
        "payoff_range_difference_b_minus_a",
        "payoff_range_b − payoff_range_a",
        ("payoff_range_b", "payoff_range_a"),
        "How much wider B's payoff support is than A's.",
        "Ranges are comparable because both gambles use the same payoff units.",
        "A difference in ranges is not a complete measure of relative risk.",
        "participant-visible",
        "Difference of two participant-visible, non-target-derived features.",
    ),
    _definition(
        "maximum_payoff_difference_b_minus_a",
        "max(B payoffs) − max(A payoffs)",
        ("c13k_problems.json:A", "c13k_problems.json:B"),
        "B's best possible payoff relative to A's best possible payoff.",
        "Participants attend to the displayed extreme outcomes.",
        "Ignores the probabilities of reaching those maxima.",
        "participant-visible",
        "Uses displayed payoff extrema only; no target or post-choice fields.",
    ),
    _definition(
        "minimum_payoff_difference_b_minus_a",
        "min(B payoffs) − min(A payoffs)",
        ("c13k_problems.json:A", "c13k_problems.json:B"),
        "B's worst possible payoff relative to A's worst possible payoff.",
        "Participants attend to the displayed extreme outcomes.",
        "Ignores the probabilities of reaching those minima.",
        "participant-visible",
        "Uses displayed payoff extrema only; no target or post-choice fields.",
    ),
    _definition(
        "payoff_std_a",
        "sqrt(Σᵢ p(Aᵢ) × (Aᵢ − expected_value_a)²)",
        ("c13k_problems.json:A", "expected_value_a"),
        "Probability-weighted payoff dispersion of Gamble A.",
        "The population standard deviation of the stated A distribution is relevant.",
        "Dispersion treats upside and downside variation symmetrically.",
        "participant-visible",
        "Uses only participant-visible A probabilities and payoffs.",
    ),
    _definition(
        "payoff_std_b_oracle",
        "sqrt(Σᵢ p(Bᵢ) × (Bᵢ − expected_value_b_oracle)²)",
        ("c13k_problems.json:B", "expected_value_b_oracle"),
        "Probability-weighted payoff dispersion of Gamble B.",
        "The full marginal B distribution in the JSON is the relevant distribution.",
        "Probabilities are not participant-visible when Amb=True.",
        "oracle under ambiguity",
        "Pre-choice design data only, but hidden B probabilities make it oracle under ambiguity.",
    ),
    _definition(
        "payoff_std_difference_b_minus_a_oracle",
        "payoff_std_b_oracle − payoff_std_a",
        ("payoff_std_b_oracle", "payoff_std_a"),
        "Difference in objective payoff dispersion between B and A.",
        "Both standard deviations use the same payoff units.",
        "Not a complete risk measure and oracle under ambiguity.",
        "oracle under ambiguity",
        "Inherits only the documented hidden-probability risk from its B component.",
    ),
    _definition(
        "best_payoff_probability_a",
        "Σᵢ p(Aᵢ) × 1[Aᵢ = max(A payoffs)]",
        ("c13k_problems.json:A",),
        "Probability that A yields its best displayed payoff.",
        "Probabilities for tied maximum outcomes are summed.",
        "A best-outcome probability does not describe the remaining distribution.",
        "participant-visible",
        "Uses only stated A outcomes and probabilities.",
    ),
    _definition(
        "best_payoff_probability_b_oracle",
        "Σᵢ p(Bᵢ) × 1[Bᵢ = max(B payoffs)]",
        ("c13k_problems.json:B",),
        "Probability that B yields its best displayed payoff.",
        "Probabilities for tied maximum outcomes are summed.",
        "The probability is hidden when Amb=True.",
        "oracle under ambiguity",
        "Uses no response data, but uses hidden design probabilities under ambiguity.",
    ),
    _definition(
        "best_payoff_probability_difference_b_minus_a_oracle",
        "best_payoff_probability_b_oracle − best_payoff_probability_a",
        ("best_payoff_probability_b_oracle", "best_payoff_probability_a"),
        "Difference in the chance of obtaining each gamble's best payoff.",
        "Best-outcome events are comparable summaries across gambles.",
        "The two best payoffs can differ greatly in magnitude; oracle under ambiguity.",
        "oracle under ambiguity",
        "Inherits hidden-probability risk only from the B component.",
    ),
    _definition(
        "loss_probability_a",
        "Σᵢ p(Aᵢ) × 1[Aᵢ < 0]",
        ("c13k_problems.json:A",),
        "Objective probability that Gamble A produces a negative payoff.",
        "Zero is the behaviorally meaningful gain/loss reference point.",
        "Does not distinguish small from severe losses.",
        "participant-visible",
        "Uses only stated A outcomes and probabilities.",
    ),
    _definition(
        "loss_probability_b_oracle",
        "Σᵢ p(Bᵢ) × 1[Bᵢ < 0]",
        ("c13k_problems.json:B",),
        "Objective probability that Gamble B produces a negative payoff.",
        "Zero is the behaviorally meaningful gain/loss reference point.",
        "Hidden probabilities make this unavailable when Amb=True.",
        "oracle under ambiguity",
        "Uses no response data, but uses hidden design probabilities under ambiguity.",
    ),
    _definition(
        "loss_probability_difference_b_minus_a_oracle",
        "loss_probability_b_oracle − loss_probability_a",
        ("loss_probability_b_oracle", "loss_probability_a"),
        "B's objective loss probability relative to A's.",
        "A zero-payoff reference point is appropriate for both gambles.",
        "Does not encode loss magnitude and is oracle under ambiguity.",
        "oracle under ambiguity",
        "Inherits hidden-probability risk only from the B component.",
    ),
    _definition(
        "ambiguity_indicator",
        "1[Amb = True]",
        ("Amb",),
        "Whether Gamble B probabilities were hidden from participants.",
        "The upstream Amb field correctly describes information availability.",
        "Does not quantify the degree of ambiguity beyond this binary design.",
        "experimental condition",
        "Known before choice; safe if ambiguity is known in the prediction setting.",
    ),
    _definition(
        "feedback_indicator",
        "1[Feedback = True]",
        ("Feedback",),
        "Whether participants received obtained and forgone outcome feedback.",
        "Feedback status is known for the prediction task.",
        "Feedback is entangled with block/order and observed experience histories are unavailable.",
        "experimental condition",
        "Known before the modeled condition, but can proxy for block/order; "
        "paired rows must be grouped.",
    ),
    _definition(
        "lottery_shape_b_undefined",
        "1[LotShapeB = 0]",
        ("LotShapeB",),
        "Indicator that the B sublottery shape is undefined because it has one outcome.",
        "Upstream category 0 has its documented meaning.",
        "Redundant with LotNumB=1 and must not be treated as ordered.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "lottery_shape_b_symmetric",
        "1[LotShapeB = 1]",
        ("LotShapeB",),
        "Indicator for a symmetric B sublottery.",
        "Upstream shape labels describe the generated sublottery.",
        "Shape alone does not identify spread or tail magnitude.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "lottery_shape_b_right_skewed",
        "1[LotShapeB = 2]",
        ("LotShapeB",),
        "Indicator for a right-skewed B sublottery.",
        "Upstream shape labels describe the generated sublottery.",
        "Shape alone does not identify spread or tail magnitude.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "lottery_shape_b_left_skewed",
        "1[LotShapeB = 3]",
        ("LotShapeB",),
        "Indicator for a left-skewed B sublottery.",
        "Upstream shape labels describe the generated sublottery.",
        "Shape alone does not identify spread or tail magnitude.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "correlation_negative",
        "1[Corr = −1]",
        ("Corr",),
        "Indicator for negatively correlated gamble payoffs.",
        "Upstream correlation category is known for the condition.",
        "Category does not provide correlation magnitude.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "correlation_zero",
        "1[Corr = 0]",
        ("Corr",),
        "Indicator for uncorrelated gamble payoffs.",
        "Upstream correlation category is known for the condition.",
        "Category does not establish empirical independence in aggregate responses.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "correlation_positive",
        "1[Corr = 1]",
        ("Corr",),
        "Indicator for positively correlated gamble payoffs.",
        "Upstream correlation category is known for the condition.",
        "Category does not provide correlation magnitude.",
        "categorical encoding",
        "Pre-choice structural category; no target information.",
    ),
    _definition(
        "expected_value_difference_oracle_x_feedback",
        "expected_value_difference_b_minus_a_oracle × feedback_indicator",
        ("expected_value_difference_b_minus_a_oracle", "feedback_indicator"),
        "Allows EV sensitivity to differ descriptively when outcome feedback is available.",
        "A feedback-specific EV slope is behaviorally interpretable.",
        "Requires both main effects later; oracle under ambiguity and not causal.",
        "oracle under ambiguity",
        "No target data, but inherits oracle status and feedback's block/order caveat.",
    ),
    _definition(
        "expected_value_difference_oracle_x_ambiguity",
        "expected_value_difference_b_minus_a_oracle × ambiguity_indicator",
        ("expected_value_difference_b_minus_a_oracle", "ambiguity_indicator"),
        "Marks the oracle EV slope specifically for ambiguous Gamble B problems.",
        "Attenuated oracle-EV sensitivity under ambiguity is behaviorally testable.",
        "The interacted EV was not participant-visible and cannot represent "
        "explicit EV calculation.",
        "oracle under ambiguity",
        "No target data, but intentionally uses latent design information in ambiguous rows.",
    ),
)

FORBIDDEN_FEATURE_SOURCES = {
    "bRate",
    "bRate_std",
    "brate",
    "brate_std",
    "n",
    "Block",
    "block",
    "Problem",
    "problem",
    "row_index",
    "structural_fingerprint",
    "outer_fold",
    "inner_fold",
}


@dataclass(frozen=True, slots=True)
class OptionSummary:
    minimum: float
    maximum: float
    payoff_range: float
    expected_value: float
    payoff_std: float
    best_payoff_probability: float
    loss_probability: float


def summarize_option(outcomes: list[list[float]]) -> OptionSummary:
    """Calculate interpretable distribution summaries for one validated gamble."""
    minimum = min(float(outcome[1]) for outcome in outcomes)
    maximum = max(float(outcome[1]) for outcome in outcomes)
    expected_value = sum(float(probability) * float(payoff) for probability, payoff in outcomes)
    variance = sum(
        float(probability) * (float(payoff) - expected_value) ** 2
        for probability, payoff in outcomes
    )
    return OptionSummary(
        minimum=minimum,
        maximum=maximum,
        payoff_range=maximum - minimum,
        expected_value=expected_value,
        payoff_std=math.sqrt(max(variance, 0.0)),
        best_payoff_probability=sum(
            float(probability) for probability, payoff in outcomes if float(payoff) == maximum
        ),
        loss_probability=sum(
            float(probability) for probability, payoff in outcomes if float(payoff) < 0.0
        ),
    )


def build_sublottery_distribution(
    mean_payoff: float, shape: int, outcome_count: int
) -> list[list[float]]:
    """Construct a choices13k-compatible Gamble B sublottery with the requested mean."""
    if isinstance(mean_payoff, bool) or not isinstance(mean_payoff, int | float):
        raise ValueError("Sublottery mean must be numeric")
    if not math.isfinite(mean_payoff):
        raise ValueError("Sublottery mean must be finite")
    if isinstance(shape, bool) or not isinstance(shape, int):
        raise ValueError("Lottery shape must be an integer")
    if isinstance(outcome_count, bool) or not isinstance(outcome_count, int):
        raise ValueError("Lottery outcome count must be an integer")
    if shape not in {0, 1, 2, 3}:
        raise ValueError("Lottery shape must be 0, 1, 2, or 3")
    if (shape == 0) != (outcome_count == 1) or not 1 <= outcome_count <= 8:
        raise ValueError("Undefined shape requires one outcome; other shapes require 2–8")
    if shape == 0:
        return [[1.0, float(mean_payoff)]]

    if shape == 1:
        denominator = 2 ** (outcome_count - 1)
        center = (outcome_count - 1) / 2.0
        return [
            [math.comb(outcome_count - 1, index) / denominator, mean_payoff + index - center]
            for index in range(outcome_count)
        ]

    base_outcomes = np.asarray([2**index for index in range(outcome_count)], dtype=float)
    probabilities = np.asarray(
        [
            1 / 2 ** (index + 1) if index < outcome_count - 1 else 1 / 2**index
            for index in range(outcome_count)
        ],
        dtype=float,
    )
    centered = 2.0 * (base_outcomes - float(np.sum(probabilities * base_outcomes)))
    if shape == 3:
        centered = -centered
    return [
        [float(probability), float(mean_payoff + offset)]
        for probability, offset in zip(probabilities, centered, strict=True)
    ]


def build_scenario_problem(
    scenario: ScenarioFeatureInput,
) -> tuple[RawFeatureRow, dict[str, list[list[float]]]]:
    """Validate app inputs and construct predictor-only production inputs."""
    numeric_values = (
        scenario.high_payoff_a,
        scenario.high_probability_a,
        scenario.low_payoff_a,
        scenario.sublottery_mean_b,
        scenario.sublottery_probability_b,
        scenario.low_payoff_b,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int | float) for value in numeric_values
    ):
        raise ValueError("Scenario numeric inputs must be numbers")
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Scenario inputs must be finite")
    if type(scenario.ambiguity) is not bool or type(scenario.feedback) is not bool:
        raise ValueError("Ambiguity and feedback inputs must be boolean")
    if not 0.0 <= scenario.high_probability_a <= 1.0:
        raise ValueError("Gamble A probability must be in [0, 1]")
    if not 0.0 <= scenario.sublottery_probability_b <= 1.0:
        raise ValueError("Gamble B sublottery probability must be in [0, 1]")
    if scenario.high_payoff_a < scenario.low_payoff_a:
        raise ValueError("Gamble A high payoff must be at least its low payoff")
    if scenario.correlation not in {-1, 0, 1}:
        raise ValueError("Correlation category must be -1, 0, or 1")

    sublottery = build_sublottery_distribution(
        scenario.sublottery_mean_b,
        scenario.lottery_shape_b,
        scenario.lottery_outcomes_b,
    )
    gamble_a = [
        [scenario.high_probability_a, scenario.high_payoff_a],
        [1.0 - scenario.high_probability_a, scenario.low_payoff_a],
    ]
    gamble_b = [[1.0 - scenario.sublottery_probability_b, scenario.low_payoff_b]] + [
        [scenario.sublottery_probability_b * probability, payoff]
        for probability, payoff in sublottery
    ]
    predictors = RawFeatureRow(
        feedback=scenario.feedback,
        ha=scenario.high_payoff_a,
        pha=scenario.high_probability_a,
        la=scenario.low_payoff_a,
        hb=scenario.sublottery_mean_b,
        phb=scenario.sublottery_probability_b,
        lb=scenario.low_payoff_b,
        lot_shape_b=scenario.lottery_shape_b,
        lot_num_b=scenario.lottery_outcomes_b,
        amb=scenario.ambiguity,
        corr=scenario.correlation,
    )
    return predictors, {"A": gamble_a, "B": gamble_b}


def engineer_scenario_features(
    scenario: ScenarioFeatureInput,
) -> tuple[EngineeredFeatureRow, dict[str, list[list[float]]]]:
    """Run a constructed scenario through the production behavioral feature logic."""
    predictors, problem = build_scenario_problem(scenario)
    engineered = engineer_behavioral_features(predictors, extract_problem_features(problem))
    validate_feature_rows([predictors], [engineered])
    return engineered, problem


def extract_raw_features(record: SelectionRecord) -> RawFeatureRow:
    """Select allowed pre-choice raw predictors without identifiers or outcomes."""
    return RawFeatureRow(
        feedback=record.feedback,
        ha=record.ha,
        pha=record.pha,
        la=record.la,
        hb=record.hb,
        phb=record.phb,
        lb=record.lb,
        lot_shape_b=record.lot_shape_b,
        lot_num_b=record.lot_num_b,
        amb=record.amb,
        corr=record.corr,
    )


def extract_problem_features(
    problem_description: dict[str, list[list[float]]],
) -> ProblemFeatureInput:
    """Select only A/B distributions and reject evaluation metadata in problem inputs."""
    if set(problem_description) != {"A", "B"}:
        raise ValueError("Problem feature input must contain exactly Gamble A and Gamble B")
    return ProblemFeatureInput(
        gamble_a=problem_description["A"],
        gamble_b=problem_description["B"],
    )


def engineer_behavioral_features(
    predictors: RawFeatureRow, problem: ProblemFeatureInput
) -> EngineeredFeatureRow:
    """Build features from predictor-only inputs that cannot contain outcomes or metadata."""
    if not isinstance(predictors, RawFeatureRow) or not isinstance(problem, ProblemFeatureInput):
        raise TypeError("Production feature engineering requires predictor-only input types")
    option_a = summarize_option(problem.gamble_a)
    option_b = summarize_option(problem.gamble_b)
    expected_value_a = predictors.pha * predictors.ha + (1.0 - predictors.pha) * predictors.la
    expected_value_b = predictors.phb * predictors.hb + (1.0 - predictors.phb) * predictors.lb
    if not math.isclose(expected_value_a, option_a.expected_value, abs_tol=1e-9):
        raise ValueError("Gamble A expected value differs between CSV and JSON")
    if not math.isclose(expected_value_b, option_b.expected_value, abs_tol=1e-9):
        raise ValueError("Gamble B expected value differs between CSV and JSON")
    expected_value_difference = expected_value_b - expected_value_a
    feedback_indicator = int(predictors.feedback)
    ambiguity_indicator = int(predictors.amb)
    return EngineeredFeatureRow(
        expected_value_a=expected_value_a,
        expected_value_b_oracle=expected_value_b,
        expected_value_difference_b_minus_a_oracle=expected_value_difference,
        payoff_range_a=option_a.payoff_range,
        payoff_range_b=option_b.payoff_range,
        payoff_range_difference_b_minus_a=option_b.payoff_range - option_a.payoff_range,
        maximum_payoff_difference_b_minus_a=option_b.maximum - option_a.maximum,
        minimum_payoff_difference_b_minus_a=option_b.minimum - option_a.minimum,
        payoff_std_a=option_a.payoff_std,
        payoff_std_b_oracle=option_b.payoff_std,
        payoff_std_difference_b_minus_a_oracle=option_b.payoff_std - option_a.payoff_std,
        best_payoff_probability_a=option_a.best_payoff_probability,
        best_payoff_probability_b_oracle=option_b.best_payoff_probability,
        best_payoff_probability_difference_b_minus_a_oracle=(
            option_b.best_payoff_probability - option_a.best_payoff_probability
        ),
        loss_probability_a=option_a.loss_probability,
        loss_probability_b_oracle=option_b.loss_probability,
        loss_probability_difference_b_minus_a_oracle=(
            option_b.loss_probability - option_a.loss_probability
        ),
        ambiguity_indicator=ambiguity_indicator,
        feedback_indicator=feedback_indicator,
        lottery_shape_b_undefined=int(predictors.lot_shape_b == 0),
        lottery_shape_b_symmetric=int(predictors.lot_shape_b == 1),
        lottery_shape_b_right_skewed=int(predictors.lot_shape_b == 2),
        lottery_shape_b_left_skewed=int(predictors.lot_shape_b == 3),
        correlation_negative=int(predictors.corr == -1),
        correlation_zero=int(predictors.corr == 0),
        correlation_positive=int(predictors.corr == 1),
        expected_value_difference_oracle_x_feedback=(
            expected_value_difference * feedback_indicator
        ),
        expected_value_difference_oracle_x_ambiguity=(
            expected_value_difference * ambiguity_indicator
        ),
    )


def audit_feature_contract() -> dict[str, Any]:
    """Audit every engineered feature for coverage and prohibited information."""
    declared_names = [definition.name for definition in FEATURE_DEFINITIONS]
    output_names = [field.name for field in fields(EngineeredFeatureRow)]
    if declared_names != output_names:
        raise ValueError("Feature definitions and engineered output fields are not aligned")
    audit: dict[str, Any] = {}
    for definition in FEATURE_DEFINITIONS:
        forbidden = sorted(set(definition.sources) & FORBIDDEN_FEATURE_SOURCES)
        if definition.target_derived or forbidden:
            raise ValueError(
                f"Feature {definition.name} contains prohibited information: {forbidden}"
            )
        audit[definition.name] = {
            "sources": list(definition.sources),
            "target_derived": definition.target_derived,
            "availability": definition.availability,
            "leakage_assessment": definition.leakage_audit,
            "status": "PASS",
        }
    raw_fields = {field.name for field in fields(RawFeatureRow)}
    problem_fields = {field.name for field in fields(ProblemFeatureInput)}
    forbidden_raw = raw_fields & FORBIDDEN_FEATURE_SOURCES
    expected_problem_fields = {"gamble_a", "gamble_b"}
    if problem_fields != expected_problem_fields:
        raise ValueError("Problem predictor contract must contain only Gamble A and Gamble B")
    if forbidden_raw:
        raise ValueError(
            f"Raw feature contract contains prohibited fields: {sorted(forbidden_raw)}"
        )
    return {
        "status": "PASS",
        "engineered_feature_count": len(FEATURE_DEFINITIONS),
        "raw_feature_count": len(raw_fields),
        "forbidden_raw_fields": sorted(forbidden_raw),
        "production_input_contract": {
            "record_type": "RawFeatureRow",
            "record_fields": sorted(raw_fields),
            "problem_type": "ProblemFeatureInput",
            "problem_fields": sorted(problem_fields),
            "forbidden_fields_unrepresentable": sorted(FORBIDDEN_FEATURE_SOURCES),
            "runtime_type_enforcement": True,
            "status": "PASS",
        },
        "features": audit,
    }


def audit_forbidden_field_invariance(
    selections: list[SelectionRecord],
    problems: dict[str, Any],
    expected_features: list[EngineeredFeatureRow],
) -> dict[str, Any]:
    """Execute exact metamorphic checks for every forbidden SelectionRecord field."""
    if not selections or len(selections) != len(expected_features):
        raise ValueError("Metamorphic audit inputs must be aligned and nonempty")
    mutations = {
        "problem": lambda row: row.problem + 1_000_000,
        "n": lambda row: 16 if row.n == 15 else 15,
        "block": lambda row: row.block + 10,
        "bRate": lambda row: 1.0 if row.brate == 0.0 else 0.0,
        "bRate_std": lambda row: 1.0 if row.brate_std == 0.0 else 0.0,
    }
    comparisons = 0
    for row_index, (record, expected) in enumerate(zip(selections, expected_features, strict=True)):
        problem = extract_problem_features(problems[str(row_index)])
        baseline_predictors = extract_raw_features(record)
        for display_name, value_factory in mutations.items():
            record_field = {
                "bRate": "brate",
                "bRate_std": "brate_std",
            }.get(display_name, display_name)
            changed = replace(record, **{record_field: value_factory(record)})
            changed_predictors = extract_raw_features(changed)
            changed_features = engineer_behavioral_features(changed_predictors, problem)
            if changed_predictors != baseline_predictors or changed_features != expected:
                raise ValueError(
                    f"Forbidden field {display_name} changed predictors at row {row_index}"
                )
            comparisons += 1
    return {
        "status": "PASS",
        "method": "exact_metamorphic_forbidden_field_mutation",
        "rows_checked": len(selections),
        "fields_checked": list(mutations),
        "comparisons": comparisons,
        "engineered_predictor_changes": 0,
    }


def validate_feature_rows(
    raw_rows: list[RawFeatureRow], engineered_rows: list[EngineeredFeatureRow]
) -> dict[str, Any]:
    """Validate generated rows before they are written to processed storage."""
    if not raw_rows or len(raw_rows) != len(engineered_rows):
        raise ValueError("Raw and engineered feature tables must have equal nonzero row counts")
    indicator_names = {
        "ambiguity_indicator",
        "feedback_indicator",
        "lottery_shape_b_undefined",
        "lottery_shape_b_symmetric",
        "lottery_shape_b_right_skewed",
        "lottery_shape_b_left_skewed",
        "correlation_negative",
        "correlation_zero",
        "correlation_positive",
    }
    probability_names = {
        "best_payoff_probability_a",
        "best_payoff_probability_b_oracle",
        "loss_probability_a",
        "loss_probability_b_oracle",
    }
    probability_difference_names = {
        "best_payoff_probability_difference_b_minus_a_oracle",
        "loss_probability_difference_b_minus_a_oracle",
    }
    nonnegative_names = {
        "payoff_range_a",
        "payoff_range_b",
        "payoff_std_a",
        "payoff_std_b_oracle",
    }
    for row_number, row in enumerate(engineered_rows):
        values = asdict(row)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"Engineered row {row_number} contains a non-finite value")
        if any(values[name] not in {0, 1} for name in indicator_names):
            raise ValueError(f"Engineered row {row_number} contains an invalid indicator")
        if sum(values[name] for name in indicator_names if name.startswith("lottery_shape")) != 1:
            raise ValueError(f"Engineered row {row_number} has invalid lottery-shape encoding")
        if sum(values[name] for name in indicator_names if name.startswith("correlation")) != 1:
            raise ValueError(f"Engineered row {row_number} has invalid correlation encoding")
        if any(not 0.0 <= values[name] <= 1.0 for name in probability_names):
            raise ValueError(f"Engineered row {row_number} contains an invalid probability")
        if any(not -1.0 <= values[name] <= 1.0 for name in probability_difference_names):
            raise ValueError(
                f"Engineered row {row_number} contains an invalid probability difference"
            )
        if any(values[name] < 0.0 for name in nonnegative_names):
            raise ValueError(f"Engineered row {row_number} contains an invalid spread")
    return {
        "status": "PASS",
        "rows": len(engineered_rows),
        "non_finite_values": 0,
        "invalid_indicators": 0,
        "lottery_shape_one_hot_violations": 0,
        "correlation_one_hot_violations": 0,
        "invalid_probabilities": 0,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
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


def write_feature_documentation(path: Path = DEFAULT_DOCUMENTATION) -> None:
    """Generate the mathematical feature reference from the audited contract."""
    lines = [
        "# Behavioral feature reference",
        "",
        (
            "This document is generated from `FEATURE_DEFINITIONS` by "
            "`decisionlab-build-features`. Engineered predictors are separate from the raw "
            "predictor table. `row_index` and `problem` appear in output files only as "
            "join/grouping keys and must not be model inputs. No feature uses `bRate`, "
            "`bRate_std`, `n`, or `Block`."
        ),
        (
            "The production feature function accepts only `RawFeatureRow` and "
            "`ProblemFeatureInput`. These types cannot represent targets, participant counts, "
            "blocks, identifiers, structural fingerprints, or fold metadata. Passing a full "
            "selection record is rejected at runtime."
        ),
        "",
        "## Feature contracts",
        "",
    ]
    for definition in FEATURE_DEFINITIONS:
        lines.extend(
            [
                f"### `{definition.name}`",
                "",
                f"- **Formula:** {definition.formula}",
                f"- **Interpretation:** {definition.interpretation}",
                f"- **Assumptions:** {definition.assumptions}",
                f"- **Potential limitations:** {definition.limitations}",
                f"- **Availability:** {definition.availability}",
                f"- **Leakage audit:** PASS — {definition.leakage_audit}",
                "",
            ]
        )
    lines.extend(
        [
            "## Deliberately excluded transformations",
            "",
            (
                "- `pHb - pHa` is not included because these probabilities refer to different "
                "events: entering B's sublottery versus obtaining A's high outcome."
            ),
            (
                "- Ratios of expected values or payoffs are not included because zero and "
                "negative denominators make their behavioral interpretation unstable."
            ),
            (
                "- Polynomial, logarithmic, rank, quantile, and target-encoded transformations "
                "are not included without a prespecified behavioral rationale."
            ),
            (
                "- `bRate_std`, `n`, and `Block` are excluded from predictors. The first is "
                "target-derived, the second is measurement metadata, and the third "
                "deterministically encodes feedback status in this dataset."
            ),
            "",
            "## Participant-visible versus oracle features",
            "",
            (
                "Names ending in `_oracle`, and interactions built from them, use objective "
                "Gamble B probabilities. They may be used for an explicitly labeled "
                "design/oracle analysis. When `Amb=True`, they must not be described as "
                "information participants could calculate from the choice display. Future "
                "participant-visible models must omit or mask these features for ambiguous "
                "rows rather than silently treating them as observed."
            ),
            "",
            (
                "Payoff-support features such as ranges and extrema do not use probabilities "
                "and remain participant-visible under the documented ambiguity manipulation."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_feature_tables(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_output: Path = DEFAULT_RAW_FEATURES,
    engineered_output: Path = DEFAULT_ENGINEERED_FEATURES,
    summary_output: Path = DEFAULT_SUMMARY,
    documentation_output: Path = DEFAULT_DOCUMENTATION,
) -> dict[str, Any]:
    """Validate raw data and write separate raw and engineered feature tables."""
    validation = load_and_validate(raw_dir, manifest_path)
    selections = load_selections(raw_dir / "c13k_selections.csv")
    problems = load_problems(raw_dir / "c13k_problems.json", selections)
    leakage_audit = audit_feature_contract()

    raw_features: list[RawFeatureRow] = []
    engineered_features: list[EngineeredFeatureRow] = []
    for row_index, record in enumerate(selections):
        predictors = extract_raw_features(record)
        problem = extract_problem_features(problems[str(row_index)])
        raw_features.append(predictors)
        engineered_features.append(engineer_behavioral_features(predictors, problem))
    leakage_audit["metamorphic_invariance"] = audit_forbidden_field_invariance(
        selections, problems, engineered_features
    )
    output_validation = validate_feature_rows(raw_features, engineered_features)
    raw_rows = [
        {"row_index": index, "problem": record.problem} | asdict(feature_row)
        for index, (record, feature_row) in enumerate(zip(selections, raw_features, strict=True))
    ]
    engineered_rows = [
        {"row_index": index, "problem": record.problem} | asdict(feature_row)
        for index, (record, feature_row) in enumerate(
            zip(selections, engineered_features, strict=True)
        )
    ]

    raw_columns = ["row_index", "problem", *[field.name for field in fields(RawFeatureRow)]]
    engineered_columns = [
        "row_index",
        "problem",
        *[field.name for field in fields(EngineeredFeatureRow)],
    ]
    _write_rows(raw_output, raw_rows, raw_columns)
    _write_rows(engineered_output, engineered_rows, engineered_columns)
    write_feature_documentation(documentation_output)

    summary = {
        "dataset": validation["dataset"],
        "source_commit": validation["source_commit"],
        "source_sha256": validation["sha256"],
        "decisionlab_version": __version__,
        "rows": len(selections),
        "raw_feature_columns": [field.name for field in fields(RawFeatureRow)],
        "engineered_feature_columns": [field.name for field in fields(EngineeredFeatureRow)],
        "identifier_columns_not_features": ["row_index", "problem"],
        "excluded_columns": [
            "bRate",
            "bRate_std",
            "n",
            "Block",
            "problem",
            "row_index",
            "structural_fingerprint",
            "outer_fold",
            "inner_fold",
        ],
        "leakage_audit": leakage_audit,
        "output_validation": output_validation,
        "outputs": {
            str(raw_output.relative_to(PROJECT_ROOT)): sha256_file(raw_output),
            str(engineered_output.relative_to(PROJECT_ROOT)): sha256_file(engineered_output),
            str(documentation_output.relative_to(PROJECT_ROOT)): sha256_file(documentation_output),
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_feature_build(
    raw_dir: Path = DEFAULT_DESTINATION,
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_output: Path = DEFAULT_RAW_FEATURES,
    engineered_output: Path = DEFAULT_ENGINEERED_FEATURES,
    summary_output: Path = DEFAULT_SUMMARY,
    documentation_output: Path = DEFAULT_DOCUMENTATION,
    provenance_output: Path = DEFAULT_PROVENANCE,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the standalone feature build with centralized provenance."""
    provenance = start_run_provenance(
        experiment_name="behavioral_feature_build",
        config_paths=[],
        configuration_values={
            "raw_output": str(raw_output),
            "engineered_output": str(engineered_output),
            "summary_output": str(summary_output),
            "documentation_output": str(documentation_output),
            "allow_dirty": allow_dirty,
        },
        dataset_manifest_path=manifest_path,
        raw_dir=raw_dir,
        fold_specification_identifier="not_applicable_feature_build",
        entry_module=Path(__file__),
        allow_dirty=allow_dirty,
    )
    summary = build_feature_tables(
        raw_dir,
        manifest_path,
        raw_output,
        engineered_output,
        summary_output,
        documentation_output,
    )
    finalize_run_provenance(
        provenance,
        fold_artifacts={},
        output_artifacts=[
            raw_output,
            engineered_output,
            summary_output,
            documentation_output,
        ],
        output_path=provenance_output,
    )
    return summary


def main() -> None:
    """Run the reproducible behavioral feature-build stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_FEATURES)
    parser.add_argument("--engineered-output", type=Path, default=DEFAULT_ENGINEERED_FEATURES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--documentation", type=Path, default=DEFAULT_DOCUMENTATION)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a clearly marked non-official run from a dirty worktree.",
    )
    args = parser.parse_args()
    summary = run_feature_build(
        args.raw_dir,
        args.manifest,
        args.raw_output,
        args.engineered_output,
        args.summary,
        args.documentation,
        args.provenance,
        allow_dirty=args.allow_dirty,
    )
    print(
        f"Feature build: PASS; rows={summary['rows']:,}; "
        f"raw={len(summary['raw_feature_columns'])}; "
        f"engineered={len(summary['engineered_feature_columns'])}; "
        f"leakage audit={summary['leakage_audit']['status']}"
    )


if __name__ == "__main__":
    main()
