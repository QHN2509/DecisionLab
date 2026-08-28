"""Testable prediction services for the DecisionLab Streamlit application."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from decisionlab.data.fetch import sha256_file
from decisionlab.features.behavioral import (
    ScenarioFeatureInput,
    engineer_scenario_features,
    summarize_option,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SELECTION_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "model_selection"
DEFAULT_MODEL_METRICS = MODEL_SELECTION_DIR / "metrics.json"
DEFAULT_PIPELINE = MODEL_SELECTION_DIR / "selected_pipeline.joblib"
DEFAULT_FEATURE_NAMES = MODEL_SELECTION_DIR / "feature_names.json"
DEFAULT_ENGINEERED_FEATURES = (
    PROJECT_ROOT / "data" / "processed" / "choices13k_engineered_features.csv"
)
DEFAULT_BEHAVIORAL_STATISTICS = (
    PROJECT_ROOT / "artifacts" / "analysis" / "behavioral" / "statistics.json"
)

SHAPE_LABELS = {
    0: "Single outcome",
    1: "Symmetric",
    2: "Right-skewed",
    3: "Left-skewed",
}
CORRELATION_LABELS = {-1: "Negative", 0: "None", 1: "Positive"}


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    """Verified model artifacts and target-free training feature reference."""

    pipeline: Any
    feature_names: list[str]
    feature_reference: dict[str, dict[str, float]]
    domain_importance: list[dict[str, Any]]
    selected_model: str
    validation_mae: float
    validation_rmse: float
    oracle_feature_count: int
    source_commit: str


@dataclass(frozen=True, slots=True)
class ScenarioPrediction:
    """All non-UI values needed to render one research-dashboard prediction."""

    predicted_b_rate: float
    predicted_a_rate: float
    expected_value_a: float
    expected_value_b: float
    expected_value_difference: float
    expected_value_benchmark: str
    gamble_a: list[list[float]]
    gamble_b: list[list[float]]
    feature_values: dict[str, float]
    driver_rows: list[dict[str, Any]]
    outside_training_range: list[str]


def _read_feature_reference(path: Path, feature_names: list[str]) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not set(feature_names).issubset(rows[0]):
        raise ValueError("Processed feature reference does not match model feature names")
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
    return {
        name: {
            "min": float(np.min(matrix[:, index])),
            "median": float(np.median(matrix[:, index])),
            "max": float(np.max(matrix[:, index])),
        }
        for index, name in enumerate(feature_names)
    }


def load_prediction_bundle(
    metrics_path: Path = DEFAULT_MODEL_METRICS,
    pipeline_path: Path = DEFAULT_PIPELINE,
    feature_names_path: Path = DEFAULT_FEATURE_NAMES,
    feature_reference_path: Path = DEFAULT_ENGINEERED_FEATURES,
    behavioral_statistics_path: Path = DEFAULT_BEHAVIORAL_STATISTICS,
) -> PredictionBundle:
    """Load and verify every artifact used for interactive prediction."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_document = json.loads(feature_names_path.read_text(encoding="utf-8"))
    behavioral = json.loads(behavioral_statistics_path.read_text(encoding="utf-8"))
    if metrics["test_rows_predicted"] != 0 or metrics["test_metrics_computed"]:
        raise ValueError("Selected model no longer satisfies the locked-test contract")
    expected_hash = metrics["outputs"][
        "artifacts/experiments/model_selection/selected_pipeline.joblib"
    ]
    if sha256_file(pipeline_path) != expected_hash:
        raise ValueError("Selected pipeline hash does not match experiment metadata")
    feature_names = feature_document["feature_names"]
    if feature_names != metrics["feature_names"]:
        raise ValueError("Feature order differs between saved model artifacts")
    if behavioral["selected_pipeline_sha256"] != expected_hash:
        raise ValueError("Behavioral interpretation refers to a different selected pipeline")
    selected = metrics["selected_model"]
    selected_metrics = metrics["candidates"][selected]["validation_metrics"]["unweighted"]
    return PredictionBundle(
        pipeline=joblib.load(pipeline_path),
        feature_names=feature_names,
        feature_reference=_read_feature_reference(feature_reference_path, feature_names),
        domain_importance=behavioral["permutation_importance"]["domains"],
        selected_model=selected,
        validation_mae=selected_metrics["mae"],
        validation_rmse=selected_metrics["rmse"],
        oracle_feature_count=metrics["oracle_feature_count"],
        source_commit=metrics["source_commit"],
    )


def _feature_vector(
    scenario: ScenarioFeatureInput, feature_names: list[str]
) -> tuple[np.ndarray, dict[str, float], dict[str, list[list[float]]]]:
    engineered, problem = engineer_scenario_features(scenario)
    values = {name: float(value) for name, value in asdict(engineered).items()}
    if set(values) != set(feature_names):
        raise ValueError("Scenario feature contract differs from selected model")
    return np.asarray([[values[name] for name in feature_names]]), values, problem


def _driver_summary(domain: str, scenario: ScenarioFeatureInput, features: dict[str, float]) -> str:
    summaries = {
        "expected_value": (
            f"EV(B) − EV(A) = {features['expected_value_difference_b_minus_a_oracle']:+.2f}"
        ),
        "payoff_and_risk": (
            "B − A payoff SD = "
            f"{features['payoff_std_difference_b_minus_a_oracle']:+.2f}; range difference = "
            f"{features['payoff_range_difference_b_minus_a']:+.2f}"
        ),
        "probability_structure": (
            "B − A loss probability = "
            f"{features['loss_probability_difference_b_minus_a_oracle']:+.2f}"
        ),
        "ambiguity": "B probabilities hidden" if scenario.ambiguity else "Probabilities known",
        "lottery_structure": (
            f"{SHAPE_LABELS[scenario.lottery_shape_b]}; "
            f"{CORRELATION_LABELS[scenario.correlation].lower()} correlation"
        ),
        "feedback": "Feedback available" if scenario.feedback else "No feedback",
    }
    return summaries[domain]


def predict_scenario(
    scenario: ScenarioFeatureInput, bundle: PredictionBundle
) -> ScenarioPrediction:
    """Engineer and predict one scenario with the persisted production pipeline."""
    vector, values, problem = _feature_vector(scenario, bundle.feature_names)
    predicted_b = float(bundle.pipeline.predict(vector)[0])
    if not 0.0 <= predicted_b <= 1.0:
        raise ValueError("Production pipeline returned a prediction outside [0, 1]")
    option_a = summarize_option(problem["A"])
    option_b = summarize_option(problem["B"])
    difference = option_b.expected_value - option_a.expected_value
    benchmark = "Gamble B" if difference > 0 else "Gamble A" if difference < 0 else "Tie"
    outside = [
        name
        for name, value in values.items()
        if value < bundle.feature_reference[name]["min"]
        or value > bundle.feature_reference[name]["max"]
    ]
    drivers = [
        {
            "behavioral_domain": row["name"].replace("_", " "),
            "validation_mae_increase": row["mean_mae_increase"],
            "scenario_value": _driver_summary(row["name"], scenario, values),
        }
        for row in bundle.domain_importance
    ]
    return ScenarioPrediction(
        predicted_b_rate=predicted_b,
        predicted_a_rate=1.0 - predicted_b,
        expected_value_a=option_a.expected_value,
        expected_value_b=option_b.expected_value,
        expected_value_difference=difference,
        expected_value_benchmark=benchmark,
        gamble_a=problem["A"],
        gamble_b=problem["B"],
        feature_values=values,
        driver_rows=drivers,
        outside_training_range=outside,
    )


def what_if_predictions(
    scenario: ScenarioFeatureInput, bundle: PredictionBundle
) -> list[dict[str, Any]]:
    """Compare the current prediction with coherent feedback/ambiguity toggles."""
    variants = (
        ("Current scenario", scenario),
        ("Toggle feedback", replace(scenario, feedback=not scenario.feedback)),
        ("Toggle ambiguity", replace(scenario, ambiguity=not scenario.ambiguity)),
        (
            "Toggle both",
            replace(
                scenario,
                feedback=not scenario.feedback,
                ambiguity=not scenario.ambiguity,
            ),
        ),
    )
    predictions = [(label, predict_scenario(value, bundle)) for label, value in variants]
    baseline = predictions[0][1].predicted_b_rate
    return [
        {
            "scenario": label,
            "feedback": value.feedback,
            "ambiguity": value.ambiguity,
            "predicted_b_rate": result.predicted_b_rate,
            "change_from_current": result.predicted_b_rate - baseline,
        }
        for (label, value), (_, result) in zip(variants, predictions, strict=True)
    ]


def format_gamble_rows(outcomes: list[list[float]]) -> list[dict[str, float]]:
    """Return nonzero-probability outcomes suitable for a UI table."""
    return [
        {"Probability": float(probability), "Payoff": float(payoff)}
        for probability, payoff in outcomes
        if probability > 1e-12
    ]
