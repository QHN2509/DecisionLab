from __future__ import annotations

from dataclasses import replace

import joblib
import numpy as np
import pytest

from decisionlab.app.prediction import (
    PredictionBundle,
    format_gamble_rows,
    load_prediction_bundle,
    predict_scenario,
    what_if_predictions,
)
from decisionlab.features.behavioral import ScenarioFeatureInput


class _InvalidPredictionPipeline:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([self.value])


def _default_scenario() -> ScenarioFeatureInput:
    return ScenarioFeatureInput(
        high_payoff_a=26,
        high_probability_a=0.95,
        low_payoff_a=-1,
        sublottery_mean_b=23,
        sublottery_probability_b=0.05,
        low_payoff_b=21,
        lottery_shape_b=0,
        lottery_outcomes_b=1,
        ambiguity=False,
        feedback=False,
        correlation=0,
    )


@pytest.fixture(scope="module")
def bundle() -> PredictionBundle:
    return load_prediction_bundle()


def test_production_prediction_smoke(bundle: PredictionBundle) -> None:

    result = predict_scenario(_default_scenario(), bundle)

    assert 0.0 <= result.predicted_b_rate <= 1.0
    assert result.predicted_a_rate == 1.0 - result.predicted_b_rate
    assert result.expected_value_a == pytest.approx(24.65)
    assert result.expected_value_b == pytest.approx(21.1)
    assert result.expected_value_benchmark == "Gamble A"
    assert len(result.feature_values) == len(bundle.feature_names) == 28
    assert [row["behavioral_domain"] for row in result.driver_rows[:3]] == [
        "expected value",
        "payoff and risk",
        "probability structure",
    ]


def test_what_if_predictions_are_coherent_and_deterministic(bundle: PredictionBundle) -> None:
    scenario = _default_scenario()

    first = what_if_predictions(scenario, bundle)
    second = what_if_predictions(scenario, bundle)

    assert [row["scenario"] for row in first] == [row["scenario"] for row in second]
    assert [row["feedback"] for row in first] == [row["feedback"] for row in second]
    assert [row["ambiguity"] for row in first] == [row["ambiguity"] for row in second]
    assert [row["predicted_b_rate"] for row in first] == pytest.approx(
        [row["predicted_b_rate"] for row in second], abs=1e-12
    )
    assert [row["scenario"] for row in first] == [
        "Current scenario",
        "Toggle feedback",
        "Toggle ambiguity",
        "Toggle both",
    ]
    assert first[0]["change_from_current"] == 0.0
    assert first[1]["feedback"] is True and first[1]["ambiguity"] is False
    assert first[2]["feedback"] is False and first[2]["ambiguity"] is True
    assert all(0.0 <= row["predicted_b_rate"] <= 1.0 for row in first)


def test_gamble_rows_hide_zero_probability_outcomes() -> None:
    rows = format_gamble_rows([[1.0, 4.0], [0.0, -2.0]])

    assert rows == [{"Probability": 1.0, "Payoff": 4.0}]


def test_default_feature_vector_is_finite(bundle: PredictionBundle) -> None:
    result = predict_scenario(_default_scenario(), bundle)

    assert np.all(np.isfinite(list(result.feature_values.values())))


@pytest.mark.parametrize("invalid_prediction", [-0.01, 1.01, float("nan")])
def test_prediction_service_rejects_invalid_model_outputs(
    bundle: PredictionBundle, invalid_prediction: float
) -> None:
    invalid_bundle = replace(
        bundle,
        pipeline=_InvalidPredictionPipeline(invalid_prediction),
    )

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        predict_scenario(_default_scenario(), invalid_bundle)


def test_bundle_uses_official_nested_outer_oof_metadata(bundle: PredictionBundle) -> None:
    assert bundle.selected_model == "random_forest"
    assert bundle.validation_metric_scope == "equal-structural-group outer OOF"
    assert bundle.validation_mae == pytest.approx(0.08000781129428924)
    assert bundle.oracle_feature_count == 10


def test_production_pipeline_round_trip_is_reproducible(bundle: PredictionBundle, tmp_path) -> None:
    output = tmp_path / "selected_pipeline.joblib"
    joblib.dump(bundle.pipeline, output)
    restored_bundle = replace(bundle, pipeline=joblib.load(output))

    original = predict_scenario(_default_scenario(), bundle)
    restored = predict_scenario(_default_scenario(), restored_bundle)

    assert restored.predicted_b_rate == pytest.approx(
        original.predicted_b_rate,
        rel=0.0,
        abs=5e-15,
    )
    assert restored.feature_values == original.feature_values
