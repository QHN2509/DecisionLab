from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from decisionlab.data.validation import SelectionRecord
from decisionlab.features.behavioral import (
    FEATURE_DEFINITIONS,
    EngineeredFeatureRow,
    RawFeatureRow,
    ScenarioFeatureInput,
    audit_feature_contract,
    build_sublottery_distribution,
    engineer_behavioral_features,
    engineer_scenario_features,
    extract_raw_features,
    summarize_option,
    validate_feature_rows,
    write_feature_documentation,
)


@pytest.fixture
def selection() -> SelectionRecord:
    return SelectionRecord(
        problem=1,
        feedback=True,
        n=16,
        block=2,
        ha=12,
        pha=0.25,
        la=-4,
        hb=10,
        phb=0.5,
        lb=2,
        lot_shape_b=0,
        lot_num_b=1,
        amb=True,
        corr=-1,
        brate=0.75,
        brate_std=0.2,
    )


@pytest.fixture
def problem() -> dict[str, list[list[float]]]:
    return {
        "A": [[0.25, 12.0], [0.75, -4.0]],
        "B": [[0.5, 10.0], [0.5, 2.0]],
    }


def test_expected_value_and_relative_payoff_formulas(
    selection: SelectionRecord, problem: dict[str, list[list[float]]]
) -> None:
    features = engineer_behavioral_features(selection, problem)

    assert features.expected_value_a == pytest.approx(0.0)
    assert features.expected_value_b_oracle == pytest.approx(6.0)
    assert features.expected_value_difference_b_minus_a_oracle == pytest.approx(6.0)
    assert features.payoff_range_a == pytest.approx(16.0)
    assert features.payoff_range_b == pytest.approx(8.0)
    assert features.payoff_range_difference_b_minus_a == pytest.approx(-8.0)
    assert features.maximum_payoff_difference_b_minus_a == pytest.approx(-2.0)
    assert features.minimum_payoff_difference_b_minus_a == pytest.approx(6.0)


def test_dispersion_and_probability_formulas(
    selection: SelectionRecord, problem: dict[str, list[list[float]]]
) -> None:
    features = engineer_behavioral_features(selection, problem)

    assert features.payoff_std_a == pytest.approx(48.0**0.5)
    assert features.payoff_std_b_oracle == pytest.approx(4.0)
    assert features.payoff_std_difference_b_minus_a_oracle == pytest.approx(4.0 - 48.0**0.5)
    assert features.best_payoff_probability_a == pytest.approx(0.25)
    assert features.best_payoff_probability_b_oracle == pytest.approx(0.5)
    assert features.best_payoff_probability_difference_b_minus_a_oracle == pytest.approx(0.25)
    assert features.loss_probability_a == pytest.approx(0.75)
    assert features.loss_probability_b_oracle == pytest.approx(0.0)
    assert features.loss_probability_difference_b_minus_a_oracle == pytest.approx(-0.75)


def test_tied_best_outcomes_sum_probabilities() -> None:
    summary = summarize_option([[0.3, 5.0], [0.7, 5.0]])

    assert summary.best_payoff_probability == pytest.approx(1.0)
    assert summary.payoff_range == pytest.approx(0.0)
    assert summary.payoff_std == pytest.approx(0.0)


def test_indicators_and_interactions_are_behaviorally_defined(
    selection: SelectionRecord, problem: dict[str, list[list[float]]]
) -> None:
    features = engineer_behavioral_features(selection, problem)

    assert features.ambiguity_indicator == 1
    assert features.feedback_indicator == 1
    assert (
        sum(
            [
                features.lottery_shape_b_undefined,
                features.lottery_shape_b_symmetric,
                features.lottery_shape_b_right_skewed,
                features.lottery_shape_b_left_skewed,
            ]
        )
        == 1
    )
    assert features.lottery_shape_b_undefined == 1
    assert (
        sum(
            [
                features.correlation_negative,
                features.correlation_zero,
                features.correlation_positive,
            ]
        )
        == 1
    )
    assert features.correlation_negative == 1
    assert features.expected_value_difference_oracle_x_feedback == pytest.approx(6.0)
    assert features.expected_value_difference_oracle_x_ambiguity == pytest.approx(6.0)


def test_raw_features_are_separate_from_identifiers_metadata_and_targets(
    selection: SelectionRecord,
) -> None:
    raw = extract_raw_features(selection)
    names = {field.name for field in fields(RawFeatureRow)}

    assert raw.feedback is True
    assert names.isdisjoint({"problem", "n", "block", "brate", "brate_std"})


def test_every_engineered_feature_has_a_passing_leakage_audit() -> None:
    audit = audit_feature_contract()
    output_names = [field.name for field in fields(EngineeredFeatureRow)]

    assert [definition.name for definition in FEATURE_DEFINITIONS] == output_names
    assert audit["status"] == "PASS"
    assert set(audit["features"]) == set(output_names)
    assert all(not item["target_derived"] for item in audit["features"].values())
    assert all(item["status"] == "PASS" for item in audit["features"].values())


def test_generated_rows_pass_output_validation(
    selection: SelectionRecord, problem: dict[str, list[list[float]]]
) -> None:
    validation = validate_feature_rows(
        [extract_raw_features(selection)],
        [engineer_behavioral_features(selection, problem)],
    )

    assert validation["status"] == "PASS"
    assert validation["rows"] == 1
    assert validation["non_finite_values"] == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("ambiguity_indicator", 2, "invalid indicator"),
        ("best_payoff_probability_a", 1.01, "invalid probability"),
        (
            "loss_probability_difference_b_minus_a_oracle",
            -1.01,
            "invalid probability difference",
        ),
        ("payoff_range_b", -0.01, "invalid spread"),
        ("payoff_std_a", float("nan"), "non-finite"),
    ],
)
def test_feature_validation_rejects_semantic_bound_violations(
    selection: SelectionRecord,
    problem: dict[str, list[list[float]]],
    field_name: str,
    invalid_value: float,
    message: str,
) -> None:
    engineered = engineer_behavioral_features(selection, problem)

    with pytest.raises(ValueError, match=message):
        validate_feature_rows(
            [extract_raw_features(selection)],
            [replace(engineered, **{field_name: invalid_value})],
        )


def test_feature_difference_identities_hold(
    selection: SelectionRecord, problem: dict[str, list[list[float]]]
) -> None:
    feature = engineer_behavioral_features(selection, problem)

    assert feature.expected_value_difference_b_minus_a_oracle == pytest.approx(
        feature.expected_value_b_oracle - feature.expected_value_a
    )
    assert feature.payoff_range_difference_b_minus_a == pytest.approx(
        feature.payoff_range_b - feature.payoff_range_a
    )
    assert feature.loss_probability_difference_b_minus_a_oracle == pytest.approx(
        feature.loss_probability_b_oracle - feature.loss_probability_a
    )
    assert feature.expected_value_difference_oracle_x_feedback == pytest.approx(
        feature.expected_value_difference_b_minus_a_oracle * feature.feedback_indicator
    )


def test_generated_documentation_covers_every_feature(tmp_path: Path) -> None:
    output = tmp_path / "features.md"

    write_feature_documentation(output)
    documentation = output.read_text(encoding="utf-8")

    for definition in FEATURE_DEFINITIONS:
        assert f"### `{definition.name}`" in documentation
        assert definition.formula in documentation
    assert "Potential limitations" in documentation
    assert "Leakage audit" in documentation


@pytest.mark.parametrize("shape", [0, 1, 2, 3])
def test_constructed_sublottery_has_requested_probability_and_mean(shape: int) -> None:
    outcome_count = 1 if shape == 0 else 6

    outcomes = build_sublottery_distribution(12.0, shape, outcome_count)

    assert sum(probability for probability, _ in outcomes) == pytest.approx(1.0)
    assert sum(probability * payoff for probability, payoff in outcomes) == pytest.approx(12.0)
    assert len(outcomes) == outcome_count


def test_scenario_uses_production_feature_engineering() -> None:
    scenario = ScenarioFeatureInput(
        high_payoff_a=26,
        high_probability_a=0.95,
        low_payoff_a=-1,
        sublottery_mean_b=23,
        sublottery_probability_b=0.05,
        low_payoff_b=21,
        lottery_shape_b=0,
        lottery_outcomes_b=1,
        ambiguity=True,
        feedback=True,
        correlation=0,
    )

    features, problem = engineer_scenario_features(scenario)

    assert features.expected_value_a == pytest.approx(24.65)
    assert features.expected_value_b_oracle == pytest.approx(21.1)
    assert features.ambiguity_indicator == 1
    assert features.feedback_indicator == 1
    assert [row[0] for row in problem["B"]] == pytest.approx([0.95, 0.05])
    assert [row[1] for row in problem["B"]] == [21, 23]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"high_payoff_a": float("nan")}, "finite"),
        ({"high_probability_a": 1.01}, "probability must be"),
        ({"sublottery_probability_b": -0.01}, "probability must be"),
        ({"high_payoff_a": -2.0, "low_payoff_a": -1.0}, "high payoff"),
        ({"lottery_shape_b": 4}, "Lottery shape"),
        ({"lottery_shape_b": 1, "lottery_outcomes_b": 1}, "other shapes require"),
        ({"correlation": 2}, "Correlation category"),
        ({"feedback": "False"}, "must be boolean"),
        ({"ambiguity": 0}, "must be boolean"),
        ({"low_payoff_b": "zero"}, "must be numbers"),
    ],
)
def test_scenario_rejects_malformed_user_inputs(changes: dict[str, object], message: str) -> None:
    scenario = ScenarioFeatureInput(
        high_payoff_a=10.0,
        high_probability_a=0.5,
        low_payoff_a=0.0,
        sublottery_mean_b=8.0,
        sublottery_probability_b=0.5,
        low_payoff_b=2.0,
        lottery_shape_b=0,
        lottery_outcomes_b=1,
        ambiguity=False,
        feedback=False,
        correlation=0,
    )

    with pytest.raises(ValueError, match=message):
        engineer_scenario_features(replace(scenario, **changes))
