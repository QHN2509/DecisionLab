from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from decisionlab.analysis.behavioral import (
    condition_sensitivity_rows,
    error_slice_rows,
    normative_case_rows,
    quantile_relationship_rows,
    select_normative_examples,
    summarize_normative_cases,
)
from decisionlab.features.behavioral import EngineeredFeatureRow


def _all_feature_names() -> list[str]:
    return [field.name for field in fields(EngineeredFeatureRow)]


def test_quantile_relationship_rows_cover_all_observations() -> None:
    values = np.arange(100, dtype=float)
    target = values / 100.0
    predicted = target + 0.01

    rows = quantile_relationship_rows(
        values,
        target,
        predicted,
        feature="example",
        domain="example_domain",
        bins=5,
    )

    assert sum(row["rows"] for row in rows) == 100
    assert all(row["condition_row_mae"] == pytest.approx(0.01) for row in rows)


def test_error_slices_use_declared_thresholds_and_cover_each_dimension() -> None:
    names = _all_feature_names()
    features = np.zeros((6, len(names)))
    position = {name: index for index, name in enumerate(names)}
    features[:, position["feedback_indicator"]] = [0, 0, 0, 1, 1, 1]
    features[:, position["ambiguity_indicator"]] = [0, 1, 0, 1, 0, 1]
    features[:, position["expected_value_difference_b_minus_a_oracle"]] = [
        -2,
        -1,
        0,
        1,
        2,
        3,
    ]
    target = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    predicted = target + 0.05
    counts = np.asarray([15, 15, 16, 16, 17, 18])
    groups = np.asarray(["a", "a", "b", "c", "d", "e"])

    rows = error_slice_rows(
        features,
        names,
        target,
        predicted,
        groups,
        counts,
        ev_threshold=1.0,
        participant_threshold=16,
    )

    for dimension in ("feedback", "ambiguity", "expected_value_regime", "participant_count"):
        assert sum(row["rows"] for row in rows if row["dimension"] == dimension) == 6
    assert all(row["problem_group_mae"] == pytest.approx(0.05) for row in rows)
    assert all(row["condition_row_mae"] == pytest.approx(0.05) for row in rows)


def test_condition_sensitivity_updates_binary_interaction() -> None:
    names = _all_feature_names()
    features = np.zeros((3, len(names)))
    position = {name: index for index, name in enumerate(names)}
    features[:, position["expected_value_difference_b_minus_a_oracle"]] = [1.0, 2.0, 3.0]

    def predict(values: np.ndarray) -> np.ndarray:
        return (
            0.4
            + 0.01 * values[:, position["feedback_indicator"]]
            + 0.02 * values[:, position["expected_value_difference_oracle_x_feedback"]]
        )

    rows = condition_sensitivity_rows(predict, features, names)
    feedback_on = next(
        row for row in rows if row["dimension"] == "feedback" and row["level"] == "1"
    )

    assert feedback_on["mean_difference_from_reference"] == pytest.approx(0.05)


def test_normative_cases_apply_declared_benchmark_and_prediction_rules() -> None:
    names = [
        "expected_value_a",
        "expected_value_b_oracle",
        "expected_value_difference_b_minus_a_oracle",
    ]
    differences = np.asarray([-6.0, 6.0, 0.2, -7.0, 7.0])
    features = np.column_stack([np.full(5, 10.0), 10.0 + differences, differences])
    target = np.asarray([0.7, 0.3, 0.5, 0.8, 0.2])
    predicted = np.asarray([0.68, 0.7, 0.52, 0.4, 0.19])
    records = [
        SimpleNamespace(
            problem=index,
            feedback=False,
            amb=False,
            n=16,
            ha=10,
            pha=1.0,
            la=10,
            hb=10,
            phb=1.0,
            lb=10,
            lot_shape_b=0,
            corr=0,
        )
        for index in range(5)
    ]
    problems = {
        str(index): {"A": [[1.0, 10.0]], "B": [[1.0, 10.0 + differences[index]]]}
        for index in range(5)
    }
    settings = {
        "strong_expected_value_difference": 5.0,
        "human_favor_b_threshold": 0.6,
        "human_favor_a_threshold": 0.4,
        "highly_divided_half_width": 0.05,
        "successful_prediction_max_absolute_error": 0.08,
        "failed_prediction_min_absolute_error": 0.15,
        "examples_per_category": 2,
    }

    rows = normative_case_rows(
        features,
        names,
        target,
        predicted,
        np.arange(5),
        records,
        problems,
        settings,
    )
    counts = summarize_normative_cases(rows)
    examples = select_normative_examples(rows, examples_per_category=2)

    assert counts == {
        "strong_ev_a_humans_favor_b": 2,
        "strong_ev_b_humans_favor_a": 2,
        "highly_divided": 1,
        "ml_successfully_predicts_deviation": 2,
        "ml_fails_to_predict_deviation": 2,
    }
    assert len(examples) == 9
    assert rows[0]["gamble_a"] == "p=1→10"
