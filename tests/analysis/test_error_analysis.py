"""Tests for systematic selected-model error analysis."""

from __future__ import annotations

import numpy as np
import pytest

from decisionlab.analysis.errors import (
    build_case_rows,
    clustered_error_interval,
    participant_count_analysis,
    regime_masks,
    select_representative_failures,
)
from decisionlab.data.validation import SelectionRecord

FEATURE_NAMES = [
    "feedback_indicator",
    "ambiguity_indicator",
    "expected_value_a",
    "expected_value_b_oracle",
    "expected_value_difference_b_minus_a_oracle",
    "lottery_shape_b_undefined",
    "lottery_shape_b_symmetric",
    "lottery_shape_b_right_skewed",
    "lottery_shape_b_left_skewed",
]


def test_clustered_error_interval_is_deterministic_and_valid() -> None:
    target = np.asarray([0.1, 0.2, 0.7, 0.8])
    predicted = np.asarray([0.2, 0.4, 0.6, 0.5])
    groups = np.asarray(["a", "a", "b", "b"])

    first = clustered_error_interval(
        target,
        predicted,
        groups,
        repeats=200,
        confidence_level=0.95,
        random_seed=42,
    )
    second = clustered_error_interval(
        target,
        predicted,
        groups,
        repeats=200,
        confidence_level=0.95,
        random_seed=42,
    )

    assert first == second
    assert first["mae_ci_lower"] <= np.mean(np.abs(predicted - target))
    assert first["mae_ci_upper"] >= np.mean(np.abs(predicted - target))


def test_clustered_error_interval_gives_sampled_groups_equal_weight() -> None:
    target = np.zeros(3)
    predicted = np.asarray([0.0, 0.0, 1.0])
    groups = np.asarray(["paired", "paired", "singleton"])
    repeats = 20
    confidence_level = 0.8
    random_seed = 9

    interval = clustered_error_interval(
        target,
        predicted,
        groups,
        repeats=repeats,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed)
    group_losses = np.asarray([0.0, 1.0])
    bootstrap = np.asarray(
        [np.mean(group_losses[rng.integers(0, 2, size=2)]) for _ in range(repeats)]
    )

    assert interval["mae_ci_lower"] == pytest.approx(np.quantile(bootstrap, 0.1))
    assert interval["mae_ci_upper"] == pytest.approx(np.quantile(bootstrap, 0.9))


def test_regime_masks_cover_each_partition_without_overlap() -> None:
    features = np.asarray(
        [
            [0, 0, 2, 0, -2, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 1, 0],
            [1, 0, 0, 5, 5, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    masks = regime_masks(
        features,
        FEATURE_NAMES,
        np.asarray([0.1, 0.5, 0.7, 0.9]),
        np.asarray([10, 20, 30, 40]),
        extreme_ev_threshold=4.0,
        near_ev_threshold=1.0,
        half_half_width=0.05,
        consensus_width=0.3,
        participant_median=25.0,
    )

    for levels in masks.values():
        membership = np.sum(np.stack(list(levels.values())), axis=0)
        assert np.array_equal(membership, np.ones(4))


def test_participant_count_analysis_finds_constructed_relationship() -> None:
    participant_counts = np.asarray([10, 10, 20, 20, 30, 30])
    target = np.full(6, 0.5)
    predicted = np.asarray([0.55, 0.56, 0.60, 0.62, 0.70, 0.74])
    groups = np.asarray(["a", "b", "c", "d", "e", "f"])

    rows, relationships = participant_count_analysis(
        participant_counts,
        target,
        predicted,
        groups,
        repeats=100,
        confidence_level=0.95,
        random_seed=7,
    )

    assert [row["participant_count_n"] for row in rows] == [10, 20, 30]
    assert rows[0]["problem_group_mae"] < rows[-1]["problem_group_mae"]
    assert relationships["pearson_n_vs_absolute_error"] > 0.9
    assert relationships["spearman_n_vs_absolute_error"] > 0.9


def test_case_reconstruction_and_representative_selection() -> None:
    record = SelectionRecord(
        problem=17,
        feedback=True,
        n=15,
        block=1,
        ha=10,
        pha=0.5,
        la=0,
        hb=20,
        phb=0.25,
        lb=0,
        lot_shape_b=0,
        lot_num_b=2,
        amb=True,
        corr=0,
        brate=0.2,
        brate_std=0.1,
    )
    features = np.asarray([[1, 1, 5, 10, 5, 0, 1, 0, 0]], dtype=float)
    cases = build_case_rows(
        np.asarray([0]),
        [record],
        {"0": {"A": [[0.5, 10], [0.5, 0]], "B": [[0.5, 20], [0.5, 0]]}},
        features,
        FEATURE_NAMES,
        np.asarray([0.8]),
        extreme_ev_threshold=4.0,
        near_ev_threshold=1.0,
        participant_median=20.0,
    )
    selected = select_representative_failures(
        cases,
        per_category=1,
        extreme_ev_threshold=4.0,
        half_half_width=0.05,
    )

    assert cases[0]["gamble_a"] == "p=0.5→10; p=0.5→0"
    assert cases[0]["expected_value_benchmark"] == "B"
    assert cases[0]["absolute_error"] == pytest.approx(0.6)
    assert "not see" in cases[0]["possible_modeling_considerations"]
    assert {row["case_category"] for row in selected} == {
        "largest_overall",
        "ambiguity",
        "feedback",
        "extreme_expected_value",
    }
