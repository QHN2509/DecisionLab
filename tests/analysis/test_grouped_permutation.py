"""Contracts for grouped, dependency-preserving behavioral permutation importance."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from decisionlab.analysis.grouped_permutation import (
    DOMAIN_PERTURBATION_FAMILIES,
    FEATURE_PERTURBATION_FAMILIES,
    ProductionFeatureInput,
    audit_structural_group_blocks,
    draw_grouped_bootstrap,
    engineer_input_matrix,
    grouped_donor_mapping,
    grouped_permutation_importance_rows,
    perturb_feature_inputs,
)
from decisionlab.features.behavioral import (
    EngineeredFeatureRow,
    ScenarioFeatureInput,
    build_scenario_problem,
    engineer_scenario_features,
    extract_problem_features,
)


def _scenario(group_index: int, *, feedback: bool) -> ScenarioFeatureInput:
    return ScenarioFeatureInput(
        high_payoff_a=12.0 + group_index,
        high_probability_a=0.6,
        low_payoff_a=-4.0 + group_index,
        sublottery_mean_b=5.0 + 2 * group_index,
        sublottery_probability_b=0.35,
        low_payoff_b=-8.0 + group_index,
        lottery_shape_b=1,
        lottery_outcomes_b=3,
        ambiguity=bool(group_index % 2),
        feedback=feedback,
        correlation=(-1, 0, 1, 0)[group_index],
    )


def _inputs() -> tuple[
    list[ProductionFeatureInput],
    list[ScenarioFeatureInput],
    np.ndarray,
    np.ndarray,
]:
    scenarios = [
        _scenario(0, feedback=False),
        _scenario(0, feedback=True),
        _scenario(1, feedback=False),
        _scenario(1, feedback=True),
        _scenario(2, feedback=False),
        _scenario(3, feedback=True),
    ]
    inputs = []
    for scenario in scenarios:
        predictors, problem = build_scenario_problem(scenario)
        inputs.append(ProductionFeatureInput(predictors, extract_problem_features(problem)))
    groups = np.asarray(["g0", "g0", "g1", "g1", "g2", "g3"])
    folds = np.asarray([0, 0, 0, 0, 1, 1])
    return inputs, scenarios, groups, folds


def _feature_names() -> list[str]:
    return list(EngineeredFeatureRow.__dataclass_fields__)


def test_grouped_donors_remain_within_outer_fold_and_move_whole_groups() -> None:
    _, _, groups, folds = _inputs()

    mapping = grouped_donor_mapping(
        groups,
        folds,
        random_seed=41,
        compatible_group_size=True,
    )

    group_fold = {group: int(np.unique(folds[groups == group])[0]) for group in np.unique(groups)}
    group_size = {group: int(np.sum(groups == group)) for group in np.unique(groups)}
    assert set(mapping) == set(np.unique(groups))
    assert all(group_fold[group] == group_fold[donor] for group, donor in mapping.items())
    assert all(group_size[group] == group_size[donor] for group, donor in mapping.items())
    assert all(mapping[group] != group for group in mapping)


def test_complete_block_permutation_preserves_paired_problem_structure() -> None:
    inputs, _, groups, folds = _inputs()
    family = next(
        row for row in DOMAIN_PERTURBATION_FAMILIES if row.name == "complete_problem_block"
    )
    mapping = grouped_donor_mapping(
        groups,
        folds,
        random_seed=9,
        compatible_group_size=True,
    )

    perturbed = perturb_feature_inputs(inputs, groups, mapping, family)
    audit = audit_structural_group_blocks(perturbed, groups)

    assert audit["split_groups"] == 0
    for group in ("g0", "g1"):
        feedback_pattern = [
            perturbed[index].predictors.feedback for index in np.flatnonzero(groups == group)
        ]
        assert feedback_pattern == [False, True]


@pytest.mark.parametrize("family", FEATURE_PERTURBATION_FAMILIES)
def test_perturbation_recomputes_mathematically_consistent_features(family) -> None:
    inputs, _, groups, folds = _inputs()
    mapping = grouped_donor_mapping(
        groups,
        folds,
        random_seed=100,
        compatible_group_size=family.requires_compatible_group_size,
    )
    perturbed = perturb_feature_inputs(inputs, groups, mapping, family)

    matrix = engineer_input_matrix(perturbed, _feature_names())
    position = {name: index for index, name in enumerate(_feature_names())}

    assert matrix[:, position["expected_value_difference_b_minus_a_oracle"]] == pytest.approx(
        matrix[:, position["expected_value_b_oracle"]] - matrix[:, position["expected_value_a"]]
    )
    assert matrix[:, position["payoff_range_difference_b_minus_a"]] == pytest.approx(
        matrix[:, position["payoff_range_b"]] - matrix[:, position["payoff_range_a"]]
    )
    assert matrix[:, position["expected_value_difference_oracle_x_feedback"]] == pytest.approx(
        matrix[:, position["expected_value_difference_b_minus_a_oracle"]]
        * matrix[:, position["feedback_indicator"]]
    )
    assert matrix[:, position["expected_value_difference_oracle_x_ambiguity"]] == pytest.approx(
        matrix[:, position["expected_value_difference_b_minus_a_oracle"]]
        * matrix[:, position["ambiguity_indicator"]]
    )
    shape_columns = [position[name] for name in position if name.startswith("lottery_shape")]
    correlation_columns = [position[name] for name in position if name.startswith("correlation")]
    assert np.sum(matrix[:, shape_columns], axis=1) == pytest.approx(1.0)
    assert np.sum(matrix[:, correlation_columns], axis=1) == pytest.approx(1.0)


def test_matrix_recomputation_matches_scenario_production_pipeline() -> None:
    inputs, scenarios, _, _ = _inputs()

    matrix = engineer_input_matrix(inputs, _feature_names())
    expected = np.asarray(
        [
            [
                float(asdict(engineer_scenario_features(scenario)[0])[name])
                for name in _feature_names()
            ]
            for scenario in scenarios
        ]
    )

    assert np.array_equal(matrix, expected)


def test_grouped_permutation_and_uncertainty_are_seed_reproducible() -> None:
    inputs, _, groups, folds = _inputs()
    names = _feature_names()
    features = engineer_input_matrix(inputs, names)
    position = {name: index for index, name in enumerate(names)}
    target = np.clip(
        0.5 + 0.03 * features[:, position["expected_value_difference_b_minus_a_oracle"]],
        0.0,
        1.0,
    )

    def predict(values: np.ndarray) -> np.ndarray:
        return np.clip(
            0.5 + 0.03 * values[:, position["expected_value_difference_b_minus_a_oracle"]],
            0.0,
            1.0,
        )

    kwargs = {
        "repeats": 3,
        "group_bootstrap_repeats": 40,
        "confidence_level": 0.95,
        "random_seed": 77,
    }
    first = grouped_permutation_importance_rows(
        predict,
        inputs,
        features,
        target,
        groups,
        folds,
        names,
        FEATURE_PERTURBATION_FAMILIES[:2],
        **kwargs,
    )
    second = grouped_permutation_importance_rows(
        predict,
        inputs,
        features,
        target,
        groups,
        folds,
        names,
        FEATURE_PERTURBATION_FAMILIES[:2],
        **kwargs,
    )

    assert first == second
    assert first[0]["mean_mae_increase"] > 0.0
    assert all(row["paired_rows_split"] == 0 for row in first)


def test_grouped_bootstrap_samples_structural_groups_not_rows() -> None:
    _, _, groups, folds = _inputs()
    group_names = np.asarray(["g0", "g1", "g2", "g3"])
    group_folds = np.asarray([0, 0, 1, 1])

    sampled = draw_grouped_bootstrap(group_names, group_folds, np.random.default_rng(12))
    expanded_rows = np.concatenate([np.flatnonzero(groups == group) for group in sampled])

    assert sampled.size == group_names.size
    assert sum(group in {"g0", "g1"} for group in sampled[:2]) == 2
    assert sum(group in {"g2", "g3"} for group in sampled[2:]) == 2
    for group in sampled:
        assert set(np.flatnonzero(groups == group)) <= set(expanded_rows)


def test_feedback_family_changes_only_feedback_and_recomputed_dependents() -> None:
    inputs, _, groups, folds = _inputs()
    family = next(
        row for row in FEATURE_PERTURBATION_FAMILIES if row.name == "feedback_condition_block"
    )
    mapping = grouped_donor_mapping(
        groups,
        folds,
        random_seed=5,
        compatible_group_size=True,
    )

    perturbed = perturb_feature_inputs(inputs, groups, mapping, family)

    for original, changed in zip(inputs, perturbed, strict=True):
        assert (
            replace(original.predictors, feedback=changed.predictors.feedback) == changed.predictors
        )
        assert original.problem == changed.problem
