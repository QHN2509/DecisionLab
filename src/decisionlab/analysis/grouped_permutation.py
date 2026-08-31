"""Grouped, dependency-preserving permutation importance for behavioral interpretation."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from decisionlab.data.validation import SelectionRecord
from decisionlab.features.behavioral import (
    ProblemFeatureInput,
    RawFeatureRow,
    engineer_behavioral_features,
    extract_problem_features,
    extract_raw_features,
    validate_feature_rows,
)

PerturbationMode = Literal[
    "gamble_a",
    "gamble_b",
    "ambiguity",
    "feedback",
    "correlation",
    "gamble_structure",
    "information_conditions",
    "complete_problem_block",
]


@dataclass(frozen=True, slots=True)
class ProductionFeatureInput:
    """One predictor-only row accepted by production feature engineering."""

    predictors: RawFeatureRow
    problem: ProblemFeatureInput


@dataclass(frozen=True, slots=True)
class PerturbationFamily:
    """A coherent primitive-input family and its interpretation metadata."""

    name: str
    mode: PerturbationMode
    primitive_fields: tuple[str, ...]
    description: str
    requires_compatible_group_size: bool = False


FEATURE_PERTURBATION_FAMILIES = (
    PerturbationFamily(
        name="gamble_a_distribution",
        mode="gamble_a",
        primitive_fields=("A", "Ha", "pHa", "La"),
        description="Complete Gamble A distribution with every dependent feature recomputed.",
    ),
    PerturbationFamily(
        name="gamble_b_distribution_and_lottery",
        mode="gamble_b",
        primitive_fields=("B", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB"),
        description=(
            "Complete Gamble B distribution and lottery metadata with every dependent feature "
            "recomputed."
        ),
    ),
    PerturbationFamily(
        name="ambiguity_condition",
        mode="ambiguity",
        primitive_fields=("Amb",),
        description="Group-level ambiguity condition with its EV interaction recomputed.",
    ),
    PerturbationFamily(
        name="feedback_condition_block",
        mode="feedback",
        primitive_fields=("Feedback",),
        description=(
            "Feedback condition transferred as a complete same-size row block; paired rows are "
            "never separated."
        ),
        requires_compatible_group_size=True,
    ),
    PerturbationFamily(
        name="correlation_condition",
        mode="correlation",
        primitive_fields=("Corr",),
        description="Group-level correlation category with its complete one-hot family rebuilt.",
    ),
)

DOMAIN_PERTURBATION_FAMILIES = (
    PerturbationFamily(
        name="complete_problem_block",
        mode="complete_problem_block",
        primitive_fields=(
            "A",
            "B",
            "Ha",
            "pHa",
            "La",
            "Hb",
            "pHb",
            "Lb",
            "LotShapeB",
            "LotNumB",
            "Amb",
            "Feedback",
            "Corr",
        ),
        description=(
            "Complete observed predictor block transferred between same-size structural groups."
        ),
        requires_compatible_group_size=True,
    ),
    PerturbationFamily(
        name="gamble_structure",
        mode="gamble_structure",
        primitive_fields=(
            "A",
            "B",
            "Ha",
            "pHa",
            "La",
            "Hb",
            "pHb",
            "Lb",
            "LotShapeB",
            "LotNumB",
            "Corr",
        ),
        description=(
            "Both gamble distributions, lottery metadata, and correlation transferred jointly; "
            "recipient ambiguity and feedback conditions are retained."
        ),
    ),
    PerturbationFamily(
        name="information_and_experience_conditions",
        mode="information_conditions",
        primitive_fields=("Amb", "Feedback"),
        description=(
            "Ambiguity and feedback transferred as a same-size condition block with dependent "
            "interactions rebuilt."
        ),
        requires_compatible_group_size=True,
    ),
)


def production_feature_inputs(
    selections: Sequence[SelectionRecord],
    problems: Mapping[str, dict[str, list[list[float]]]],
) -> list[ProductionFeatureInput]:
    """Create predictor-only inputs using the production extraction contracts."""
    if set(problems) != {str(index) for index in range(len(selections))}:
        raise ValueError("Problem descriptions must align with selection row indices")
    return [
        ProductionFeatureInput(
            predictors=extract_raw_features(record),
            problem=extract_problem_features(problems[str(index)]),
        )
        for index, record in enumerate(selections)
    ]


def engineer_input_matrix(
    inputs: Sequence[ProductionFeatureInput], feature_names: Sequence[str]
) -> np.ndarray:
    """Recompute and validate a feature matrix through production feature engineering."""
    engineered = [engineer_behavioral_features(row.predictors, row.problem) for row in inputs]
    validate_feature_rows([row.predictors for row in inputs], engineered)
    expected = set(feature_names)
    if not engineered or set(asdict(engineered[0])) != expected:
        raise ValueError("Production engineered feature names differ from the model contract")
    return np.asarray(
        [[float(asdict(row)[name]) for name in feature_names] for row in engineered],
        dtype=float,
    )


def _group_rows(groups: np.ndarray) -> dict[str, np.ndarray]:
    rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups.astype(str)):
        rows[group].append(index)
    return {group: np.asarray(indices, dtype=int) for group, indices in rows.items()}


def grouped_donor_mapping(
    groups: np.ndarray,
    outer_folds: np.ndarray,
    *,
    random_seed: int,
    compatible_group_size: bool,
) -> dict[str, str]:
    """Derange whole structural groups within outer folds and compatible size strata."""
    groups = np.asarray(groups).astype(str)
    outer_folds = np.asarray(outer_folds)
    if groups.ndim != 1 or groups.size == 0 or outer_folds.shape != groups.shape:
        raise ValueError("Group donor inputs must be aligned nonempty vectors")
    group_rows = _group_rows(groups)
    group_fold: dict[str, int] = {}
    strata: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group, indices in group_rows.items():
        folds = np.unique(outer_folds[indices])
        if folds.size != 1:
            raise ValueError("A structural group crosses outer folds")
        fold = int(folds[0])
        group_fold[group] = fold
        size_key = int(indices.size) if compatible_group_size else 0
        strata[(fold, size_key)].append(group)

    rng = np.random.default_rng(random_seed)
    mapping: dict[str, str] = {}
    for stratum in sorted(strata):
        recipients = np.asarray(sorted(strata[stratum]), dtype=object)
        shuffled = recipients[rng.permutation(recipients.size)]
        if shuffled.size == 1:
            donors = shuffled
        else:
            shift = int(rng.integers(1, shuffled.size))
            donors = np.roll(shuffled, shift)
        mapping.update(
            {str(recipient): str(donor) for recipient, donor in zip(shuffled, donors, strict=True)}
        )
    if set(mapping) != set(group_rows):
        raise ValueError("Donor mapping does not cover every structural group")
    if any(group_fold[group] != group_fold[donor] for group, donor in mapping.items()):
        raise ValueError("A donor mapping crosses outer folds")
    return mapping


def _replace_gamble_a(
    recipient: ProductionFeatureInput, donor: ProductionFeatureInput
) -> ProductionFeatureInput:
    predictors = replace(
        recipient.predictors,
        ha=donor.predictors.ha,
        pha=donor.predictors.pha,
        la=donor.predictors.la,
    )
    problem = replace(recipient.problem, gamble_a=donor.problem.gamble_a)
    return ProductionFeatureInput(predictors, problem)


def _replace_gamble_b(
    recipient: ProductionFeatureInput, donor: ProductionFeatureInput
) -> ProductionFeatureInput:
    predictors = replace(
        recipient.predictors,
        hb=donor.predictors.hb,
        phb=donor.predictors.phb,
        lb=donor.predictors.lb,
        lot_shape_b=donor.predictors.lot_shape_b,
        lot_num_b=donor.predictors.lot_num_b,
    )
    problem = replace(recipient.problem, gamble_b=donor.problem.gamble_b)
    return ProductionFeatureInput(predictors, problem)


def _compose_perturbation(
    recipient: ProductionFeatureInput,
    donor: ProductionFeatureInput,
    mode: PerturbationMode,
) -> ProductionFeatureInput:
    if mode == "gamble_a":
        return _replace_gamble_a(recipient, donor)
    if mode == "gamble_b":
        return _replace_gamble_b(recipient, donor)
    if mode == "ambiguity":
        return replace(
            recipient,
            predictors=replace(recipient.predictors, amb=donor.predictors.amb),
        )
    if mode == "feedback":
        return replace(
            recipient,
            predictors=replace(recipient.predictors, feedback=donor.predictors.feedback),
        )
    if mode == "correlation":
        return replace(
            recipient,
            predictors=replace(recipient.predictors, corr=donor.predictors.corr),
        )
    if mode == "gamble_structure":
        changed = _replace_gamble_b(_replace_gamble_a(recipient, donor), donor)
        return replace(changed, predictors=replace(changed.predictors, corr=donor.predictors.corr))
    if mode == "information_conditions":
        return replace(
            recipient,
            predictors=replace(
                recipient.predictors,
                amb=donor.predictors.amb,
                feedback=donor.predictors.feedback,
            ),
        )
    if mode == "complete_problem_block":
        return donor
    raise ValueError(f"Unsupported perturbation mode: {mode}")


def perturb_feature_inputs(
    inputs: Sequence[ProductionFeatureInput],
    groups: np.ndarray,
    donor_mapping: Mapping[str, str],
    family: PerturbationFamily,
) -> list[ProductionFeatureInput]:
    """Transfer a coherent input family without splitting structural row blocks."""
    groups = np.asarray(groups).astype(str)
    if groups.size != len(inputs):
        raise ValueError("Perturbation inputs and groups must be aligned")
    group_rows = _group_rows(groups)
    if set(donor_mapping) != set(group_rows):
        raise ValueError("Donor mapping and structural groups differ")
    result: list[ProductionFeatureInput | None] = [None] * len(inputs)
    for recipient_group, recipient_indices in group_rows.items():
        donor_group = donor_mapping[recipient_group]
        donor_indices = group_rows[donor_group]
        if family.requires_compatible_group_size and donor_indices.size != recipient_indices.size:
            raise ValueError("A block perturbation mapped groups with different row counts")
        for ordinal, recipient_index in enumerate(recipient_indices):
            donor_index = (
                donor_indices[ordinal]
                if family.requires_compatible_group_size
                else donor_indices[0]
            )
            result[int(recipient_index)] = _compose_perturbation(
                inputs[int(recipient_index)], inputs[int(donor_index)], family.mode
            )
    if any(row is None for row in result):
        raise ValueError("Perturbation did not cover every row")
    typed = [row for row in result if row is not None]
    audit_structural_group_blocks(typed, groups)
    return typed


def _structural_signature(row: ProductionFeatureInput) -> str:
    raw = asdict(row.predictors)
    raw.pop("feedback")
    return json.dumps(
        {"predictors": raw, "problem": asdict(row.problem)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def audit_structural_group_blocks(
    inputs: Sequence[ProductionFeatureInput], groups: np.ndarray
) -> dict[str, Any]:
    """Prove that every recipient group retains one coherent structural problem."""
    groups = np.asarray(groups).astype(str)
    if groups.size != len(inputs):
        raise ValueError("Structural audit inputs must be aligned")
    group_rows = _group_rows(groups)
    for group, indices in group_rows.items():
        signatures = {_structural_signature(inputs[int(index)]) for index in indices}
        if len(signatures) != 1:
            raise ValueError(f"Perturbation split structural problem group {group}")
    return {
        "status": "PASS",
        "rows": len(inputs),
        "structural_groups": len(group_rows),
        "split_groups": 0,
    }


def draw_grouped_bootstrap(
    group_names: np.ndarray,
    group_folds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw whole structural groups with replacement, stratified by outer fold."""
    group_names = np.asarray(group_names).astype(str)
    group_folds = np.asarray(group_folds)
    if group_names.ndim != 1 or group_names.size == 0 or group_folds.shape != group_names.shape:
        raise ValueError("Grouped bootstrap inputs must be aligned nonempty vectors")
    sampled = []
    for fold in np.unique(group_folds):
        eligible = group_names[group_folds == fold]
        sampled.extend(rng.choice(eligible, size=eligible.size, replace=True).tolist())
    return np.asarray(sampled, dtype=str)


def _group_loss_differences(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    perturbed_prediction: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    group_rows = _group_rows(np.asarray(groups).astype(str))
    names = np.asarray(sorted(group_rows), dtype=str)
    differences = np.asarray(
        [
            float(
                np.mean(np.abs(target[indices] - perturbed_prediction[indices]))
                - np.mean(np.abs(target[indices] - baseline_prediction[indices]))
            )
            for indices in (group_rows[name] for name in names)
        ]
    )
    return names, differences


def grouped_permutation_importance_rows(
    predict: Callable[[np.ndarray], np.ndarray],
    inputs: Sequence[ProductionFeatureInput],
    baseline_features: np.ndarray,
    target: np.ndarray,
    structural_groups: np.ndarray,
    outer_folds: np.ndarray,
    feature_names: Sequence[str],
    families: Sequence[PerturbationFamily],
    *,
    repeats: int,
    group_bootstrap_repeats: int,
    confidence_level: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Estimate coherent family reliance with grouped permutations and uncertainty."""
    target = np.asarray(target, dtype=float)
    groups = np.asarray(structural_groups).astype(str)
    folds = np.asarray(outer_folds)
    if (
        baseline_features.shape[0] != target.size
        or groups.shape != target.shape
        or folds.shape != target.shape
        or len(inputs) != target.size
    ):
        raise ValueError("Grouped permutation inputs must be aligned")
    if repeats < 2 or group_bootstrap_repeats < 2 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Permutation and bootstrap settings are invalid")
    baseline_prediction = np.asarray(predict(baseline_features), dtype=float)
    baseline_names, baseline_zero = _group_loss_differences(
        target, baseline_prediction, baseline_prediction, groups
    )
    if not np.allclose(baseline_zero, 0.0):
        raise ValueError("Baseline grouped loss differences must be zero")
    group_rows = _group_rows(groups)
    group_folds = np.asarray(
        [int(np.unique(folds[group_rows[name]])[0]) for name in baseline_names], dtype=int
    )
    baseline_mae = float(
        np.mean(
            [
                np.mean(np.abs(target[indices] - baseline_prediction[indices]))
                for indices in (group_rows[name] for name in baseline_names)
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        repeat_differences = []
        effective_counts = []
        for repeat_index in range(repeats):
            seed = random_seed + family_index * 100_000 + repeat_index
            mapping = grouped_donor_mapping(
                groups,
                folds,
                random_seed=seed,
                compatible_group_size=family.requires_compatible_group_size,
            )
            perturbed_inputs = perturb_feature_inputs(inputs, groups, mapping, family)
            perturbed_features = engineer_input_matrix(perturbed_inputs, feature_names)
            perturbed_prediction = np.asarray(predict(perturbed_features), dtype=float)
            names, differences = _group_loss_differences(
                target, baseline_prediction, perturbed_prediction, groups
            )
            if not np.array_equal(names, baseline_names):
                raise ValueError("Structural group order changed during permutation")
            repeat_differences.append(differences)
            changed_rows = np.any(~np.isclose(perturbed_features, baseline_features), axis=1)
            effective_counts.append(
                sum(bool(np.any(changed_rows[group_rows[name]])) for name in baseline_names)
            )
        difference_matrix = np.asarray(repeat_differences)
        group_mean_differences = np.mean(difference_matrix, axis=0)
        repeat_estimates = np.mean(difference_matrix, axis=1)
        bootstrap_rng = np.random.default_rng(random_seed + 10_000_000 + family_index)
        bootstrap_estimates = []
        difference_by_group = dict(zip(baseline_names, group_mean_differences, strict=True))
        for _ in range(group_bootstrap_repeats):
            sampled = draw_grouped_bootstrap(baseline_names, group_folds, bootstrap_rng)
            bootstrap_estimates.append(
                float(np.mean([difference_by_group[group] for group in sampled]))
            )
        alpha = (1.0 - confidence_level) / 2.0
        rows.append(
            {
                "name": family.name,
                "importance_unit": "coherent_primitive_dependency_family",
                "primitive_fields": ";".join(family.primitive_fields),
                "description": family.description,
                "baseline_mae": baseline_mae,
                "mean_mae_increase": float(np.mean(repeat_estimates)),
                "permutation_sd": float(np.std(repeat_estimates, ddof=1)),
                "min_mae_increase": float(np.min(repeat_estimates)),
                "max_mae_increase": float(np.max(repeat_estimates)),
                "group_bootstrap_ci_lower": float(np.quantile(bootstrap_estimates, alpha)),
                "group_bootstrap_ci_upper": float(np.quantile(bootstrap_estimates, 1.0 - alpha)),
                "confidence_level": confidence_level,
                "repeats": repeats,
                "group_bootstrap_repeats": group_bootstrap_repeats,
                "structural_groups": int(baseline_names.size),
                "mean_effectively_perturbed_groups": float(np.mean(effective_counts)),
                "donor_scope": "same_outer_fold_structural_group",
                "paired_rows_split": 0,
            }
        )
    return sorted(rows, key=lambda row: row["mean_mae_increase"], reverse=True)
