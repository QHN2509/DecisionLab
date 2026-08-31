"""Deterministic structural-group splitting for choices13k."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import GroupKFold

from decisionlab.data.validation import SelectionRecord

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One row's immutable structural group and split assignment."""

    row_index: int
    problem: int
    structural_fingerprint: str
    split: SplitName


@dataclass(frozen=True, slots=True)
class OuterFoldAssignment:
    """One row's structural group and outer test-fold assignment."""

    row_index: int
    problem: int
    structural_fingerprint: str
    outer_fold: int


@dataclass(frozen=True, slots=True)
class InnerFoldAssignment:
    """One outer-training row's inner validation-fold assignment."""

    outer_fold: int
    row_index: int
    structural_fingerprint: str
    inner_fold: int


def grouped_fold_indices(
    groups: np.ndarray, *, folds: int, random_seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic grouped folds with complete held-out-row coverage."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("Groups must be a nonempty vector")
    if np.unique(groups).size < folds:
        raise ValueError("Number of structural groups must be at least the fold count")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    splits = list(splitter.split(np.zeros((groups.size, 1)), groups=groups))
    held_out: list[int] = []
    for train_indices, test_indices in splits:
        if set(groups[train_indices]) & set(groups[test_indices]):
            raise ValueError("Structural group overlap detected within grouped CV")
        held_out.extend(test_indices.tolist())
    if sorted(held_out) != list(range(groups.size)):
        raise ValueError("Grouped folds must hold out every row exactly once")
    return splits


def create_nested_fold_assignments(
    problems: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: int,
    inner_folds: int,
    outer_seed: int,
    inner_seed: int,
) -> tuple[list[OuterFoldAssignment], list[InnerFoldAssignment]]:
    """Create persisted outer and outer-specific inner grouped assignments."""
    problems = np.asarray(problems)
    groups = np.asarray(groups)
    if problems.ndim != 1 or problems.size != groups.size:
        raise ValueError("Problems and groups must be aligned vectors")
    outer_splits = grouped_fold_indices(groups, folds=outer_folds, random_seed=outer_seed)
    outer_by_row = np.empty(groups.size, dtype=int)
    inner_assignments: list[InnerFoldAssignment] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
        outer_by_row[outer_test] = outer_fold
        inner_splits = grouped_fold_indices(
            groups[outer_train], folds=inner_folds, random_seed=inner_seed + outer_fold
        )
        for inner_fold, (_, inner_validation_local) in enumerate(inner_splits):
            for row_index in outer_train[inner_validation_local]:
                inner_assignments.append(
                    InnerFoldAssignment(
                        outer_fold=outer_fold,
                        row_index=int(row_index),
                        structural_fingerprint=str(groups[row_index]),
                        inner_fold=inner_fold,
                    )
                )

    outer_assignments = [
        OuterFoldAssignment(
            row_index=row_index,
            problem=int(problems[row_index]),
            structural_fingerprint=str(groups[row_index]),
            outer_fold=int(outer_by_row[row_index]),
        )
        for row_index in range(groups.size)
    ]
    audit_nested_fold_assignments(
        outer_assignments,
        inner_assignments,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
    )
    return outer_assignments, inner_assignments


def audit_nested_fold_assignments(
    outer: list[OuterFoldAssignment],
    inner: list[InnerFoldAssignment],
    *,
    outer_folds: int,
    inner_folds: int,
) -> dict[str, Any]:
    """Prove outer coverage and isolation of every outer-specific inner CV."""
    if not outer or {row.row_index for row in outer} != set(range(len(outer))):
        raise ValueError("Outer assignments must cover every row exactly once")
    if len({row.row_index for row in outer}) != len(outer):
        raise ValueError("Duplicate outer row assignment")
    outer_group_folds: dict[str, set[int]] = defaultdict(set)
    for row in outer:
        outer_group_folds[row.structural_fingerprint].add(row.outer_fold)
    if any(len(values) != 1 for values in outer_group_folds.values()):
        raise ValueError("A structural group crosses outer test folds")
    if {row.outer_fold for row in outer} != set(range(outer_folds)):
        raise ValueError("Outer assignments do not contain every configured fold")

    outer_by_index = {row.row_index: row for row in outer}
    for outer_fold in range(outer_folds):
        expected = {row.row_index for row in outer if row.outer_fold != outer_fold}
        rows = [row for row in inner if row.outer_fold == outer_fold]
        actual = [row.row_index for row in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Inner validation assignments must cover outer training rows once")
        if {row.inner_fold for row in rows} != set(range(inner_folds)):
            raise ValueError("Inner assignments do not contain every configured fold")
        inner_group_folds: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            parent = outer_by_index[row.row_index]
            if parent.outer_fold == outer_fold:
                raise ValueError("Outer test row leaked into inner CV")
            if parent.structural_fingerprint != row.structural_fingerprint:
                raise ValueError("Inner structural fingerprint differs from outer assignment")
            inner_group_folds[row.structural_fingerprint].add(row.inner_fold)
        if any(len(values) != 1 for values in inner_group_folds.values()):
            raise ValueError("A structural group crosses inner validation folds")
    return {
        "status": "PASS",
        "rows": len(outer),
        "structural_groups": len(outer_group_folds),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "outer_group_overlap_count": 0,
        "inner_group_overlap_count": 0,
        "outer_test_rows_in_inner_cv": 0,
    }


def structural_fingerprint(record: SelectionRecord, problem_description: dict[str, Any]) -> str:
    """Hash every target-free field that defines the underlying decision problem.

    Feedback, Block, n, bRate, and bRate_std are intentionally excluded so condition
    variants of one problem receive the same fingerprint.
    """
    payload = {
        "A": problem_description["A"],
        "B": problem_description["B"],
        "Ha": record.ha,
        "pHa": record.pha,
        "La": record.la,
        "Hb": record.hb,
        "pHb": record.phb,
        "Lb": record.lb,
        "LotShapeB": record.lot_shape_b,
        "LotNumB": record.lot_num_b,
        "Amb": record.amb,
        "Corr": record.corr,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_fraction(seed: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_split(
    fingerprint: str,
    seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> SplitName:
    """Map a structural group to a stable hash bucket without using target values."""
    value = _hash_fraction(seed, fingerprint)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def create_grouped_assignments(
    selections: list[SelectionRecord],
    problems: dict[str, Any],
    *,
    seed: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> list[SplitAssignment]:
    """Assign rows by exact problem structure and prove Problem/fingerprint consistency."""
    if not selections:
        raise ValueError("Cannot split an empty dataset")
    if any(fraction <= 0.0 for fraction in (train_fraction, validation_fraction, test_fraction)):
        raise ValueError("Every split fraction must be positive")
    if not abs(train_fraction + validation_fraction + test_fraction - 1.0) < 1e-12:
        raise ValueError("Split fractions must sum to one")
    if set(problems) != {str(index) for index in range(len(selections))}:
        raise ValueError("Problem descriptions must be keyed by zero-based row index")

    fingerprints = [
        structural_fingerprint(record, problems[str(index)])
        for index, record in enumerate(selections)
    ]
    fingerprints_by_problem: dict[int, set[str]] = defaultdict(set)
    for record, fingerprint in zip(selections, fingerprints, strict=True):
        fingerprints_by_problem[record.problem].add(fingerprint)
    inconsistent = [
        problem for problem, values in fingerprints_by_problem.items() if len(values) != 1
    ]
    if inconsistent:
        raise ValueError(f"Problem IDs map to multiple structural fingerprints: {inconsistent[:5]}")

    return [
        SplitAssignment(
            row_index=index,
            problem=record.problem,
            structural_fingerprint=fingerprint,
            split=choose_split(
                fingerprint,
                seed,
                train_fraction,
                validation_fraction,
            ),
        )
        for index, (record, fingerprint) in enumerate(zip(selections, fingerprints, strict=True))
    ]


def audit_grouped_assignments(assignments: list[SplitAssignment]) -> dict[str, Any]:
    """Validate row coverage and zero group overlap across all partitions."""
    if not assignments:
        raise ValueError("No split assignments to audit")
    row_indexes = [assignment.row_index for assignment in assignments]
    if len(set(row_indexes)) != len(assignments) or set(row_indexes) != set(
        range(len(assignments))
    ):
        raise ValueError("Assignments must cover each zero-based row index exactly once")

    group_splits: dict[str, set[SplitName]] = defaultdict(set)
    problem_splits: dict[int, set[SplitName]] = defaultdict(set)
    for assignment in assignments:
        group_splits[assignment.structural_fingerprint].add(assignment.split)
        problem_splits[assignment.problem].add(assignment.split)
    overlapping_groups = sum(len(splits) > 1 for splits in group_splits.values())
    overlapping_problems = sum(len(splits) > 1 for splits in problem_splits.values())
    if overlapping_groups or overlapping_problems:
        raise ValueError(
            f"Split leakage detected: groups={overlapping_groups}, problems={overlapping_problems}"
        )

    row_counts = Counter(assignment.split for assignment in assignments)
    group_counts = Counter(next(iter(splits)) for splits in group_splits.values())
    if set(row_counts) != {"train", "validation", "test"}:
        raise ValueError("All three partitions must contain rows")
    return {
        "status": "PASS",
        "rows": dict(sorted(row_counts.items())),
        "structural_groups": dict(sorted(group_counts.items())),
        "unique_structural_groups": len(group_splits),
        "unique_problem_ids": len(problem_splits),
        "structural_group_overlap_count": overlapping_groups,
        "problem_id_overlap_count": overlapping_problems,
    }


def ordinary_row_split_leakage_demo(
    selections: list[SelectionRecord],
    problems: dict[str, Any],
    *,
    seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    """Demonstrate overlap caused by hashing rows rather than structural groups."""
    fingerprints = [
        structural_fingerprint(record, problems[str(row_index)])
        for row_index, record in enumerate(selections)
    ]
    fingerprint_splits: dict[str, set[SplitName]] = defaultdict(set)
    for row_index, fingerprint in enumerate(fingerprints):
        split = choose_split(
            f"row-{row_index}",
            seed,
            train_fraction,
            validation_fraction,
        )
        fingerprint_splits[fingerprint].add(split)
    overlapping = sum(len(splits) > 1 for splits in fingerprint_splits.values())
    repeated = sum(count > 1 for count in Counter(fingerprints).values())
    return {
        "method": "stable random row hashing",
        "repeated_structural_groups": repeated,
        "structural_groups_crossing_splits": overlapping,
        "leakage_detected": overlapping > 0,
    }
