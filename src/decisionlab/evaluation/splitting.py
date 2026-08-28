"""Deterministic structural-group splitting for choices13k."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from decisionlab.data.validation import SelectionRecord

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One row's immutable structural group and split assignment."""

    row_index: int
    problem: int
    structural_fingerprint: str
    split: SplitName


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
