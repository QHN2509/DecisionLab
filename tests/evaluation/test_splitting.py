from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from decisionlab.data.validation import SelectionRecord
from decisionlab.evaluation.splitting import (
    InnerFoldAssignment,
    OuterFoldAssignment,
    SplitAssignment,
    audit_grouped_assignments,
    audit_nested_fold_assignments,
    create_grouped_assignments,
    create_nested_fold_assignments,
    ordinary_row_split_leakage_demo,
    structural_fingerprint,
)


def selection(
    problem: int,
    feedback: bool,
    *,
    ha: int = 10,
    la: int = 0,
    hb: int = 8,
    lb: int = 2,
) -> SelectionRecord:
    return SelectionRecord(
        problem=problem,
        feedback=feedback,
        n=15 if not feedback else 18,
        block=1 if not feedback else 3,
        ha=ha,
        pha=0.5,
        la=la,
        hb=hb,
        phb=0.5,
        lb=lb,
        lot_shape_b=0,
        lot_num_b=1,
        amb=False,
        corr=0,
        brate=0.4 if not feedback else 0.7,
        brate_std=0.2 if not feedback else 0.3,
    )


def description(ha: int = 10, la: int = 0, hb: int = 8, lb: int = 2) -> dict[str, Any]:
    return {"A": [[0.5, ha], [0.5, la]], "B": [[0.5, hb], [0.5, lb]]}


def test_fingerprint_ignores_feedback_block_sample_size_and_target() -> None:
    no_feedback = selection(1, False)
    feedback = selection(1, True)
    problem = description()

    assert structural_fingerprint(no_feedback, problem) == structural_fingerprint(feedback, problem)


def test_grouped_split_keeps_problem_variants_together() -> None:
    selections = [
        selection(1, False),
        selection(1, True),
        selection(2, False, ha=12),
        selection(2, True, ha=12),
        selection(3, False, ha=14),
        selection(3, True, ha=14),
    ]
    problems = {
        "0": description(),
        "1": description(),
        "2": description(ha=12),
        "3": description(ha=12),
        "4": description(ha=14),
        "5": description(ha=14),
    }

    assignments = create_grouped_assignments(
        selections,
        problems,
        seed="test-seed",
        train_fraction=0.34,
        validation_fraction=0.33,
        test_fraction=0.33,
    )

    split_by_problem: dict[int, set[str]] = {}
    for assignment in assignments:
        split_by_problem.setdefault(assignment.problem, set()).add(assignment.split)
    assert all(len(splits) == 1 for splits in split_by_problem.values())


def test_exact_structure_under_different_problem_ids_gets_same_split() -> None:
    selections = [selection(1, False), selection(999, True)]
    problems = {"0": description(), "1": description()}

    assignments = create_grouped_assignments(
        selections,
        problems,
        seed="test-seed",
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    assert assignments[0].structural_fingerprint == assignments[1].structural_fingerprint
    assert assignments[0].split == assignments[1].split


def test_audit_proves_zero_group_and_problem_overlap() -> None:
    selections: list[SelectionRecord] = []
    problems: dict[str, Any] = {}
    for problem_id in range(1, 101):
        for feedback in (False, True):
            row_index = len(selections)
            selections.append(selection(problem_id, feedback, ha=problem_id + 10))
            problems[str(row_index)] = description(ha=problem_id + 10)
    assignments = create_grouped_assignments(
        selections,
        problems,
        seed="broad-test-seed",
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    audit = audit_grouped_assignments(assignments)

    assert audit["status"] == "PASS"
    assert audit["structural_group_overlap_count"] == 0
    assert audit["problem_id_overlap_count"] == 0
    assert sum(audit["rows"].values()) == 200


def test_ordinary_row_split_demonstrates_paired_problem_leakage() -> None:
    selections: list[SelectionRecord] = []
    problems: dict[str, Any] = {}
    for problem_id in range(1, 101):
        for feedback in (False, True):
            row_index = len(selections)
            selections.append(selection(problem_id, feedback, ha=problem_id + 10))
            problems[str(row_index)] = description(ha=problem_id + 10)

    demo = ordinary_row_split_leakage_demo(
        selections,
        problems,
        seed="broad-test-seed",
        train_fraction=0.7,
        validation_fraction=0.15,
    )

    assert demo["repeated_structural_groups"] == 100
    assert demo["structural_groups_crossing_splits"] > 0
    assert demo["leakage_detected"] is True


def test_grouped_assignment_is_deterministic() -> None:
    selections = [selection(1, False), selection(1, True)]
    problems = {"0": description(), "1": description()}
    arguments = {
        "seed": "fixed",
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    }

    first = create_grouped_assignments(selections, problems, **arguments)
    second = create_grouped_assignments(selections, problems, **arguments)

    assert first == second


def test_assignment_audit_rejects_cross_partition_group_leakage() -> None:
    assignments = [
        SplitAssignment(0, 1, "same-structure", "train"),
        SplitAssignment(1, 1, "same-structure", "validation"),
        SplitAssignment(2, 2, "different-structure", "test"),
    ]

    with pytest.raises(ValueError, match="Split leakage detected"):
        audit_grouped_assignments(assignments)


def test_nested_folds_keep_groups_isolated_and_cover_each_outer_test_once() -> None:
    groups = np.asarray([f"group-{index // 2}" for index in range(60)])
    outer, inner = create_nested_fold_assignments(
        np.arange(60),
        groups,
        outer_folds=5,
        inner_folds=4,
        outer_seed=17,
        inner_seed=23,
    )

    audit = audit_nested_fold_assignments(outer, inner, outer_folds=5, inner_folds=4)

    assert audit["status"] == "PASS"
    assert audit["outer_group_overlap_count"] == 0
    assert audit["inner_group_overlap_count"] == 0
    assert [row.row_index for row in outer] == list(range(60))
    assert all(
        len({row.outer_fold for row in outer if row.structural_fingerprint == group}) == 1
        for group in np.unique(groups)
    )


def test_nested_audit_rejects_outer_test_row_inside_inner_cv() -> None:
    outer = [
        OuterFoldAssignment(0, 1, "a", 0),
        OuterFoldAssignment(1, 2, "b", 1),
    ]
    inner = [
        InnerFoldAssignment(0, 0, "a", 0),
        InnerFoldAssignment(1, 0, "a", 0),
    ]

    with pytest.raises(ValueError, match="Inner validation assignments"):
        audit_nested_fold_assignments(outer, inner, outer_folds=2, inner_folds=1)
