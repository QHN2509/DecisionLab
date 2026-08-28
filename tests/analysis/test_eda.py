from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from decisionlab.analysis.eda import (
    build_analysis_arrays,
    compute_eda_statistics,
    load_eda_config,
    write_report,
)
from decisionlab.data.validation import SelectionRecord


def record(problem: int, feedback: bool, brate: float) -> SelectionRecord:
    return SelectionRecord(
        problem=problem,
        feedback=feedback,
        n=16,
        block=2 if feedback else 1,
        ha=10,
        pha=0.5,
        la=0,
        hb=8,
        phb=0.5,
        lb=2,
        lot_shape_b=0,
        lot_num_b=1,
        amb=False,
        corr=0,
        brate=brate,
        brate_std=0.2,
    )


def analysis_fixture() -> tuple[list[SelectionRecord], dict[str, object]]:
    selections = [record(1, False, 0.4), record(1, True, 0.6), record(2, True, 0.8)]
    description = {"A": [[0.5, 10], [0.5, 0]], "B": [[0.5, 8], [0.5, 2]]}
    problems = {str(index): description for index in range(len(selections))}
    return selections, problems


def test_build_analysis_arrays_derives_expected_values_and_variances() -> None:
    selections, problems = analysis_fixture()

    arrays = build_analysis_arrays(selections, problems)

    assert arrays["ev_a"] == pytest.approx([5.0, 5.0, 5.0])
    assert arrays["ev_b"] == pytest.approx([5.0, 5.0, 5.0])
    assert arrays["variance_a"] == pytest.approx([25.0, 25.0, 25.0])
    assert arrays["variance_b"] == pytest.approx([9.0, 9.0, 9.0])
    assert arrays["estimated_brate_se"] == pytest.approx([0.05, 0.05, 0.05])


def test_statistics_use_matched_problem_feedback_difference() -> None:
    selections, problems = analysis_fixture()
    statistics = compute_eda_statistics(build_analysis_arrays(selections, problems))

    paired = statistics["paired_feedback"]
    assert paired["pair_count"] == 1
    assert paired["difference_feedback_minus_no_feedback"]["mean"] == pytest.approx(0.2)
    assert paired["feedback_mean_bRate"] == pytest.approx(0.6)


def test_load_eda_config_rejects_missing_settings(tmp_path: Path) -> None:
    config = tmp_path / "eda.json"
    config.write_text(json.dumps({"random_seed": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="EDA config"):
        load_eda_config(config)


def test_generated_report_contains_required_sections(tmp_path: Path) -> None:
    selections, problems = analysis_fixture()
    statistics = compute_eda_statistics(build_analysis_arrays(selections, problems))
    statistics["source_commit"] = "test-commit"
    figures = [tmp_path / f"figure-{index}.png" for index in range(5)]
    output = tmp_path / "eda_summary.md"

    write_report(statistics, figures, output)

    report = output.read_text(encoding="utf-8")
    assert "## 1. Important findings" in report
    assert "## 2. Potential modeling issues" in report
    assert "## 3. Possible leakage risks" in report
    assert "## 4. Questions worth testing later" in report


def test_no_analysis_array_uses_brate_std_as_a_model_feature() -> None:
    selections, problems = analysis_fixture()
    arrays = build_analysis_arrays(selections, problems)

    assert np.allclose(arrays["estimated_brate_se"], arrays["brate_std"] / np.sqrt(arrays["n"]))
