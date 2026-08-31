"""Repository-level configuration contracts."""

from __future__ import annotations

import json
import tomllib
from csv import DictReader
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/baselines.json"),
        Path("configs/behavioral_analysis.json"),
        Path("configs/eda.json"),
        Path("configs/error_analysis.json"),
        Path("configs/model_selection.json"),
    ],
)
def test_stochastic_workflows_declare_integer_random_seeds(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert type(config["random_seed"]) is int
    assert config["random_seed"] >= 0


def test_nested_group_cv_declares_deterministic_integer_seeds() -> None:
    config = json.loads(Path("configs/evaluation.json").read_text(encoding="utf-8"))

    assert type(config["outer_seed"]) is int
    assert type(config["inner_seed"]) is int
    assert config["dataset_role"] == "development"
    assert config["historical_partitions_role"] == "development_only_not_confirmatory"


def test_model_selection_uses_the_canonical_nested_fold_contract() -> None:
    evaluation = json.loads(Path("configs/evaluation.json").read_text(encoding="utf-8"))
    model = json.loads(Path("configs/model_selection.json").read_text(encoding="utf-8"))
    nested = model["nested_cross_validation"]

    assert model["evaluation_design"] == evaluation["protocol_name"]
    for key in ("outer_folds", "inner_folds", "outer_seed", "inner_seed", "primary_metric"):
        assert nested[key] == evaluation[key]


def test_locked_environment_contains_every_direct_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = {
        line.strip().lower()
        for line in Path("requirements-dev.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    declared = [*project["dependencies"], *project["optional-dependencies"]["dev"]]

    assert {requirement.lower() for requirement in declared} <= locked


def test_readme_quick_and_official_workflows_are_complete_and_ordered() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    quick = readme.split("## Quick start: launch the tracked demo", maxsplit=1)[1].split(
        "## Local reproduction in one worktree", maxsplit=1
    )[0]
    assert "streamlit run app.py" in quick
    assert "decisionlab-fetch-data" not in quick

    official = readme.split("## Official provenance-controlled regeneration", maxsplit=1)[1].split(
        "## Repository layout", maxsplit=1
    )[0]
    official_commands = [
        "decisionlab-fetch-data",
        "decisionlab-validate-data",
        "decisionlab-eda",
        "decisionlab-build-features",
        "decisionlab-create-folds",
        "decisionlab-run-baselines",
        "decisionlab-select-model",
        "decisionlab-behavioral-analysis",
        "decisionlab-error-analysis",
        "decisionlab-build-app-metadata",
        "python -m pytest",
        "streamlit run app.py",
    ]

    positions = [official.index(command) for command in official_commands]
    assert positions == sorted(positions)
    assert official.count("git add -A && git commit") == 8


def test_official_analysis_reports_describe_the_recorded_oof_scope() -> None:
    behavioral = Path("reports/behavioral_analysis.md").read_text(encoding="utf-8")
    errors = Path("reports/error_analysis.md").read_text(encoding="utf-8")
    statistics = json.loads(
        Path("artifacts/analysis/behavioral/statistics.json").read_text(encoding="utf-8")
    )

    assert statistics["participant_count_threshold_source"] == "development_data_median"
    assert "complete development-data median (`n = 16`)" in behavioral
    assert "complete nested outer out-of-fold predictions" in errors
    assert "post-selection analysis on validation data" not in errors


@pytest.mark.parametrize(
    ("markdown_path", "csv_path", "official_table"),
    [
        (
            Path("reports/tables/baseline_comparison.md"),
            Path("reports/tables/baseline_comparison.csv"),
            "nested_baselines.csv",
        ),
        (
            Path("reports/tables/model_comparison.md"),
            Path("reports/tables/model_comparison.csv"),
            "nested_model_comparison.csv",
        ),
    ],
)
def test_legacy_comparison_tables_are_unambiguously_labeled(
    markdown_path: Path,
    csv_path: Path,
    official_table: str,
) -> None:
    markdown = markdown_path.read_text(encoding="utf-8")
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(DictReader(csv_file))

    assert markdown.startswith("> **STALE / LEGACY:**")
    assert official_table in markdown
    assert rows
    assert {row["artifact_status"] for row in rows} == {"stale_legacy"}
    assert all("Not part of the current official" in row["artifact_note"] for row in rows)
    assert all(official_table in row["artifact_note"] for row in rows)
