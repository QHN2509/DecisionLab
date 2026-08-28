"""Repository-level configuration contracts."""

from __future__ import annotations

import json
import tomllib
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


def test_group_split_declares_stable_nonempty_seed() -> None:
    config = json.loads(Path("configs/evaluation.json").read_text(encoding="utf-8"))

    assert isinstance(config["split_seed"], str)
    assert config["split_seed"].strip()


def test_locked_environment_contains_every_direct_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = {
        line.strip().lower()
        for line in Path("requirements-dev.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    declared = [*project["dependencies"], *project["optional-dependencies"]["dev"]]

    assert {requirement.lower() for requirement in declared} <= locked


def test_readme_clean_workflow_commands_are_complete_and_ordered() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    commands = [
        "python3 -m venv .venv",
        "python -m pip install -r requirements-dev.lock",
        "python -m pip install --no-build-isolation --no-deps -e .",
        "decisionlab-fetch-data",
        "decisionlab-validate-data",
        "decisionlab-eda",
        "decisionlab-build-features",
        "decisionlab-create-splits",
        "decisionlab-run-baselines",
        "decisionlab-select-model",
        "decisionlab-behavioral-analysis",
        "decisionlab-error-analysis",
        "python -m pytest",
        "streamlit run app.py",
    ]

    positions = [readme.index(command) for command in commands]
    assert positions == sorted(positions)
