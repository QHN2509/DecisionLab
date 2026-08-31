from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decisionlab.analysis import behavioral as behavioral_analysis
from decisionlab.analysis import eda, errors
from decisionlab.app import metadata as app_metadata
from decisionlab.experiments import baselines, model_selection
from decisionlab.experiments.provenance import (
    PROJECT_ROOT,
    DirtyRepositoryError,
    collect_git_state,
    collect_transitive_pipeline_hashes,
    finalize_run_provenance,
    hash_file_set,
    start_run_provenance,
    validate_upstream_artifacts,
)
from decisionlab.features import behavioral as behavioral_features


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    dataset = raw / "data.csv"
    dataset.write_text("predictor\n1\n", encoding="utf-8")
    checksum = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "choices13k-test",
                "source_repository": "https://example.test/data",
                "commit": "dataset-revision",
                "files": {"data.csv": {"sha256": checksum}},
            }
        ),
        encoding="utf-8",
    )
    return raw, manifest


def test_git_state_records_commit_and_dirty_boolean() -> None:
    state = collect_git_state()

    assert len(state["commit_sha"]) == 40
    assert state["worktree_state"] in {"clean", "dirty"}
    assert state["is_dirty"] == (state["worktree_state"] == "dirty")


def test_file_set_combined_hash_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    forward = hash_file_set([first, second], repo_root=tmp_path)
    reverse = hash_file_set([second, first], repo_root=tmp_path)
    second.write_text("changed", encoding="utf-8")
    changed = hash_file_set([first, second], repo_root=tmp_path)

    assert forward == reverse
    assert changed["combined_sha256"] != forward["combined_sha256"]


def test_transitive_source_hashes_cover_pipeline_dependencies() -> None:
    values = collect_transitive_pipeline_hashes(
        PROJECT_ROOT / "src" / "decisionlab" / "experiments" / "model_selection.py"
    )

    covered = set(values["files"])
    assert "src/decisionlab/experiments/model_selection.py" in covered
    assert "src/decisionlab/features/behavioral.py" in covered
    assert "src/decisionlab/evaluation/metrics.py" in covered
    assert "src/decisionlab/models/ensembles.py" in covered


def test_dirty_repository_is_refused_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "decisionlab.experiments.provenance.collect_git_state",
        lambda root: {
            "commit_sha": "a" * 40,
            "worktree_state": "dirty",
            "is_dirty": True,
            "status_porcelain": [" M source.py"],
        },
    )

    with pytest.raises(DirtyRepositoryError, match="repository is dirty"):
        start_run_provenance(
            experiment_name="test",
            config_paths=[],
            dataset_manifest_path=Path("unused"),
            raw_dir=Path("unused"),
            fold_specification_identifier="folds-v1",
            entry_module=Path("unused"),
        )


def test_dirty_override_is_warned_and_marked_non_official(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, manifest = _dataset_fixture(tmp_path)
    config = tmp_path / "config.json"
    lock = tmp_path / "lock.txt"
    entry = tmp_path / "entry.py"
    config.write_text("{}", encoding="utf-8")
    lock.write_text("package==1", encoding="utf-8")
    entry.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "decisionlab.experiments.provenance.collect_git_state",
        lambda root: {
            "commit_sha": "b" * 40,
            "worktree_state": "dirty",
            "is_dirty": True,
            "status_porcelain": [" M source.py"],
        },
    )

    with pytest.warns(RuntimeWarning, match="marked non-official"):
        provenance = start_run_provenance(
            experiment_name="test",
            config_paths=[config],
            dataset_manifest_path=manifest,
            raw_dir=raw,
            fold_specification_identifier="folds-v1",
            entry_module=entry,
            allow_dirty=True,
            repo_root=tmp_path,
            lockfiles=[lock],
        )

    assert provenance["run_eligibility"] == "non_official_dirty_worktree"
    assert provenance["dataset"]["revision"] == "dataset-revision"
    assert provenance["configuration"]["combined_sha256"]
    assert provenance["configuration"]["complete_sha256"]
    assert provenance["environment"]["combined_sha256"]
    assert provenance["environment"]["environment_sha256"]
    assert provenance["environment"]["runtime"]["python_version"]


def test_complete_configuration_hash_includes_explicit_run_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, manifest = _dataset_fixture(tmp_path)
    config = tmp_path / "config.json"
    lock = tmp_path / "lock.txt"
    entry = tmp_path / "entry.py"
    config.write_text('{"folds":5}', encoding="utf-8")
    lock.write_text("package==1", encoding="utf-8")
    entry.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "decisionlab.experiments.provenance.collect_git_state",
        lambda root: {
            "commit_sha": "a" * 40,
            "worktree_state": "clean",
            "is_dirty": False,
            "status_porcelain": [],
        },
    )

    first = start_run_provenance(
        experiment_name="test",
        config_paths=[config],
        configuration_values={"seed": 1},
        dataset_manifest_path=manifest,
        raw_dir=raw,
        fold_specification_identifier="folds-v1",
        entry_module=entry,
        repo_root=tmp_path,
        lockfiles=[lock],
    )
    second = start_run_provenance(
        experiment_name="test",
        config_paths=[config],
        configuration_values={"seed": 2},
        dataset_manifest_path=manifest,
        raw_dir=raw,
        fold_specification_identifier="folds-v1",
        entry_module=entry,
        repo_root=tmp_path,
        lockfiles=[lock],
    )

    assert first["configuration"]["combined_sha256"] == second["configuration"]["combined_sha256"]
    assert first["configuration"]["complete_sha256"] != second["configuration"]["complete_sha256"]


def test_finalize_records_fold_and_every_declared_output_hash(tmp_path: Path) -> None:
    fold = tmp_path / "folds.csv"
    input_artifact = tmp_path / "upstream-provenance.json"
    output = tmp_path / "metrics.json"
    manifest = tmp_path / "provenance.json"
    fold.write_text("fold\n0\n", encoding="utf-8")
    input_artifact.write_text('{"run_eligibility":"official_clean_commit"}', encoding="utf-8")
    output.write_text("{}", encoding="utf-8")
    base = {"fold_specification": {"identifier": "folds-v1", "artifact_hashes": None}}

    result = finalize_run_provenance(
        base,
        fold_artifacts={"outer": fold},
        input_artifacts=[input_artifact],
        output_artifacts=[output],
        output_path=manifest,
        repo_root=tmp_path,
    )

    assert result["fold_specification"]["artifact_hashes"]["outer"]
    assert result["inputs"]["files"]["upstream-provenance.json"]
    assert result["outputs"]["files"] == {"metrics.json": hashlib.sha256(b"{}").hexdigest()}
    assert result["provenance_file"]["self_hash_in_outputs"] is False
    assert json.loads(manifest.read_text(encoding="utf-8")) == result


def test_upstream_validation_requires_official_matching_outputs(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "provenance_schema": "decisionlab_run_provenance_v1",
                "run_eligibility": "official_clean_commit",
                "outputs": {"files": {"metrics.json": hashlib.sha256(b"{}").hexdigest()}},
            }
        ),
        encoding="utf-8",
    )

    result = validate_upstream_artifacts(provenance_path, [artifact], repo_root=tmp_path)
    assert result["run_eligibility"] == "official_clean_commit"

    artifact.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash differs"):
        validate_upstream_artifacts(provenance_path, [artifact], repo_root=tmp_path)


def test_non_official_smoke_manifest_contains_complete_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, dataset_manifest = _dataset_fixture(tmp_path)
    config = tmp_path / "config.json"
    lock = tmp_path / "requirements.lock"
    entry = tmp_path / "runner.py"
    folds = tmp_path / "folds.csv"
    input_artifact = tmp_path / "input.json"
    output_artifact = tmp_path / "output.json"
    provenance_path = tmp_path / "provenance.json"
    config.write_text('{"seed":17}', encoding="utf-8")
    lock.write_text("package==1", encoding="utf-8")
    entry.write_text("VALUE = 1\n", encoding="utf-8")
    folds.write_text("group,fold\na,0\n", encoding="utf-8")
    input_artifact.write_text('{"upstream":true}', encoding="utf-8")
    output_artifact.write_text('{"metric":0.1}', encoding="utf-8")
    monkeypatch.setattr(
        "decisionlab.experiments.provenance.collect_git_state",
        lambda root: {
            "commit_sha": "c" * 40,
            "worktree_state": "dirty",
            "is_dirty": True,
            "status_porcelain": [" M smoke.py"],
        },
    )

    with pytest.warns(RuntimeWarning, match="marked non-official"):
        started = start_run_provenance(
            experiment_name="smoke",
            config_paths=[config],
            configuration_values={"seed": 17, "mode": "smoke"},
            dataset_manifest_path=dataset_manifest,
            raw_dir=raw,
            fold_specification_identifier="grouped-cv-smoke-v1",
            entry_module=entry,
            allow_dirty=True,
            repo_root=tmp_path,
            lockfiles=[lock],
        )
    completed = finalize_run_provenance(
        started,
        fold_artifacts={"outer": folds},
        input_artifacts=[input_artifact],
        output_artifacts=[output_artifact],
        output_path=provenance_path,
        repo_root=tmp_path,
    )

    assert completed["git"]["commit_sha"] == "c" * 40
    assert completed["run_eligibility"] == "non_official_dirty_worktree"
    assert completed["dataset"]["revision"] == "dataset-revision"
    assert completed["dataset"]["checksums"]
    assert completed["configuration"]["values"] == {"seed": 17, "mode": "smoke"}
    assert completed["configuration"]["complete_sha256"]
    assert completed["environment"]["environment_sha256"]
    assert completed["fold_specification"]["identifier"] == "grouped-cv-smoke-v1"
    assert completed["fold_specification"]["artifact_hashes"]["outer"]
    assert completed["pipeline_source"]["files"] == {
        "runner.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()
    }
    assert completed["inputs"]["files"]["input.json"]
    assert completed["outputs"]["files"]["output.json"]


@pytest.mark.parametrize(
    ("module", "runner", "fold_identifier"),
    [
        (baselines, baselines.run_nested_baseline_experiment, "nested_grouped_cv_v1"),
        (model_selection, model_selection.run_model_selection_experiment, "nested_grouped_cv_v1"),
        (
            behavioral_features,
            behavioral_features.run_feature_build,
            "not_applicable_feature_build",
        ),
        (eda, eda.run_eda, "not_applicable_target_aware_eda"),
        (
            behavioral_analysis,
            behavioral_analysis.run_behavioral_analysis,
            "nested_grouped_cv_v1",
        ),
        (errors, errors.run_error_analysis, "nested_grouped_cv_v1"),
        (
            app_metadata,
            app_metadata.build_application_metadata,
            "nested_grouped_cv_v1_application_metadata",
        ),
    ],
)
def test_official_runners_collect_provenance_before_pipeline_work(
    module: object,
    runner: object,
    fold_identifier: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProvenanceStarted(RuntimeError):
        pass

    def stop_after_start(**kwargs: object) -> None:
        assert kwargs["fold_specification_identifier"] == fold_identifier
        assert kwargs["allow_dirty"] is False
        assert "configuration_values" in kwargs
        raise ProvenanceStarted

    monkeypatch.setattr(module, "start_run_provenance", stop_after_start)

    assert "start_run_provenance" in runner.__code__.co_names  # type: ignore[attr-defined]
    assert "finalize_run_provenance" in runner.__code__.co_names  # type: ignore[attr-defined]

    with pytest.raises(ProvenanceStarted):
        runner()  # type: ignore[operator]
