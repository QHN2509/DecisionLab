"""Centralized, automatic provenance for DecisionLab experiment runs."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decisionlab.data.fetch import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCKFILES = (
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "requirements-dev.lock",
)


class DirtyRepositoryError(RuntimeError):
    """Raised when an official run is requested from a dirty repository."""


class IneligibleUpstreamError(RuntimeError):
    """Raised when a downstream official run depends on non-official artifacts."""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _combined_hash(values: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(sorted(values.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_file_set(paths: Iterable[Path], *, repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Hash a named file set and a deterministic manifest of those hashes."""
    hashes = {_relative(path, repo_root): sha256_file(path) for path in paths}
    return {"files": dict(sorted(hashes.items())), "combined_sha256": _combined_hash(hashes)}


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_git_state(repo_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Collect the exact commit and pre-run worktree state."""
    commit = _git(repo_root, "rev-parse", "HEAD")
    status_text = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    entries = status_text.splitlines() if status_text else []
    return {
        "commit_sha": commit,
        "worktree_state": "dirty" if entries else "clean",
        "is_dirty": bool(entries),
        "status_porcelain": entries,
    }


def _decisionlab_module_path(module: str, repo_root: Path) -> Path | None:
    if module == "decisionlab":
        candidate = repo_root / "src" / "decisionlab" / "__init__.py"
        return candidate if candidate.exists() else None
    if not module.startswith("decisionlab."):
        return None
    relative = module.split(".")[1:]
    file_candidate = repo_root / "src" / "decisionlab" / Path(*relative).with_suffix(".py")
    if file_candidate.exists():
        return file_candidate
    package_candidate = repo_root / "src" / "decisionlab" / Path(*relative) / "__init__.py"
    return package_candidate if package_candidate.exists() else None


def collect_transitive_pipeline_hashes(
    entry_module: Path, *, repo_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Hash the entry module and all statically imported DecisionLab modules."""
    pending = [entry_module.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                modules.update(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            dependency = _decisionlab_module_path(module, repo_root)
            if dependency is not None and dependency.resolve() not in visited:
                pending.append(dependency.resolve())
    return hash_file_set(sorted(visited), repo_root=repo_root)


def validate_upstream_artifacts(
    provenance_path: Path,
    required_artifacts: Iterable[Path],
    *,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Require an official upstream manifest and verify each consumed artifact hash."""
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Regenerated upstream provenance is required: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("provenance_schema") != "decisionlab_run_provenance_v1":
        raise IneligibleUpstreamError("Upstream artifact has no recognized provenance schema")
    if provenance.get("run_eligibility") != "official_clean_commit":
        raise IneligibleUpstreamError("Upstream artifacts are not eligible official outputs")
    recorded = provenance.get("outputs", {}).get("files", {})
    for path in required_artifacts:
        name = _relative(path, repo_root)
        if recorded.get(name) != sha256_file(path):
            raise IneligibleUpstreamError(
                f"Upstream artifact is missing from provenance or its hash differs: {name}"
            )
    return provenance


def _read_dataset_manifest(path: Path, raw_dir: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {name: values["sha256"] for name, values in manifest["files"].items()}
    actual = {name: sha256_file(raw_dir / name) for name in expected}
    if actual != expected:
        raise ValueError("Dataset files do not match the pinned manifest checksums")
    return {
        "name": manifest["dataset"],
        "source_repository": manifest["source_repository"],
        "revision": manifest["commit"],
        "checksums": actual,
        "manifest_sha256": sha256_file(path),
    }


def start_run_provenance(
    *,
    experiment_name: str,
    config_paths: Iterable[Path],
    dataset_manifest_path: Path,
    raw_dir: Path,
    fold_specification_identifier: str,
    entry_module: Path,
    configuration_values: Mapping[str, Any] | None = None,
    allow_dirty: bool = False,
    repo_root: Path = PROJECT_ROOT,
    lockfiles: Iterable[Path] = DEFAULT_LOCKFILES,
) -> dict[str, Any]:
    """Capture immutable pre-run context and enforce clean official runs by default."""
    git = collect_git_state(repo_root)
    if git["is_dirty"] and not allow_dirty:
        raise DirtyRepositoryError(
            "Official experiment refused: repository is dirty. Commit or stash changes, "
            "or pass allow_dirty=True for a clearly marked non-official run."
        )
    eligibility = "official_clean_commit"
    if git["is_dirty"]:
        eligibility = "non_official_dirty_worktree"
        warnings.warn(
            "Experiment is running from a dirty repository and is marked non-official.",
            RuntimeWarning,
            stacklevel=2,
        )
    environment = hash_file_set(lockfiles, repo_root=repo_root)
    environment["runtime"] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    environment["environment_sha256"] = hashlib.sha256(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    configuration = hash_file_set(config_paths, repo_root=repo_root)
    configuration["values"] = dict(configuration_values or {})
    configuration["complete_sha256"] = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "provenance_schema": "decisionlab_run_provenance_v1",
        "experiment_name": experiment_name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "run_eligibility": eligibility,
        "git": git,
        "dataset": _read_dataset_manifest(dataset_manifest_path, raw_dir),
        "configuration": configuration,
        "environment": environment,
        "fold_specification": {
            "identifier": fold_specification_identifier,
            "artifact_hashes": None,
        },
        "pipeline_source": collect_transitive_pipeline_hashes(entry_module, repo_root=repo_root),
        "outputs": None,
        "legacy_artifacts_reused": False,
    }


def finalize_run_provenance(
    provenance: dict[str, Any],
    *,
    fold_artifacts: Mapping[str, Path],
    input_artifacts: Iterable[Path] = (),
    output_artifacts: Iterable[Path],
    output_path: Path,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Attach exact fold/output hashes and atomically write the run manifest."""
    result = deepcopy(provenance)
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    result["fold_specification"]["artifact_hashes"] = {
        name: sha256_file(path) for name, path in sorted(fold_artifacts.items())
    }
    result["inputs"] = hash_file_set(input_artifacts, repo_root=repo_root)
    result["outputs"] = hash_file_set(output_artifacts, repo_root=repo_root)
    result["provenance_file"] = {
        "path": _relative(output_path, repo_root),
        "self_hash_in_outputs": False,
        "reason": "A manifest cannot include its own stable content hash.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return result
