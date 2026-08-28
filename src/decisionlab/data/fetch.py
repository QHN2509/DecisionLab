"""Fetch immutable choices13k source files from a pinned upstream revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "choices13k_manifest.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw"


class AcquisitionError(RuntimeError):
    """Raised when a source file cannot be acquired without altering raw data."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and minimally validate the acquisition manifest."""
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise AcquisitionError("Manifest must define at least one source file")
    for name, metadata in manifest["files"].items():
        if Path(name).name != name:
            raise AcquisitionError(f"Manifest filename is not a basename: {name!r}")
        if not isinstance(metadata, dict) or set(metadata) != {"url", "sha256"}:
            raise AcquisitionError(f"Invalid metadata for {name!r}")
        expected = metadata["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise AcquisitionError(f"Invalid SHA-256 for {name!r}")
    return manifest


def fetch_file(url: str, expected_sha256: str, destination: Path) -> str:
    """Download one file atomically, refusing to overwrite unexpected content."""
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise AcquisitionError(
                f"Refusing to overwrite {destination}: expected {expected_sha256}, found {actual}"
            )
        return "already present and verified"

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DecisionLab/0.1"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary)
        actual = sha256_file(temporary_path)
        if actual != expected_sha256:
            raise AcquisitionError(
                f"Checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, found {actual}"
            )
        os.chmod(temporary_path, 0o444)
        temporary_path.replace(destination)
        return "downloaded and verified"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def fetch_dataset(
    manifest_path: Path = DEFAULT_MANIFEST,
    destination_dir: Path = DEFAULT_DESTINATION,
) -> dict[str, str]:
    """Fetch every manifest file and return its acquisition status."""
    manifest = load_manifest(manifest_path)
    statuses: dict[str, str] = {}
    for name, metadata in manifest["files"].items():
        statuses[name] = fetch_file(metadata["url"], metadata["sha256"], destination_dir / name)
    return statuses


def main() -> None:
    """Run the choices13k acquisition command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    for name, status in fetch_dataset(args.manifest, args.destination).items():
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
