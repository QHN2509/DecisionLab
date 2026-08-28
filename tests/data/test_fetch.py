from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from decisionlab.data.fetch import AcquisitionError, fetch_file, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"decisionlab")

    assert sha256_file(source) == hashlib.sha256(b"decisionlab").hexdigest()


def test_fetch_refuses_to_overwrite_unexpected_raw_file(tmp_path: Path) -> None:
    destination = tmp_path / "raw.csv"
    destination.write_bytes(b"unexpected")

    with pytest.raises(AcquisitionError, match="Refusing to overwrite"):
        fetch_file("https://example.invalid/raw.csv", "0" * 64, destination)

    assert destination.read_bytes() == b"unexpected"


def test_fetch_accepts_already_verified_file_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "raw.csv"
    destination.write_bytes(b"expected")
    digest = hashlib.sha256(b"expected").hexdigest()

    assert fetch_file("https://example.invalid/raw.csv", digest, destination) == (
        "already present and verified"
    )
