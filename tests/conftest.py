"""Shared deterministic test configuration for DecisionLab."""

from __future__ import annotations

import random

import numpy as np
import pytest

TEST_RANDOM_SEED = 20260828


@pytest.fixture(autouse=True)
def deterministic_random_generators() -> None:
    """Reset process-global random generators before every test."""
    random.seed(TEST_RANDOM_SEED)
    np.random.seed(TEST_RANDOM_SEED)
