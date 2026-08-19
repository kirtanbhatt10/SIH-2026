"""Shared fixtures for Phase 5 tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.phase5.config import N_FEATURES


@pytest.fixture
def valid_sample():
    return np.linspace(0.1, 1.0, N_FEATURES)


@pytest.fixture
def valid_batch(valid_sample):
    return np.vstack([valid_sample, valid_sample + 0.01, valid_sample + 0.02])
