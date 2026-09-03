"""Phase 5 inference helper tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.phase5.config import N_FEATURES
from ml.phase5.inference import (
    InputValidationError,
    validate_and_prepare_batch,
    validate_and_prepare_features,
    validate_probability_matrix,
)


@pytest.fixture
def valid_sample():
    return np.linspace(0.1, 1.0, N_FEATURES)


@pytest.fixture
def valid_batch(valid_sample):
    return np.vstack([valid_sample, valid_sample + 0.01])


def test_validate_single_features(valid_sample):
    arr = validate_and_prepare_features(valid_sample)
    assert arr.shape == (N_FEATURES,)
    assert arr.dtype == np.float64


def test_validate_batch(valid_batch):
    arr = validate_and_prepare_batch(valid_batch)
    assert arr.shape == (valid_batch.shape[0], N_FEATURES)


def test_wrong_feature_count():
    with pytest.raises(InputValidationError):
        validate_and_prepare_features(np.zeros(10))


def test_nan_in_batch():
    x = np.ones((2, N_FEATURES))
    x[0, 0] = np.nan
    with pytest.raises(InputValidationError):
        validate_and_prepare_batch(x)


def test_probability_matrix_validation():
    proba = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    validate_probability_matrix(proba)

    bad = np.array([[0.5, 0.5, 0.5, 0.0, 0.0]])
    with pytest.raises(ValueError):
        validate_probability_matrix(bad)
