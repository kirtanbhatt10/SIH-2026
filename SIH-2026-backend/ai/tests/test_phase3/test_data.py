"""Phase 3 dataset loading and validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.phase3.config import EXPECTED_CLASSES, FEATURE_NAMES, N_FEATURES
from ml.phase3.data import DatasetValidationError, class_balance_report, load_dataset_v2


def test_load_actual_dataset_v2():
    dataset = load_dataset_v2()
    assert dataset.features.shape == (2500, 32)
    assert dataset.labels.shape == (2500,)
    assert len(dataset.metadata) == 2500
    assert dataset.feature_names == FEATURE_NAMES


def test_class_mapping():
    dataset = load_dataset_v2()
    assert sorted(dataset.class_names.keys()) == list(EXPECTED_CLASSES)
    assert dataset.class_names[0] == "benign"
    assert dataset.class_names[4] == "tone"


def test_class_balance():
    dataset = load_dataset_v2()
    balance = class_balance_report(dataset.labels)
    assert balance == {0: 500, 1: 500, 2: 500, 3: 500, 4: 500}


def test_metadata_alignment(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    assert len(dataset.metadata) == dataset.features.shape[0]


def test_duplicate_sample_id_rejection(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    X = np.ones((5, N_FEATURES))
    y = np.array([0, 1, 2, 3, 4])
    meta = pd.DataFrame({
        "sample_id": ["a", "a", "c", "d", "e"],
        "label": y,
        "generation_group": ["g1", "g1", "g2", "g3", "g4"],
    })
    np.save(root / "features.npy", X)
    np.save(root / "labels.npy", y)
    meta.to_csv(root / "metadata.csv", index=False)
    with pytest.raises(DatasetValidationError, match="Duplicate sample IDs"):
        load_dataset_v2(root)


def test_nan_rejection(tmp_path):
    root = tmp_path / "nan"
    root.mkdir()
    X = np.ones((5, N_FEATURES))
    X[0, 0] = np.nan
    y = np.array([0, 1, 2, 3, 4])
    meta = _valid_meta(y)
    np.save(root / "features.npy", X)
    np.save(root / "labels.npy", y)
    meta.to_csv(root / "metadata.csv", index=False)
    with pytest.raises(DatasetValidationError, match="NaN"):
        load_dataset_v2(root)


def test_inf_rejection(tmp_path):
    root = tmp_path / "inf"
    root.mkdir()
    X = np.ones((5, N_FEATURES))
    X[0, 0] = np.inf
    y = np.array([0, 1, 2, 3, 4])
    meta = _valid_meta(y)
    np.save(root / "features.npy", X)
    np.save(root / "labels.npy", y)
    meta.to_csv(root / "metadata.csv", index=False)
    with pytest.raises(DatasetValidationError, match="Inf"):
        load_dataset_v2(root)


def test_feature_count_contract(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    assert dataset.features.shape[1] == N_FEATURES
    assert len(dataset.feature_names) == 32


def _valid_meta(y):
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(len(y))],
        "label": y,
        "generation_group": [f"g{i}" for i in range(len(y))],
    })
