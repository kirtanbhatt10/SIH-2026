"""Phase 3 grouped split tests."""

from __future__ import annotations

import numpy as np

from ml.phase3.data import grouped_train_test_split, load_dataset_v2


def test_grouped_split_no_leakage(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    split.assert_no_group_leakage(dataset.metadata)
    assert split.method == "stratified_grouped_by_generation_group"
    assert len(split.train_idx) + len(split.test_idx) == len(dataset.labels)


def test_grouped_split_deterministic(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    split_a = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    split_b = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    assert np.array_equal(split_a.train_idx, split_b.train_idx)
    assert np.array_equal(split_a.test_idx, split_b.test_idx)


def test_actual_dataset_grouped_split():
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    split.assert_no_group_leakage(dataset.metadata)
    assert split.train_groups.isdisjoint(split.test_groups)
    assert len(split.train_idx) == 2039
    assert len(split.test_idx) == 461
