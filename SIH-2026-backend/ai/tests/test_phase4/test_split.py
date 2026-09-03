"""Phase 4 split reproduction and leakage tests."""

from __future__ import annotations

from ml.phase3.data import build_split_report, grouped_train_test_split, load_dataset_v2
from ml.phase4.config import SPLIT_METHOD, SPLIT_SEED, TEST_FRACTION


def test_exact_phase3_split_reproduction():
    dataset = load_dataset_v2()
    split = grouped_train_test_split(
        dataset.metadata,
        dataset.labels,
        test_fraction=TEST_FRACTION,
        seed=SPLIT_SEED,
    )
    report = build_split_report(split, dataset.metadata, dataset.labels)

    assert split.method == SPLIT_METHOD
    assert report["split_seed"] == 42
    assert report["train_size"] == 2039
    assert report["test_size"] == 461
    assert report["group_overlap_count"] == 0
    assert report["leakage_check_passed"] is True


def test_no_group_leakage():
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    split.assert_no_group_leakage(dataset.metadata)
