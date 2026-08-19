"""Dataset V2 loading and leakage-aware grouped splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CLASS_NAMES,
    DATASET_V2_DIR,
    EXPECTED_CLASSES,
    FEATURE_NAMES,
    N_FEATURES,
    SPLIT_SEED,
    TEST_FRACTION,
    load_dataset_info,
)


class DatasetValidationError(ValueError):
    """Raised when Dataset V2 fails a mandatory validation check."""


@dataclass(frozen=True)
class DatasetV2:
    features: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    class_names: dict[int, str]
    dataset_info: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


def load_dataset_v2(dataset_dir: Path | None = None) -> DatasetV2:
    """Load and validate Dataset V2 from disk."""
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_V2_DIR
    features_path = root / "features.npy"
    labels_path = root / "labels.npy"
    metadata_path = root / "metadata.csv"

    for path in (features_path, labels_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Dataset V2 file missing: {path}")

    X = np.load(features_path)
    y = np.load(labels_path)
    metadata = pd.read_csv(metadata_path)
    dataset_info = load_dataset_info() if dataset_dir is None else _load_info(root)

    _validate_dataset(X, y, metadata, dataset_info)
    return DatasetV2(
        features=X,
        labels=y,
        metadata=metadata,
        feature_names=list(FEATURE_NAMES),
        class_names=dict(CLASS_NAMES),
        dataset_info=dataset_info,
    )


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "dataset_info.json"
    if info_path.exists():
        import json

        with info_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    dataset_info: dict[str, Any],
) -> None:
    if X.ndim != 2:
        raise DatasetValidationError(f"X must be 2-dimensional, got shape {X.shape}")
    if X.shape[1] != N_FEATURES:
        raise DatasetValidationError(
            f"Expected {N_FEATURES} features, got {X.shape[1]}"
        )
    if len(y) != X.shape[0]:
        raise DatasetValidationError(
            f"Label count {len(y)} does not match sample count {X.shape[0]}"
        )
    if np.isnan(X).any():
        raise DatasetValidationError("X contains NaN values")
    if np.isinf(X).any():
        raise DatasetValidationError("X contains Inf values")

    classes = sorted(int(v) for v in np.unique(y))
    if classes != list(EXPECTED_CLASSES):
        raise DatasetValidationError(
            f"Expected classes {list(EXPECTED_CLASSES)}, found {classes}"
        )

    if len(metadata) != X.shape[0]:
        raise DatasetValidationError(
            f"metadata row count {len(metadata)} != sample count {X.shape[0]}"
        )
    if metadata["sample_id"].duplicated().any():
        dupes = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].tolist()
        raise DatasetValidationError(
            f"Duplicate sample IDs detected: {dupes[:5]}"
        )

    if "generation_group" not in metadata.columns:
        raise DatasetValidationError("metadata.csv missing generation_group column")

    info_names = dataset_info.get("feature_names")
    if info_names is not None and list(info_names) != list(FEATURE_NAMES):
        raise DatasetValidationError(
            "Dataset feature_names do not match Phase-1 feature contract"
        )


def class_balance_report(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    method: str
    seed: int
    test_fraction: float
    train_groups: set[str]
    test_groups: set[str]
    fallback_reason: str | None = None

    def assert_no_group_leakage(self, metadata: pd.DataFrame) -> None:
        train_groups = set(metadata.iloc[self.train_idx]["generation_group"])
        test_groups = set(metadata.iloc[self.test_idx]["generation_group"])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(
                f"generation_group leakage detected: {sorted(overlap)[:5]}"
            )


def grouped_train_test_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> SplitResult:
    """Stratified grouped split using generation_group (class-homogeneous groups)."""
    if "generation_group" not in metadata.columns:
        raise ValueError("metadata must contain generation_group")

    # Verify groups are class-homogeneous; if not, document fallback.
    mixed = metadata.groupby("generation_group")["label"].nunique()
    if (mixed > 1).any():
        return _fallback_random_split(metadata, labels, test_fraction, seed)

    rng = np.random.RandomState(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    train_groups: set[str] = set()
    test_groups: set[str] = set()

    for label in sorted(metadata["label"].unique()):
        class_meta = metadata[metadata["label"] == label]
        groups = class_meta["generation_group"].unique().copy()
        rng.shuffle(groups)
        n_test_groups = max(1, int(round(len(groups) * test_fraction)))
        test_class_groups = set(groups[:n_test_groups])
        train_class_groups = set(groups[n_test_groups:])

        train_mask = class_meta["generation_group"].isin(train_class_groups)
        test_mask = class_meta["generation_group"].isin(test_class_groups)

        train_idx.extend(class_meta.index[train_mask].tolist())
        test_idx.extend(class_meta.index[test_mask].tolist())
        train_groups.update(train_class_groups)
        test_groups.update(test_class_groups)

    result = SplitResult(
        train_idx=np.array(sorted(train_idx), dtype=int),
        test_idx=np.array(sorted(test_idx), dtype=int),
        method="stratified_grouped_by_generation_group",
        seed=seed,
        test_fraction=test_fraction,
        train_groups=train_groups,
        test_groups=test_groups,
    )
    result.assert_no_group_leakage(metadata)
    return result


def _fallback_random_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> SplitResult:
    """Deterministic fallback when generation groups are not class-homogeneous."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        stratify=labels,
        random_state=seed,
    )
    train_groups = set(metadata.iloc[train_idx]["generation_group"])
    test_groups = set(metadata.iloc[test_idx]["generation_group"])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(
            "Fallback stratified split still produced generation_group overlap; "
            "cannot proceed without leakage."
        )
    return SplitResult(
        train_idx=np.array(sorted(train_idx), dtype=int),
        test_idx=np.array(sorted(test_idx), dtype=int),
        method="stratified_random_fallback",
        seed=seed,
        test_fraction=test_fraction,
        train_groups=train_groups,
        test_groups=test_groups,
        fallback_reason=(
            "generation_group is not class-homogeneous; used stratified index split "
            "and verified no generation_group overlap"
        ),
    )


def build_split_report(
    split: SplitResult,
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, Any]:
    train_labels = labels[split.train_idx]
    test_labels = labels[split.test_idx]
    return {
        "split_method": split.method,
        "split_seed": split.seed,
        "test_fraction": split.test_fraction,
        "train_size": int(len(split.train_idx)),
        "test_size": int(len(split.test_idx)),
        "train_class_counts": class_balance_report(train_labels),
        "test_class_counts": class_balance_report(test_labels),
        "n_train_groups": len(split.train_groups),
        "n_test_groups": len(split.test_groups),
        "group_overlap_count": len(split.train_groups & split.test_groups),
        "generation_group_policy": (
            "Assign entire generation_group to train or test; "
            "never split a group across partitions."
        ),
        "fallback_reason": split.fallback_reason,
        "leakage_check_passed": len(split.train_groups & split.test_groups) == 0,
    }
