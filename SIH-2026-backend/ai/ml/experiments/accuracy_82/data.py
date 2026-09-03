"""Dataset and split loading for the accuracy_82 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.phase3.data import load_dataset_v2
from ml.promotion.splits import (
    build_frozen_manifest,
    build_three_way_split,
    load_frozen_manifest,
    save_frozen_manifest,
)

from .config import (
    CAL_DEV_SEED,
    CLEAN_DATASET_DIR,
    EXPERIMENT_DATASET_DIR,
    EXPERIMENT_DATASET_VERSION,
    EXPERIMENT_SPLIT_MANIFEST,
    GENERATOR_VERSION,
    INNER_DEV_FRACTION,
    INNER_DEV_SEED,
    OUTER_SPLIT_SEED,
)


@dataclass(frozen=True)
class ExperimentData:
    features: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    model_train_idx: np.ndarray
    cal_dev_idx: np.ndarray
    locked_test_idx: np.ndarray
    train_fit_idx: np.ndarray
    inner_dev_idx: np.ndarray
    dataset_dir: str
    split_manifest_path: str

    @property
    def development_idx(self) -> np.ndarray:
        return np.concatenate([self.model_train_idx, self.cal_dev_idx])


def _split_inner_dev(
    metadata: pd.DataFrame,
    model_train_idx: np.ndarray,
    *,
    fraction: float = INNER_DEV_FRACTION,
    seed: int = INNER_DEV_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_meta = metadata.iloc[model_train_idx]
    train_fit: list[int] = []
    inner_dev: list[int] = []

    for label in sorted(train_meta["label"].unique()):
        class_meta = train_meta[train_meta["label"] == label]
        groups = class_meta["generation_group"].unique().copy()
        rng.shuffle(groups)
        n_dev_groups = max(1, int(round(len(groups) * fraction)))
        dev_groups = set(groups[:n_dev_groups])
        fit_groups = set(groups[n_dev_groups:])
        train_fit.extend(
            class_meta.index[class_meta["generation_group"].isin(fit_groups)].tolist()
        )
        inner_dev.extend(
            class_meta.index[class_meta["generation_group"].isin(dev_groups)].tolist()
        )

    tf = np.array(sorted(train_fit), dtype=int)
    iv = np.array(sorted(inner_dev), dtype=int)
    if set(tf) & set(iv):
        raise AssertionError("inner dev split overlap")
    return tf, iv


def ensure_experiment_split_manifest() -> dict[str, Any]:
    if EXPERIMENT_SPLIT_MANIFEST.exists():
        return load_frozen_manifest(EXPERIMENT_SPLIT_MANIFEST)

    dataset = load_dataset_v2(EXPERIMENT_DATASET_DIR)
    split = build_three_way_split(
        dataset.metadata,
        dataset.labels,
        seed=OUTER_SPLIT_SEED,
        cal_dev_seed=CAL_DEV_SEED,
    )
    manifest = build_frozen_manifest(
        split,
        dataset.metadata,
        dataset.labels,
        dataset_version=EXPERIMENT_DATASET_VERSION,
        generator_version=GENERATOR_VERSION,
    )
    manifest["status"] = "EXPERIMENT_SPLIT"
    manifest["note"] = (
        "Experiment-only split manifest. Does not modify frozen clean split."
    )
    save_frozen_manifest(manifest, EXPERIMENT_SPLIT_MANIFEST)
    return manifest


def load_clean_development_data() -> ExperimentData:
    """Load frozen clean dataset with frozen clean split (for error analysis only)."""
    from .config import FROZEN_CLEAN_MANIFEST

    manifest = load_frozen_manifest(FROZEN_CLEAN_MANIFEST)
    dataset = load_dataset_v2(CLEAN_DATASET_DIR)
    model_train_idx = np.array(manifest["model_train_idx"], dtype=int)
    cal_dev_idx = np.array(manifest["cal_dev_idx"], dtype=int)
    locked_test_idx = np.array(manifest["locked_test_idx"], dtype=int)
    train_fit_idx, inner_dev_idx = _split_inner_dev(dataset.metadata, model_train_idx)
    return ExperimentData(
        features=dataset.features,
        labels=dataset.labels,
        metadata=dataset.metadata,
        feature_names=list(dataset.feature_names),
        model_train_idx=model_train_idx,
        cal_dev_idx=cal_dev_idx,
        locked_test_idx=locked_test_idx,
        train_fit_idx=train_fit_idx,
        inner_dev_idx=inner_dev_idx,
        dataset_dir=str(CLEAN_DATASET_DIR),
        split_manifest_path=str(FROZEN_CLEAN_MANIFEST),
    )


def load_experiment_data(
    *,
    feature_names: list[str] | None = None,
    features: np.ndarray | None = None,
) -> ExperimentData:
    manifest = ensure_experiment_split_manifest()
    dataset = load_dataset_v2(EXPERIMENT_DATASET_DIR)
    model_train_idx = np.array(manifest["model_train_idx"], dtype=int)
    cal_dev_idx = np.array(manifest["cal_dev_idx"], dtype=int)
    locked_test_idx = np.array(manifest["locked_test_idx"], dtype=int)
    train_fit_idx, inner_dev_idx = _split_inner_dev(dataset.metadata, model_train_idx)

    names = feature_names or list(dataset.feature_names)
    X = features if features is not None else dataset.features
    if X.shape[1] != len(names):
        raise ValueError("Feature matrix width does not match feature_names")

    return ExperimentData(
        features=X,
        labels=dataset.labels,
        metadata=dataset.metadata,
        feature_names=names,
        model_train_idx=model_train_idx,
        cal_dev_idx=cal_dev_idx,
        locked_test_idx=locked_test_idx,
        train_fit_idx=train_fit_idx,
        inner_dev_idx=inner_dev_idx,
        dataset_dir=str(EXPERIMENT_DATASET_DIR),
        split_manifest_path=str(EXPERIMENT_SPLIT_MANIFEST),
    )


def split_report(data: ExperimentData) -> dict[str, Any]:
    class_counts = {}
    for part_name, idx in (
        ("model_train", data.model_train_idx),
        ("cal_dev", data.cal_dev_idx),
        ("locked_test", data.locked_test_idx),
    ):
        labels = data.labels[idx]
        class_counts[part_name] = {
            str(label): int((labels == label).sum())
            for label in sorted(np.unique(labels))
        }

    return {
        "dataset_dir": data.dataset_dir,
        "split_manifest": data.split_manifest_path,
        "model_train": int(len(data.model_train_idx)),
        "train_fit": int(len(data.train_fit_idx)),
        "inner_dev": int(len(data.inner_dev_idx)),
        "cal_dev": int(len(data.cal_dev_idx)),
        "locked_test": int(len(data.locked_test_idx)),
        "outer_split_seed": OUTER_SPLIT_SEED,
        "cal_dev_seed": CAL_DEV_SEED,
        "inner_dev_seed": INNER_DEV_SEED,
        "inner_dev_fraction": INNER_DEV_FRACTION,
        "class_counts": class_counts,
        "locked_test_off_limits_until_final_candidate": True,
    }
