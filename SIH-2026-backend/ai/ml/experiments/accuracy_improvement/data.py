"""Dataset and split loading for accuracy improvement experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.phase3.data import load_dataset_v2
from ml.promotion.splits import load_frozen_manifest

from .config import (
    DATASET_DIR,
    FROZEN_SPLIT_MANIFEST,
    INNER_DEV_FRACTION,
    INNER_DEV_SEED,
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

    @property
    def development_idx(self) -> np.ndarray:
        """Non-locked development indices (model_train + cal_dev)."""
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


def load_experiment_data(
    *,
    feature_names: list[str] | None = None,
) -> ExperimentData:
    manifest = load_frozen_manifest(FROZEN_SPLIT_MANIFEST)
    dataset = load_dataset_v2(DATASET_DIR)

    model_train_idx = np.array(manifest["model_train_idx"], dtype=int)
    cal_dev_idx = np.array(manifest["cal_dev_idx"], dtype=int)
    locked_test_idx = np.array(manifest["locked_test_idx"], dtype=int)
    train_fit_idx, inner_dev_idx = _split_inner_dev(
        dataset.metadata, model_train_idx
    )

    names = feature_names or list(dataset.feature_names)
    X = dataset.features
    if names != list(dataset.feature_names):
        raise ValueError("Extended features must be supplied via features.py")

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
    )


def split_report(data: ExperimentData) -> dict[str, Any]:
    return {
        "model_train": int(len(data.model_train_idx)),
        "train_fit": int(len(data.train_fit_idx)),
        "inner_dev": int(len(data.inner_dev_idx)),
        "cal_dev": int(len(data.cal_dev_idx)),
        "locked_test": int(len(data.locked_test_idx)),
        "inner_dev_seed": INNER_DEV_SEED,
        "inner_dev_fraction": INNER_DEV_FRACTION,
        "locked_test_off_limits": True,
    }
