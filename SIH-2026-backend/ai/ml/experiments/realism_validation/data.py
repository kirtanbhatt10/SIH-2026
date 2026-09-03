"""Load clean development data only — locked test is never accessed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.phase3.data import load_dataset_v2
from ml.promotion.splits import load_frozen_manifest

from .config import CLEAN_DATASET_DIR, FEATURE_NAMES, FROZEN_SPLIT_MANIFEST


@dataclass(frozen=True)
class AuditData:
    features: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    model_train_idx: np.ndarray
    cal_dev_idx: np.ndarray
    development_idx: np.ndarray
    dataset_dir: str

    def mask_class(self, name: str) -> np.ndarray:
        cid = {"benign": 0, "fsk": 1, "ook": 2, "chirp": 3, "tone": 4}[name]
        return self.labels == cid

    def dev_features(self) -> np.ndarray:
        return self.features[self.development_idx]

    def dev_labels(self) -> np.ndarray:
        return self.labels[self.development_idx]

    def dev_metadata(self) -> pd.DataFrame:
        return self.metadata.iloc[self.development_idx].copy()


def load_audit_data() -> AuditData:
    manifest = load_frozen_manifest(FROZEN_SPLIT_MANIFEST)
    dataset = load_dataset_v2(CLEAN_DATASET_DIR)

    model_train_idx = np.array(manifest["model_train_idx"], dtype=int)
    cal_dev_idx = np.array(manifest["cal_dev_idx"], dtype=int)
    development_idx = np.concatenate([model_train_idx, cal_dev_idx])

    if list(dataset.feature_names) != FEATURE_NAMES:
        raise ValueError("Feature contract mismatch with Phase-1 FEATURE_NAMES")

    return AuditData(
        features=dataset.features,
        labels=dataset.labels,
        metadata=dataset.metadata,
        feature_names=list(FEATURE_NAMES),
        model_train_idx=model_train_idx,
        cal_dev_idx=cal_dev_idx,
        development_idx=development_idx,
        dataset_dir=str(CLEAN_DATASET_DIR),
    )


def audit_scope_report(data: AuditData) -> dict[str, Any]:
    return {
        "dataset": data.dataset_dir,
        "partitions_used": ["model_train", "cal_dev"],
        "locked_test_accessed": False,
        "n_development": int(len(data.development_idx)),
        "n_model_train": int(len(data.model_train_idx)),
        "n_cal_dev": int(len(data.cal_dev_idx)),
        "class_counts_dev": {
            str(k): int((data.labels[data.development_idx] == k).sum())
            for k in sorted(np.unique(data.labels))
        },
    }
