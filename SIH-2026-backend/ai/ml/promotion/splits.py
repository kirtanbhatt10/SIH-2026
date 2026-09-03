"""Three-way grouped split: model train / calibration dev / locked test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.phase3.data import SplitResult, build_split_report, grouped_train_test_split
from ml.promotion.config_clean import (
    CAL_DEV_FRACTION,
    CAL_DEV_SEED,
    DATASET_VERSION,
    FROZEN_SPLIT_MANIFEST_PATH,
    GENERATOR_VERSION,
    SPLIT_SEED,
    TEST_FRACTION,
)


@dataclass(frozen=True)
class ThreeWaySplit:
    model_train_idx: np.ndarray
    cal_dev_idx: np.ndarray
    locked_test_idx: np.ndarray
    outer_split: SplitResult
    cal_dev_seed: int
    cal_dev_fraction: float
    model_train_groups: set[str]
    cal_dev_groups: set[str]
    locked_test_groups: set[str]

    def assert_no_leakage(self, metadata: pd.DataFrame) -> None:
        mt = set(metadata.iloc[self.model_train_idx]["generation_group"])
        cd = set(metadata.iloc[self.cal_dev_idx]["generation_group"])
        lt = set(metadata.iloc[self.locked_test_idx]["generation_group"])
        if mt & lt or cd & lt or mt & cd:
            raise AssertionError(
                f"Partition leakage: mt∩lt={mt & lt}, cd∩lt={cd & lt}, mt∩cd={mt & cd}"
            )


def split_train_calibration_dev(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    cal_fraction: float = CAL_DEV_FRACTION,
    seed: int = CAL_DEV_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out calibration-dev groups from the train partition only."""
    train_meta = metadata.iloc[train_idx].copy()
    rng = np.random.RandomState(seed)

    model_train_idx: list[int] = []
    cal_dev_idx: list[int] = []

    for label in sorted(train_meta["label"].unique()):
        class_meta = train_meta[train_meta["label"] == label]
        groups = class_meta["generation_group"].unique().copy()
        rng.shuffle(groups)
        n_cal_groups = max(1, int(round(len(groups) * cal_fraction)))
        cal_groups = set(groups[:n_cal_groups])
        fit_groups = set(groups[n_cal_groups:])

        model_train_idx.extend(
            class_meta.index[class_meta["generation_group"].isin(fit_groups)].tolist()
        )
        cal_dev_idx.extend(
            class_meta.index[class_meta["generation_group"].isin(cal_groups)].tolist()
        )

    mt = np.array(sorted(model_train_idx), dtype=int)
    cd = np.array(sorted(cal_dev_idx), dtype=int)
    overlap = set(mt) & set(cd)
    if overlap:
        raise AssertionError(f"model_train and cal_dev index overlap: {sorted(overlap)[:5]}")
    return mt, cd


def build_three_way_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    cal_dev_seed: int = CAL_DEV_SEED,
    cal_dev_fraction: float = CAL_DEV_FRACTION,
) -> ThreeWaySplit:
    outer = grouped_train_test_split(
        metadata, labels, test_fraction=test_fraction, seed=seed
    )
    model_train_idx, cal_dev_idx = split_train_calibration_dev(
        metadata,
        outer.train_idx,
        cal_fraction=cal_dev_fraction,
        seed=cal_dev_seed,
    )
    split = ThreeWaySplit(
        model_train_idx=model_train_idx,
        cal_dev_idx=cal_dev_idx,
        locked_test_idx=outer.test_idx,
        outer_split=outer,
        cal_dev_seed=cal_dev_seed,
        cal_dev_fraction=cal_dev_fraction,
        model_train_groups=set(metadata.iloc[model_train_idx]["generation_group"]),
        cal_dev_groups=set(metadata.iloc[cal_dev_idx]["generation_group"]),
        locked_test_groups=set(metadata.iloc[outer.test_idx]["generation_group"]),
    )
    split.assert_no_leakage(metadata)
    return split


def build_frozen_manifest(
    split: ThreeWaySplit,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    dataset_version: str,
    generator_version: str,
) -> dict[str, Any]:
    outer_report = build_split_report(split.outer_split, metadata, labels)
    return {
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "generator_version": generator_version,
        "split_policy": {
            "outer_method": split.outer_split.method,
            "outer_seed": split.outer_split.seed,
            "test_fraction": split.outer_split.test_fraction,
            "cal_dev_seed": split.cal_dev_seed,
            "cal_dev_fraction": split.cal_dev_fraction,
            "generation_group_policy": (
                "Assign entire generation_group to exactly one partition "
                "(model_train, cal_dev, or locked_test). Never split a group."
            ),
        },
        "partition_sizes": {
            "model_train": int(len(split.model_train_idx)),
            "cal_dev": int(len(split.cal_dev_idx)),
            "locked_test": int(len(split.locked_test_idx)),
        },
        "outer_split_report": outer_report,
        "model_train_groups": sorted(split.model_train_groups),
        "cal_dev_groups": sorted(split.cal_dev_groups),
        "locked_test_groups": sorted(split.locked_test_groups),
        "model_train_idx": split.model_train_idx.tolist(),
        "cal_dev_idx": split.cal_dev_idx.tolist(),
        "locked_test_idx": split.locked_test_idx.tolist(),
        "model_train_sample_ids": metadata.iloc[split.model_train_idx]["sample_id"].tolist(),
        "cal_dev_sample_ids": metadata.iloc[split.cal_dev_idx]["sample_id"].tolist(),
        "locked_test_sample_ids": metadata.iloc[split.locked_test_idx]["sample_id"].tolist(),
        "leakage_checks": {
            "group_overlap_model_train_locked_test": len(
                split.model_train_groups & split.locked_test_groups
            ),
            "group_overlap_cal_dev_locked_test": len(
                split.cal_dev_groups & split.locked_test_groups
            ),
            "group_overlap_model_train_cal_dev": len(
                split.model_train_groups & split.cal_dev_groups
            ),
            "passed": True,
        },
    }


def save_frozen_manifest(manifest: dict[str, Any], path: Path | None = None) -> Path:
    out = Path(path) if path is not None else FROZEN_SPLIT_MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def load_frozen_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else FROZEN_SPLIT_MANIFEST_PATH
    with manifest_path.open(encoding="utf-8") as fh:
        return json.load(fh)
