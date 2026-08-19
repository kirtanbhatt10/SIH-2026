"""Shared fixtures for Phase 3 tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.phase3.config import FEATURE_NAMES


@pytest.fixture
def mini_dataset_dir(tmp_path: Path) -> Path:
    """Small controlled dataset with known generation groups."""
    n_per_class = 10
    groups = {
        0: ["benign|g1", "benign|g2"],
        1: ["fsk|g1", "fsk|g2"],
        2: ["ook|g1", "ook|g2"],
        3: ["chirp|g1", "chirp|g2"],
        4: ["tone|g1", "tone|g2"],
    }
    rows = []
    features = []
    labels = []
    idx = 0
    for label, group_list in groups.items():
        per_group = n_per_class // len(group_list)
        for group in group_list:
            for i in range(per_group):
                sample_id = f"mini_{label}_{idx:03d}"
                rows.append({
                    "sample_id": sample_id,
                    "label": label,
                    "signal_type": ["benign", "fsk", "ook", "chirp", "tone"][label],
                    "generation_group": group,
                })
                features.append(np.linspace(label + 0.1, label + 1.0, len(FEATURE_NAMES)) + i * 0.01)
                labels.append(label)
                idx += 1

    meta = pd.DataFrame(rows)
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    root = tmp_path / "mini_dataset"
    root.mkdir()
    np.save(root / "features.npy", X)
    np.save(root / "labels.npy", y)
    meta.to_csv(root / "metadata.csv", index=False)
    info = {
        "dataset_version": "test_mini",
        "generator_version": "test",
        "seed": 42,
        "feature_names": FEATURE_NAMES,
    }
    with (root / "dataset_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh)
    return root
