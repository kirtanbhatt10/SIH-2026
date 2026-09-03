"""Shared fixtures for Phase 4 tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.phase4.config import FEATURE_NAMES


@pytest.fixture
def mini_dataset_dir(tmp_path: Path) -> Path:
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
                rows.append({
                    "sample_id": f"mini_{label}_{idx:03d}",
                    "label": label,
                    "signal_type": ["benign", "fsk", "ook", "chirp", "tone"][label],
                    "generation_group": group,
                })
                features.append(
                    np.linspace(label + 0.1, label + 1.0, len(FEATURE_NAMES)) + i * 0.01
                )
                labels.append(label)
                idx += 1

    root = tmp_path / "mini_dataset"
    root.mkdir()
    np.save(root / "features.npy", np.asarray(features, dtype=float))
    np.save(root / "labels.npy", np.asarray(labels, dtype=int))
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    info = {
        "dataset_version": "test_mini",
        "generator_version": "test",
        "seed": 42,
        "feature_names": FEATURE_NAMES,
    }
    with (root / "dataset_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh)
    return root


@pytest.fixture
def mini_phase3_artifacts(tmp_path: Path, mini_dataset_dir: Path):
    from ml.phase3.data import grouped_train_test_split, load_dataset_v2

    dataset = load_dataset_v2(mini_dataset_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X_train, y_train)

    model_path = tmp_path / "selected_model.joblib"
    meta_path = tmp_path / "selected_model_metadata.json"
    import joblib
    joblib.dump(model, model_path)
    metadata = {
        "model_name": "logistic_regression",
        "dataset_version": "test_mini",
        "generator_version": "test",
        "dataset_seed": 42,
        "feature_count": 32,
        "feature_names": FEATURE_NAMES,
        "class_mapping": {"0": "benign", "1": "fsk", "2": "ook", "3": "chirp", "4": "tone"},
        "split_method": "stratified_grouped_by_generation_group",
        "split_seed": 42,
        "train_size": int(len(split.train_idx)),
        "test_size": int(len(split.test_idx)),
        "predicted_classes_order": [0, 1, 2, 3, 4],
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh)
    return {
        "model_path": model_path,
        "metadata_path": meta_path,
        "dataset_dir": mini_dataset_dir,
        "split": split,
        "dataset": dataset,
    }
