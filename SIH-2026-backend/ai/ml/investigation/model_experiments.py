"""Controlled model experiments on grouped held-out split (dev selection only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.phase3.data import grouped_train_test_split, load_dataset_v2
from ml.phase3.evaluation import evaluate_model
from ml.phase3.models import create_baseline_models


def _make_models() -> dict[str, Any]:
    models = create_baseline_models()
    models["logistic_regression_balanced"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
    ])
    models["random_forest_balanced"] = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    models["linear_svm_balanced"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, random_state=42, class_weight="balanced")),
    ])
    return models


def run_experiments(*, output_path: Path) -> list[dict[str, Any]]:
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]

    rows = []
    for name, model in _make_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "benign_fpr": metrics["benign_false_positive_rate"],
                "benign_recall": metrics["benign_recall"],
                "threat_recall": metrics["threat_recall"],
                "latency_ms_per_sample": metrics["latency_ms_mean"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out model comparison experiments")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/investigation/model_experiments_report.json"),
    )
    args = parser.parse_args()
    rows = run_experiments(output_path=args.output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
