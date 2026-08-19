"""Evaluation utilities for the Phase 5 inference service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import BENIGN_CLASS, CLASS_NAMES, EXPECTED_CLASSES, SYNTHETIC_DATA_LIMITATION, THREAT_CLASSES
from .service import InferenceService


def evaluate_service(
    service: InferenceService,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    predictions = service.predict_batch(X)
    y_pred = np.array([p["predicted_class_id"] for p in predictions], dtype=int)

    labels = list(EXPECTED_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=labels)

    per_class = {}
    for i, label in enumerate(labels):
        name = CLASS_NAMES[label]
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    benign_mask = y == BENIGN_CLASS
    threat_pred = y_pred != BENIGN_CLASS
    n_benign = int(benign_mask.sum())
    benign_fpr = float((benign_mask & threat_pred).sum() / n_benign) if n_benign else 0.0

    threat_recall = float(np.mean([per_class[CLASS_NAMES[c]]["recall"] for c in THREAT_CLASSES]))
    threat_precision = float(np.mean([per_class[CLASS_NAMES[c]]["precision"] for c in THREAT_CLASSES]))

    risk_levels = [p["risk_level"] for p in predictions]
    risk_distribution = {level: int(risk_levels.count(level)) for level in ("LOW", "MEDIUM", "HIGH")}

    latency = service.benchmark_latency(X)

    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "benign_false_positive_rate": benign_fpr,
        "threat_recall": threat_recall,
        "threat_precision": threat_precision,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in labels],
        "risk_distribution": risk_distribution,
        "risk_thresholds": service.get_risk_thresholds(),
        "latency_ms_mean": latency["latency_ms_mean"],
        "latency_ms_std": latency["latency_ms_std"],
        "latency_ms_per_sample": latency["latency_ms_per_sample"],
        "predictions": predictions,
        "sample_ids": sample_ids,
    }


def save_phase5_report(
    *,
    output_dir: Path,
    service: InferenceService,
    dataset_metrics: dict[str, Any],
    phase3_metadata: dict[str, Any],
    phase4_metadata: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    report = {
        "phase": 5,
        "description": "Frozen inference/service evaluation on Dataset V2",
        "phase3_model": phase3_metadata.get("model_name"),
        "phase4_calibration_method": phase4_metadata.get("calibration_method"),
        "risk_thresholds": service.get_risk_thresholds(),
        "metrics": {
            k: dataset_metrics[k]
            for k in (
                "n_samples", "accuracy", "macro_f1", "weighted_f1",
                "benign_false_positive_rate", "threat_recall", "threat_precision",
                "risk_distribution", "latency_ms_mean", "latency_ms_per_sample",
            )
        },
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "inference_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    paths["inference_report"] = str(report_path)

    metrics_path = output_dir / "inference_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_evaluation": {
                    k: v for k, v in dataset_metrics.items() if k != "predictions"
                },
                "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
            },
            fh,
            indent=2,
        )
    paths["inference_metrics"] = str(metrics_path)

    preds = dataset_metrics["predictions"]
    sample_ids = dataset_metrics.get("sample_ids") or [f"s{i}" for i in range(len(preds))]
    rows = []
    for sid, pred in zip(sample_ids, preds):
        row = {"sample_id": sid, **pred}
        rows.append(row)
    preds_path = output_dir / "inference_predictions.csv"
    pd.DataFrame(rows).to_csv(preds_path, index=False)
    paths["inference_predictions"] = str(preds_path)

    service_path = output_dir / "inference_service.joblib"
    service.save(service_path)
    paths["inference_service"] = str(service_path)

    meta_path = output_dir / "inference_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "phase3_model_name": phase3_metadata.get("model_name"),
                "phase4_calibration_method": phase4_metadata.get("calibration_method"),
                "feature_count": len(service.feature_names),
                "feature_names": service.feature_names,
                "class_mapping": phase3_metadata.get("class_mapping"),
                "risk_thresholds": service.get_risk_thresholds(),
                "no_training": True,
                "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
                "timestamp_utc": report["timestamp_utc"],
            },
            fh,
            indent=2,
        )
    paths["inference_metadata"] = str(meta_path)

    return paths
