"""Metrics, ranking, and candidate qualification for accuracy_82."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix

from ml.phase3.evaluation import evaluate_model

from .config import BENIGN_CLASS, CLASS_NAMES, TARGETS, THREAT_CLASSES


def evaluate_candidate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    partition: str,
    model_name: str,
) -> dict[str, Any]:
    base = evaluate_model(model, X, y, model_name=model_name)
    base["partition"] = partition
    base["benign_precision"] = float(base["per_class"]["benign"]["precision"])
    base["threat_precision"] = float(
        np.mean([base["per_class"][CLASS_NAMES[c]]["precision"] for c in THREAT_CLASSES])
    )
    base["latency_ms_per_sample"] = base["latency_ms_mean"]
    return base


def qualifies(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    ok = True
    checks = [
        ("accuracy", metrics["accuracy"], ">=", TARGETS["accuracy"]),
        ("macro_f1", metrics["macro_f1"], ">=", TARGETS["macro_f1"]),
        ("benign_fpr", metrics["benign_false_positive_rate"], "<=", TARGETS["benign_fpr"]),
        ("threat_recall", metrics["threat_recall"], ">=", TARGETS["threat_recall"]),
    ]
    for name, value, op, target in checks:
        passed = value >= target if op == ">=" else value <= target
        if not passed:
            ok = False
            reasons.append(f"{name}={value:.4f} failed target {op}{target:.4f}")
    return ok, reasons


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ranking: accuracy, macro F1, benign FPR (lower), threat recall."""
    return sorted(
        rows,
        key=lambda r: (
            r["accuracy"],
            r["macro_f1"],
            -r["benign_false_positive_rate"],
            r["threat_recall"],
        ),
        reverse=True,
    )


def security_healthy(metrics: dict[str, Any]) -> bool:
    return (
        metrics["macro_f1"] >= TARGETS["macro_f1"]
        and metrics["benign_false_positive_rate"] <= TARGETS["benign_fpr"]
        and metrics["threat_recall"] >= TARGETS["threat_recall"]
    )


def confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES.keys()))
    labels_names = [CLASS_NAMES[i] for i in CLASS_NAMES]
    pairs: list[dict[str, Any]] = []
    for true_i, true_name in enumerate(labels_names):
        for pred_i, pred_name in enumerate(labels_names):
            count = int(cm[true_i, pred_i])
            if count == 0 or true_name == pred_name:
                continue
            pairs.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": count,
                    "rate_within_true_class": count / max(int((y_true == true_i).sum()), 1),
                }
            )
    pairs.sort(key=lambda row: row["count"], reverse=True)
    return pairs
