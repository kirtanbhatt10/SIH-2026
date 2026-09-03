"""Metrics, error analysis, and candidate qualification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ml.phase3.evaluation import evaluate_model, measure_latency_ms

from .config import BENIGN_CLASS, CLASS_NAMES, TARGETS, THREAT_CLASSES


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate_candidate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    partition: str,
    model_name: str,
) -> dict[str, Any]:
    base = evaluate_model(model, X, y, model_name=model_name)
    probs = model.predict_proba(X)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["clf"].classes_)
    benign_idx = classes.index(BENIGN_CLASS)
    threat_true = (y != BENIGN_CLASS).astype(int)
    threat_prob = 1.0 - probs[:, benign_idx]
    base["partition"] = partition
    base["benign_precision"] = float(base["per_class"]["benign"]["precision"])
    base["threat_precision"] = float(
        np.mean([base["per_class"][CLASS_NAMES[c]]["precision"] for c in THREAT_CLASSES])
    )
    base["brier_score_threat"] = float(brier_score_loss(threat_true, np.clip(threat_prob, 0, 1)))
    base["expected_calibration_error_threat"] = _ece(threat_true, np.clip(threat_prob, 0, 1))
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
    return sorted(
        rows,
        key=lambda r: (
            r["macro_f1"],
            -r["benign_false_positive_rate"],
            r["threat_recall"],
            r["accuracy"],
            -r["latency_ms_per_sample"],
        ),
        reverse=True,
    )


def error_analysis(
    model: Any,
    data_features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    eval_idx: np.ndarray,
    *,
    partition: str,
) -> dict[str, Any]:
    X = data_features[eval_idx]
    y = labels[eval_idx]
    meta = metadata.iloc[eval_idx].copy()
    y_pred = model.predict(X)
    metrics = evaluate_candidate(model, X, y, partition=partition, model_name="baseline")

    cm = confusion_matrix(y, y_pred, labels=list(CLASS_NAMES.keys()))
    labels_names = [CLASS_NAMES[i] for i in CLASS_NAMES]

    confusion_pairs: list[dict[str, Any]] = []
    for true_i, true_name in enumerate(labels_names):
        for pred_i, pred_name in enumerate(labels_names):
            count = int(cm[true_i, pred_i])
            if count == 0 or true_name == pred_name:
                continue
            confusion_pairs.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": count,
                    "rate_within_true_class": count / max(int((y == true_i).sum()), 1),
                }
            )
    confusion_pairs.sort(key=lambda row: row["count"], reverse=True)

    meta = meta.copy()
    meta["y_true"] = y
    meta["y_pred"] = y_pred
    meta["correct"] = y == y_pred
    meta["benign_false_alarm"] = (meta["y_true"] == BENIGN_CLASS) & (
        np.isin(meta["y_pred"], list(THREAT_CLASSES))
    )
    meta["threat_miss"] = np.isin(meta["y_true"], list(THREAT_CLASSES)) & (
        meta["y_pred"] == BENIGN_CLASS
    )

    def _group_stats(frame: pd.DataFrame, col: str) -> dict[str, Any]:
        if col not in frame.columns or frame[col].isna().all():
            return {}
        grouped = frame.groupby(col)["correct"].agg(["mean", "count"])
        return {
            str(k): {"accuracy": float(v["mean"]), "count": int(v["count"])}
            for k, v in grouped.iterrows()
        }

    correlation_fields = {
        "snr": _group_stats(meta, "snr"),
        "noise_level": _group_stats(meta, "noise_level"),
        "amplitude": _group_stats(meta.dropna(subset=["amplitude"]), "amplitude"),
        "signal_type": _group_stats(meta, "signal_type"),
        "benign_variant": _group_stats(
            meta[meta["signal_type"] == "benign"], "benign_variant"
        ),
    }

    return {
        "partition": partition,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels_names,
        "dominant_confusion_pairs": confusion_pairs[:15],
        "benign_false_positive_count": int(meta["benign_false_alarm"].sum()),
        "threat_false_negative_count": int(meta["threat_miss"].sum()),
        "error_correlates": correlation_fields,
    }
