"""Evaluation metrics and artifact writers for Phase 4."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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

from .calibrator import RiskCalibrator, RiskPrediction
from .config import (
    BENIGN_CLASS,
    CALIBRATION_METHOD,
    CLASS_NAMES,
    EXPECTED_CLASSES,
    SYNTHETIC_DATA_LIMITATION,
    THREAT_CLASSES,
)


def predictions_to_dataframe(
    predictions: list[RiskPrediction],
    y_true: np.ndarray | None = None,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, pred in enumerate(predictions):
        row = {
            "sample_id": sample_ids[i] if sample_ids is not None else f"sample_{i}",
            "true_label": int(y_true[i]) if y_true is not None else None,
            "true_class": CLASS_NAMES[int(y_true[i])] if y_true is not None else None,
            "predicted_class": pred.predicted_class,
            "predicted_label": pred.predicted_label,
            "class_probability": pred.class_probability,
            "confidence": pred.confidence,
            "threat_probability": pred.threat_probability,
            "threat_score": pred.threat_score,
            "calibrated_risk_score": pred.calibrated_risk_score,
            "risk_level": pred.risk_level,
        }
        for name, value in pred.class_probabilities.items():
            row[f"prob_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    calibrator: RiskCalibrator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    partition: str,
) -> dict[str, Any]:
    predictions = calibrator.predict_risk(X)
    y_pred = np.array([p.predicted_class for p in predictions], dtype=int)
    proba = calibrator.predict_proba(X)

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
    threat_true = y != BENIGN_CLASS
    threat_pred = y_pred != BENIGN_CLASS
    n_benign = int(benign_mask.sum())
    n_benign_as_threat = int((benign_mask & threat_pred).sum())
    benign_fpr = n_benign_as_threat / n_benign if n_benign else 0.0

    threat_recalls = [per_class[CLASS_NAMES[c]]["recall"] for c in THREAT_CLASSES]
    threat_precisions = [per_class[CLASS_NAMES[c]]["precision"] for c in THREAT_CLASSES]
    threat_recall = float(np.mean(threat_recalls))
    threat_precision = float(np.mean(threat_precisions))

    risk_scores = np.array([p.calibrated_risk_score for p in predictions])
    threat_probs = np.array([p.threat_probability for p in predictions])
    threat_labels = threat_true.astype(int)

    brier_threat = float(brier_score_loss(threat_labels, threat_probs))
    ece_threat = _expected_calibration_error(threat_labels, threat_probs)

    risk_levels = [p.risk_level for p in predictions]
    risk_distribution = {
        level: int(risk_levels.count(level)) for level in ("LOW", "MEDIUM", "HIGH")
    }

    thresholds = calibrator.get_thresholds()
    benign_risk = risk_scores[benign_mask]
    risk_band_fpr = {
        "high_band_benign_fpr": float(
            np.mean(benign_risk >= thresholds["high_threshold"])
        ) if n_benign else 0.0,
        "medium_plus_benign_fpr": float(
            np.mean(benign_risk >= thresholds["medium_threshold"])
        ) if n_benign else 0.0,
    }

    return {
        "partition": partition,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "benign_false_positive_rate": float(benign_fpr),
        "threat_recall": threat_recall,
        "threat_precision": threat_precision,
        "threat_recall_definition": (
            "Macro-average recall over threat classes (fsk, ook, chirp, tone)."
        ),
        "threat_precision_definition": (
            "Macro-average precision over threat classes (fsk, ook, chirp, tone)."
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in labels],
        "brier_score_threat": brier_threat,
        "expected_calibration_error_threat": ece_threat,
        "risk_distribution": risk_distribution,
        "risk_band_benign_fpr": risk_band_fpr,
        "risk_thresholds": thresholds,
        "probability_checks": _probability_checks(proba, risk_scores, threat_probs),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        frac_pos = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()
        ece += mask.sum() / n * abs(frac_pos - mean_pred)
    return float(ece)


def _probability_checks(
    proba: np.ndarray,
    risk_scores: np.ndarray,
    threat_probs: np.ndarray,
) -> dict[str, Any]:
    return {
        "nan_count": int(np.isnan(proba).sum() + np.isnan(risk_scores).sum()),
        "inf_count": int(np.isinf(proba).sum() + np.isinf(risk_scores).sum()),
        "proba_in_unit_interval": bool(
            (proba >= 0).all() and (proba <= 1).all()
            and (risk_scores >= 0).all() and (risk_scores <= 1).all()
            and (threat_probs >= 0).all() and (threat_probs <= 1).all()
        ),
        "proba_sum_close_to_one": bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)),
    }


def save_phase4_artifacts(
    *,
    output_dir: Path,
    calibrator: RiskCalibrator,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    metadata: dict[str, Any],
    split_report: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    calibration_metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "calibration_method": CALIBRATION_METHOD,
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
    }
    metrics_path = output_dir / "calibration_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(calibration_metrics, fh, indent=2)
    paths["calibration_metrics"] = str(metrics_path)

    report = {
        "phase": 4,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_description": (
            "Frozen Phase 3 logistic_regression probabilities are calibrated with "
            "isotonic regression (cv=prefit) on the train partition. Threat scores "
            "use an additional isotonic map from raw threat probability to empirical "
            "threat prevalence. LOW/MEDIUM/HIGH thresholds are fit data-driven from "
            "benign calibration scores targeting controlled benign FPR budgets."
        ),
        "confidence_definition": "Confidence equals the calibrated probability of the predicted class.",
        "threat_score_definition": (
            "threat_score is the raw (uncalibrated) probability mass on threat classes "
            "from the frozen classifier. threat_probability and calibrated_risk_score "
            "apply isotonic calibration."
        ),
        "split_report": split_report,
        "train_metrics_summary": _summarize_metrics(train_metrics),
        "test_metrics_summary": _summarize_metrics(test_metrics),
        "risk_thresholds": calibrator.get_thresholds(),
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "calibration_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    paths["calibration_report"] = str(report_path)

    risk_dist = pd.DataFrame([
        {"partition": "train", **train_metrics["risk_distribution"]},
        {"partition": "test", **test_metrics["risk_distribution"]},
    ])
    risk_dist_path = output_dir / "risk_distribution.csv"
    risk_dist.to_csv(risk_dist_path, index=False)
    paths["risk_distribution"] = str(risk_dist_path)

    preds = pd.concat(
        [
            train_predictions.assign(partition="train"),
            test_predictions.assign(partition="test"),
        ],
        ignore_index=True,
    )
    preds_path = output_dir / "calibrated_predictions.csv"
    preds.to_csv(preds_path, index=False)
    paths["calibrated_predictions"] = str(preds_path)

    cal_meta = {
        "phase3_model_name": metadata.get("model_name"),
        "phase3_model_path": str(metadata.get("_model_path", "")),
        "dataset_version": metadata.get("dataset_version"),
        "generator_version": metadata.get("generator_version"),
        "feature_count": metadata.get("feature_count"),
        "feature_names": metadata.get("feature_names"),
        "class_mapping": metadata.get("class_mapping"),
        "split_method": split_report.get("split_method"),
        "split_seed": split_report.get("split_seed"),
        "calibration_method": CALIBRATION_METHOD,
        "risk_thresholds": calibrator.get_thresholds(),
        "confidence_definition": report["confidence_definition"],
        "threat_score_definition": report["threat_score_definition"],
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
        "timestamp_utc": report["timestamp_utc"],
    }
    meta_path = output_dir / "calibration_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(cal_meta, fh, indent=2)
    paths["calibration_metadata"] = str(meta_path)

    calibrator_path = output_dir / "risk_calibrator.joblib"
    calibrator.save(calibrator_path)
    paths["risk_calibrator"] = str(calibrator_path)

    return paths


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "benign_false_positive_rate": metrics["benign_false_positive_rate"],
        "threat_recall": metrics["threat_recall"],
        "threat_precision": metrics["threat_precision"],
        "brier_score_threat": metrics["brier_score_threat"],
        "expected_calibration_error_threat": metrics["expected_calibration_error_threat"],
        "risk_distribution": metrics["risk_distribution"],
    }
