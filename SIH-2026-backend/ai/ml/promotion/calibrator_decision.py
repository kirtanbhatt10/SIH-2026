"""Calibration method selection on the calibration-dev partition only."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ml.phase4.calibrator import RiskCalibrator
from ml.phase4.evaluation import evaluate
from ml.promotion.uncalibrated_adapter import UncalibratedRiskAdapter


CalibrationChoice = Literal["isotonic", "none"]


def _attach_benign_recall(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    out["benign_recall"] = float(metrics["per_class"]["benign"]["recall"])
    return out


def evaluate_calibrated(
    model: Any,
    metadata: dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[RiskCalibrator, dict[str, Any]]:
    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(X_cal, y_cal)
    metrics = _attach_benign_recall(
        evaluate(calibrator, X_cal, y_cal, partition="cal_dev")
    )
    return calibrator, metrics


def evaluate_uncalibrated(
    model: Any,
    metadata: dict[str, Any],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[UncalibratedRiskAdapter, dict[str, Any]]:
    adapter = UncalibratedRiskAdapter(model, metadata)
    adapter.calibrate(X_cal, y_cal)
    metrics = _attach_benign_recall(
        evaluate(adapter, X_cal, y_cal, partition="cal_dev")
    )
    return adapter, metrics


def choose_calibration_method(
    uncal_metrics: dict[str, Any],
    cal_metrics: dict[str, Any],
) -> tuple[CalibrationChoice, str]:
    """Pick calibration only when probability quality and detection both improve."""
    u_brier = uncal_metrics["brier_score_threat"]
    c_brier = cal_metrics["brier_score_threat"]
    u_ece = uncal_metrics["expected_calibration_error_threat"]
    c_ece = cal_metrics["expected_calibration_error_threat"]
    u_fpr = uncal_metrics["benign_false_positive_rate"]
    c_fpr = cal_metrics["benign_false_positive_rate"]
    u_recall = uncal_metrics["threat_recall"]
    c_recall = cal_metrics["threat_recall"]
    u_f1 = uncal_metrics["macro_f1"]
    c_f1 = cal_metrics["macro_f1"]

    prob_improved = (c_brier <= u_brier + 1e-6) and (c_ece <= u_ece + 0.005)
    prob_strictly_better = (c_brier < u_brier - 1e-6) or (c_ece < u_ece - 0.005)

    detection_not_worse = (
        c_f1 >= u_f1 - 0.005
        and c_fpr <= u_fpr + 0.01
        and c_recall >= u_recall - 0.01
    )
    detection_improved = (
        c_f1 > u_f1 + 0.005
        or c_fpr < u_fpr - 0.005
        or c_recall > u_recall + 0.005
    )

    if prob_improved and prob_strictly_better and detection_not_worse and detection_improved:
        reason = (
            f"Isotonic calibration selected on cal_dev: Brier {u_brier:.4f}->{c_brier:.4f}, "
            f"ECE {u_ece:.4f}->{c_ece:.4f}, macro F1 {u_f1:.4f}->{c_f1:.4f}, "
            f"benign FPR {u_fpr:.4f}->{c_fpr:.4f}, threat recall {u_recall:.4f}->{c_recall:.4f}."
        )
        return "isotonic", reason

    reason = (
        "Uncalibrated model retained: calibration did not improve both probability quality "
        f"(Brier {u_brier:.4f} vs {c_brier:.4f}, ECE {u_ece:.4f} vs {c_ece:.4f}) and "
        f"detection behavior (macro F1 {u_f1:.4f} vs {c_f1:.4f}, benign FPR {u_fpr:.4f} vs "
        f"{c_fpr:.4f}, threat recall {u_recall:.4f} vs {c_recall:.4f}) on cal_dev."
    )
    return "none", reason


def audit_prior_calibration_experiment(
    *,
    train_threat_p: np.ndarray,
    y_train_threat: np.ndarray,
    test_threat_p: np.ndarray,
    y_test_threat: np.ndarray,
) -> dict[str, Any]:
    """Document whether the prior 7.5% / 95.4% numbers used test-set calibrator fit."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_threat_p, y_train_threat)
    cal_test = iso.predict(test_threat_p)
    cal_pred = np.where(cal_test >= 0.5, 1, 0)

    benign_mask = y_test_threat == 0
    threat_mask = y_test_threat == 1
    benign_fpr = float((cal_pred[benign_mask] == 1).mean()) if benign_mask.any() else 0.0
    threat_recall = float(cal_pred[threat_mask].mean()) if threat_mask.any() else 0.0

    return {
        "method": "prior_clean_rebuild_pipeline_isotonic_threat_binary",
        "calibrator_fit_partition": "train",
        "evaluation_partition": "test",
        "test_set_used_for_calibrator_fit": False,
        "valid_for_final_promotion": False,
        "invalidation_reasons": [
            "Simplified binary isotonic on summed threat probability — not Phase 4 RiskCalibrator.",
            "Model selection and calibration comparison used locked test metrics before this gate.",
            "Final promotion requires cal_dev selection and one locked-test evaluation only.",
        ],
        "reproduced_test_metrics": {
            "benign_fpr": benign_fpr,
            "threat_recall": threat_recall,
            "brier": float(brier_score_loss(y_test_threat, np.clip(cal_test, 0, 1))),
        },
        "verdict": (
            "Calibrator was NOT fit on the test set, but the 7.5%/95.4% headline numbers "
            "are INVALID for promotion because test-set metrics were used for selection "
            "and the methodology differs from the Phase 4 RiskCalibrator path."
        ),
    }
