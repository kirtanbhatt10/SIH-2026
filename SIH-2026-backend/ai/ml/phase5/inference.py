"""Input validation and inference helpers for Phase 5."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import FEATURE_NAMES, N_FEATURES, PROB_SUM_TOLERANCE


class InputValidationError(ValueError):
    """Raised when inference input fails validation."""


def validate_and_prepare_features(
    features: Any,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Validate and convert a single 32-feature sample to float64 ndarray."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 1:
        raise InputValidationError(
            f"Single-sample input must be 1-D, got shape {arr.shape}"
        )
    if arr.shape[0] != N_FEATURES:
        raise InputValidationError(
            f"Expected exactly {N_FEATURES} features, got {arr.shape[0]}"
        )
    _check_finite(arr, "single sample")
    if feature_names is not None and len(feature_names) != N_FEATURES:
        raise InputValidationError("feature_names length must be 32")
    return arr


def validate_and_prepare_batch(
    features: Any,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Validate and convert an (N, 32) batch to float64 ndarray."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise InputValidationError(
            f"Batch input must be 2-D (N, {N_FEATURES}), got shape {arr.shape}"
        )
    if arr.shape[1] != N_FEATURES:
        raise InputValidationError(
            f"Expected feature dimension {N_FEATURES}, got {arr.shape[1]}"
        )
    _check_finite(arr, "batch")
    if feature_names is not None and list(feature_names) != list(FEATURE_NAMES):
        raise InputValidationError("feature_names do not match Phase-1 contract order")
    return arr


def _check_finite(arr: np.ndarray, label: str) -> None:
    if np.isnan(arr).any():
        raise InputValidationError(f"{label} contains NaN values")
    if np.isinf(arr).any():
        raise InputValidationError(f"{label} contains Inf values")


def validate_probability_matrix(proba: np.ndarray) -> None:
    if proba.ndim != 2:
        raise ValueError(f"Probability matrix must be 2-D, got shape {proba.shape}")
    if np.isnan(proba).any() or np.isinf(proba).any():
        raise ValueError("Probabilities contain NaN or Inf")
    if (proba < 0).any() or (proba > 1).any():
        raise ValueError("Probabilities outside [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=PROB_SUM_TOLERANCE):
        raise ValueError("Probabilities do not sum to 1")


def risk_prediction_to_dict(
    pred,
    *,
    model_calibration: str = "isotonic",
) -> dict[str, Any]:
    """Convert a risk prediction to the Phase 5 API dict.

    For ``model_calibration='uncalibrated'``, all probabilities are raw model
    outputs. No field implies isotonic or Phase 4 calibration.
    """
    result: dict[str, Any] = {
        "predicted_class_id": pred.predicted_class,
        "predicted_class": pred.predicted_label,
        "confidence": pred.confidence,
        "class_probabilities": dict(pred.class_probabilities),
        "threat_score": pred.threat_score,
        "risk_level": pred.risk_level,
        "model_calibration": model_calibration,
    }
    if model_calibration == "uncalibrated":
        return result

    result["threat_probability"] = pred.threat_probability
    result["calibrated_risk_score"] = pred.calibrated_risk_score
    return result
