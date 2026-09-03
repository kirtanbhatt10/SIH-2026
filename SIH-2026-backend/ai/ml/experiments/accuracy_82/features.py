"""Deterministic engineered features derived from the 32-D DSP contract."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .config import BASE_FEATURE_NAMES, ENGINEERED_FEATURE_NAMES
from .data import ExperimentData


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / np.maximum(np.abs(den), eps)


def build_engineered_features(X32: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(BASE_FEATURE_NAMES)}
    ultrasonic = X32[:, idx["ultrasonic_energy"]]
    total = X32[:, idx["total_energy"]]
    centroid = X32[:, idx["spectral_centroid"]]
    bandwidth = X32[:, idx["spectral_bandwidth"]]
    flatness = X32[:, idx["spectral_flatness"]]
    rolloff = X32[:, idx["spectral_rolloff"]]
    rms = X32[:, idx["rms_amplitude"]]
    envelope = X32[:, idx["amplitude_envelope_std"]]
    peak_mag = X32[:, idx["peak_magnitude"]]
    duty = X32[:, idx["duty_cycle"]]
    bit_rate = X32[:, idx["bit_rate_estimate"]]

    engineered = np.column_stack(
        [
            np.log1p(np.maximum(ultrasonic, 0.0)),
            _safe_ratio(ultrasonic, total),
            _safe_ratio(peak_mag, rms),
            _safe_ratio(bandwidth, centroid + 1e-12),
            flatness * rolloff,
            _safe_ratio(ultrasonic, peak_mag + 1e-12),
            _safe_ratio(envelope, rms),
            np.abs(duty - bit_rate / np.maximum(bit_rate.max(), 1.0)),
        ]
    )
    if not np.isfinite(engineered).all():
        raise ValueError("Engineered features contain non-finite values")
    return engineered


def augment_features(X32: np.ndarray, *, use_engineered: bool) -> tuple[np.ndarray, list[str]]:
    if not use_engineered:
        return X32, list(BASE_FEATURE_NAMES)
    extra = build_engineered_features(X32)
    X = np.hstack([X32, extra])
    names = list(BASE_FEATURE_NAMES) + list(ENGINEERED_FEATURE_NAMES)
    return X, names


def analyze_features(data: ExperimentData, *, analysis_idx: np.ndarray) -> dict[str, Any]:
    X = data.features[analysis_idx]
    y = data.labels[analysis_idx]
    names = data.feature_names
    variances = X.var(axis=0)
    mi = mutual_info_classif(X, y, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
    rf.fit(X, y)
    importances = rf.feature_importances_
    ranked = sorted(
        [
            {
                "feature": names[i],
                "variance": float(variances[i]),
                "mutual_information": float(mi[i]),
                "rf_importance": float(importances[i]),
            }
            for i in range(len(names))
        ],
        key=lambda row: row["mutual_information"],
        reverse=True,
    )
    return {
        "n_features": len(names),
        "feature_names": names,
        "ranked_features": ranked,
        "notes": ["Feature analysis on experiment development indices only."],
    }
