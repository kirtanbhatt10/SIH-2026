"""Uncalibrated risk adapter with the same interface as RiskCalibrator."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ml.phase4.calibrator import RiskPrediction
from ml.phase4.config import (
    BENIGN_CLASS,
    CLASS_NAMES,
    TARGET_HIGH_BENIGN_FPR,
    TARGET_MEDIUM_PLUS_BENIGN_FPR,
    THREAT_CLASSES,
)


class UncalibratedRiskAdapter:
    """Map raw classifier probabilities to risk levels without isotonic calibration."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self._frozen_model = model
        self.metadata = metadata
        self.class_order_ = self._resolve_class_order()
        self.benign_index_ = self.class_order_.index(BENIGN_CLASS)
        self.high_threshold_: float | None = None
        self.medium_threshold_: float | None = None
        self._is_fitted = False

    def _resolve_class_order(self) -> list[int]:
        order = self.metadata.get("predicted_classes_order")
        if order is not None:
            return [int(c) for c in order]
        if hasattr(self._frozen_model, "classes_"):
            return [int(c) for c in self._frozen_model.classes_]
        if hasattr(self._frozen_model, "named_steps"):
            return [int(c) for c in self._frozen_model.named_steps["clf"].classes_]
        return list(CLASS_NAMES.keys())

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "UncalibratedRiskAdapter":
        """Fit risk thresholds on calibration data only (no probability remapping)."""
        raw_proba = self._frozen_model.predict_proba(X_cal)
        raw_threat = self._threat_probability_from_proba(raw_proba)
        self.high_threshold_, self.medium_threshold_ = self._fit_risk_thresholds(
            raw_threat, y_cal
        )
        self._is_fitted = True
        return self

    def _fit_risk_thresholds(
        self,
        risk_scores: np.ndarray,
        y_cal: np.ndarray,
    ) -> tuple[float, float]:
        benign_mask = y_cal == BENIGN_CLASS
        benign_scores = np.sort(risk_scores[benign_mask])
        n = len(benign_scores)
        if n == 0:
            return 0.75, 0.45

        high_idx = self._threshold_index(benign_scores, TARGET_HIGH_BENIGN_FPR)
        medium_idx = self._threshold_index(
            benign_scores, TARGET_MEDIUM_PLUS_BENIGN_FPR
        )
        high_threshold = float(benign_scores[high_idx])
        medium_threshold = float(benign_scores[medium_idx])
        if medium_threshold > high_threshold:
            medium_threshold = high_threshold
        return high_threshold, medium_threshold

    @staticmethod
    def _threshold_index(sorted_benign_scores: np.ndarray, target_fpr: float) -> int:
        n = len(sorted_benign_scores)
        for idx in range(n):
            fpr = (n - idx) / n
            if fpr <= target_fpr:
                return idx
        return 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        proba = self._frozen_model.predict_proba(X)
        if np.isnan(proba).any() or np.isinf(proba).any():
            raise ValueError("Probabilities contain NaN or Inf")
        return proba

    def predict_risk(self, X: np.ndarray) -> list[RiskPrediction]:
        self._check_fitted()
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        raw_threat = self._threat_probability_from_proba(proba)

        results: list[RiskPrediction] = []
        for i in range(X.shape[0]):
            pred_class = int(self.class_order_[pred_indices[i]])
            class_prob = float(proba[i, pred_indices[i]])
            threat_prob = float(raw_threat[i])
            risk_score = threat_prob
            results.append(
                RiskPrediction(
                    predicted_class=pred_class,
                    predicted_label=CLASS_NAMES[pred_class],
                    class_probability=class_prob,
                    confidence=class_prob,
                    threat_probability=threat_prob,
                    threat_score=threat_prob,
                    calibrated_risk_score=risk_score,
                    risk_level=self._risk_level(risk_score),
                    class_probabilities={
                        CLASS_NAMES[self.class_order_[j]]: float(proba[i, j])
                        for j in range(len(self.class_order_))
                    },
                )
            )
        return results

    def _threat_probability_from_proba(self, proba: np.ndarray) -> np.ndarray:
        threat_idx = [self.class_order_.index(c) for c in THREAT_CLASSES]
        return proba[:, threat_idx].sum(axis=1)

    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= self.high_threshold_:
            return "HIGH"
        if risk_score >= self.medium_threshold_:
            return "MEDIUM"
        return "LOW"

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("UncalibratedRiskAdapter must be fitted via calibrate() first")

    def get_thresholds(self) -> dict[str, float]:
        self._check_fitted()
        return {
            "high_threshold": float(self.high_threshold_),
            "medium_threshold": float(self.medium_threshold_),
            "target_high_benign_fpr": TARGET_HIGH_BENIGN_FPR,
            "target_medium_plus_benign_fpr": TARGET_MEDIUM_PLUS_BENIGN_FPR,
        }

    def save(self, path: str) -> None:
        import joblib

        self._check_fitted()
        payload = {
            "adapter_type": "uncalibrated",
            "frozen_model": self._frozen_model,
            "metadata": self.metadata,
            "high_threshold": self.high_threshold_,
            "medium_threshold": self.medium_threshold_,
            "class_order": self.class_order_,
            "benign_index": self.benign_index_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "UncalibratedRiskAdapter":
        import joblib

        payload = joblib.load(path)
        obj = cls(payload["frozen_model"], payload["metadata"])
        obj.high_threshold_ = payload["high_threshold"]
        obj.medium_threshold_ = payload["medium_threshold"]
        obj.class_order_ = payload["class_order"]
        obj.benign_index_ = payload["benign_index"]
        obj._is_fitted = True
        return obj
