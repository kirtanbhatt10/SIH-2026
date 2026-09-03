"""Risk calibration layer on top of the frozen Phase 3 classifier."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from .artifacts import assert_model_frozen, get_benign_class_index, snapshot_model
from .config import (
    BENIGN_CLASS,
    CALIBRATION_METHOD,
    CLASS_NAMES,
    PROB_SUM_TOLERANCE,
    RISK_LEVELS,
    TARGET_HIGH_BENIGN_FPR,
    TARGET_MEDIUM_PLUS_BENIGN_FPR,
    THREAT_CLASSES,
)


@dataclass(frozen=True)
class RiskPrediction:
    predicted_class: int
    predicted_label: str
    class_probability: float
    confidence: float
    threat_probability: float
    threat_score: float
    calibrated_risk_score: float
    risk_level: str
    class_probabilities: dict[str, float]


class RiskCalibrator:
    """Calibrate frozen classifier probabilities and map them to risk levels.

    The base classifier is never refit. Multiclass probability calibration uses
    ``CalibratedClassifierCV(cv='prefit')`` on the training partition only.
    Threat/risk scores are further calibrated with isotonic regression mapping
    raw threat probability to empirical threat prevalence on the train set.
    Risk thresholds (LOW / MEDIUM / HIGH) are chosen data-driven from benign
    calibration scores on the train partition.
    """

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self._frozen_model = model
        self.metadata = metadata
        self.class_order_ = self._resolve_class_order()
        self.benign_index_ = self.class_order_.index(BENIGN_CLASS)
        self._model_snapshot = snapshot_model(model)

        self.calibrated_model_: CalibratedClassifierCV | None = None
        self.threat_isotonic_: IsotonicRegression | None = None
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

    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray) -> "RiskCalibrator":
        """Fit calibration and risk thresholds on the Phase 3 train partition."""
        before = snapshot_model(self._frozen_model)

        base_estimator = copy.deepcopy(self._frozen_model)
        try:
            from sklearn.frozen import FrozenEstimator
            base_estimator = FrozenEstimator(base_estimator)
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=base_estimator,
                method=CALIBRATION_METHOD,
            )
        except ImportError:
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=base_estimator,
                method=CALIBRATION_METHOD,
                cv="prefit",
            )
        self.calibrated_model_.fit(X_train, y_train)

        raw_proba = self._frozen_model.predict_proba(X_train)
        raw_threat = 1.0 - raw_proba[:, self.benign_index_]
        threat_true = (y_train != BENIGN_CLASS).astype(float)

        self.threat_isotonic_ = IsotonicRegression(out_of_bounds="clip")
        self.threat_isotonic_.fit(raw_threat, threat_true)

        calibrated_proba = self.calibrated_model_.predict_proba(X_train)
        calibrated_threat = self._threat_probability_from_proba(calibrated_proba)
        calibrated_risk = self.threat_isotonic_.predict(calibrated_threat)

        self.high_threshold_, self.medium_threshold_ = self._fit_risk_thresholds(
            calibrated_risk, y_train
        )
        self._is_fitted = True

        after = snapshot_model(self._frozen_model)
        assert_model_frozen(before, after)

        return self

    def _fit_risk_thresholds(
        self,
        risk_scores: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[float, float]:
        benign_mask = y_train == BENIGN_CLASS
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
        proba = self.calibrated_model_.predict_proba(X)
        self._validate_probabilities(proba)
        return proba

    def predict_risk(self, X: np.ndarray) -> list[RiskPrediction]:
        self._check_fitted()
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)

        raw_proba = self._frozen_model.predict_proba(X)
        raw_threat = 1.0 - raw_proba[:, self.benign_index_]
        calibrated_threat = self._threat_probability_from_proba(proba)
        calibrated_risk = self.threat_isotonic_.predict(calibrated_threat)

        results: list[RiskPrediction] = []
        for i in range(X.shape[0]):
            pred_class = int(self.class_order_[pred_indices[i]])
            class_prob = float(proba[i, pred_indices[i]])
            threat_prob = float(calibrated_threat[i])
            risk_score = float(calibrated_risk[i])
            results.append(
                RiskPrediction(
                    predicted_class=pred_class,
                    predicted_label=CLASS_NAMES[pred_class],
                    class_probability=class_prob,
                    confidence=class_prob,
                    threat_probability=threat_prob,
                    threat_score=float(raw_threat[i]),
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

    def _validate_probabilities(self, proba: np.ndarray) -> None:
        if np.isnan(proba).any() or np.isinf(proba).any():
            raise ValueError("Calibrated probabilities contain NaN or Inf")
        if (proba < 0).any() or (proba > 1).any():
            raise ValueError("Calibrated probabilities outside [0, 1]")
        sums = proba.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=PROB_SUM_TOLERANCE):
            raise ValueError("Calibrated probabilities do not sum to 1")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("RiskCalibrator must be fitted via calibrate() first")

    def get_thresholds(self) -> dict[str, float]:
        self._check_fitted()
        return {
            "high_threshold": float(self.high_threshold_),
            "medium_threshold": float(self.medium_threshold_),
            "target_high_benign_fpr": TARGET_HIGH_BENIGN_FPR,
            "target_medium_plus_benign_fpr": TARGET_MEDIUM_PLUS_BENIGN_FPR,
        }

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        payload = {
            "frozen_model": self._frozen_model,
            "metadata": self.metadata,
            "calibrated_model": self.calibrated_model_,
            "threat_isotonic": self.threat_isotonic_,
            "high_threshold": self.high_threshold_,
            "medium_threshold": self.medium_threshold_,
            "class_order": self.class_order_,
            "benign_index": self.benign_index_,
            "model_snapshot": self._model_snapshot,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "RiskCalibrator":
        payload = joblib.load(path)
        obj = cls(payload["frozen_model"], payload["metadata"])
        obj.calibrated_model_ = payload["calibrated_model"]
        obj.threat_isotonic_ = payload["threat_isotonic"]
        obj.high_threshold_ = payload["high_threshold"]
        obj.medium_threshold_ = payload["medium_threshold"]
        obj.class_order_ = payload["class_order"]
        obj.benign_index_ = payload["benign_index"]
        obj._model_snapshot = payload["model_snapshot"]
        obj._is_fitted = True
        return obj
