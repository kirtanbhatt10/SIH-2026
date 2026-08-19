"""Phase 5 inference service API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .artifacts import (
    assert_model_unchanged,
    load_all_artifacts,
    snapshot_frozen_model,
)
from .config import FEATURE_NAMES, PHASE5_OUTPUT_DIR
from .inference import (
    risk_prediction_to_dict,
    validate_and_prepare_batch,
    validate_and_prepare_features,
    validate_probability_matrix,
)


class InferenceService:
    """Frozen inference wrapper: 32-feature vector(s) -> calibrated risk output.

    Phase 3 classifier and Phase 4 calibration are loaded once and never refit.
    """

    def __init__(
        self,
        model: Any,
        calibrator: Any,
        phase3_metadata: dict[str, Any],
        phase4_metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.phase3_metadata = phase3_metadata
        self.phase4_metadata = phase4_metadata
        self.feature_names = list(FEATURE_NAMES)
        self._model_snapshot = snapshot_frozen_model(model)
        self._thresholds = dict(phase4_metadata["risk_thresholds"])

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_path: Path | None = None,
        phase3_metadata_path: Path | None = None,
        calibrator_path: Path | None = None,
        phase4_metadata_path: Path | None = None,
    ) -> "InferenceService":
        bundle = load_all_artifacts(
            model_path=model_path,
            phase3_metadata_path=phase3_metadata_path,
            calibrator_path=calibrator_path,
            phase4_metadata_path=phase4_metadata_path,
        )
        return cls(
            model=bundle["model"],
            calibrator=bundle["calibrator"],
            phase3_metadata=bundle["phase3_metadata"],
            phase4_metadata=bundle["phase4_metadata"],
        )

    def predict(self, features: Any) -> dict[str, Any]:
        """Run single-sample inference on exactly 32 features."""
        x = validate_and_prepare_features(features, feature_names=self.feature_names)
        x = x.reshape(1, -1)
        before = snapshot_frozen_model(self.model)
        result = self.predict_batch(x)[0]
        assert_model_unchanged(before, snapshot_frozen_model(self.model))
        return result

    def predict_batch(self, features: Any) -> list[dict[str, Any]]:
        """Run batch inference on an (N, 32) feature array."""
        x = validate_and_prepare_batch(features, feature_names=self.feature_names)
        before = snapshot_frozen_model(self.model)
        predictions = self.calibrator.predict_risk(x)
        proba = self.calibrator.predict_proba(x)
        validate_probability_matrix(proba)
        assert_model_unchanged(before, snapshot_frozen_model(self.model))
        return [risk_prediction_to_dict(p) for p in predictions]

    def predict_proba(self, features: Any) -> np.ndarray:
        x = validate_and_prepare_batch(features, feature_names=self.feature_names)
        before = snapshot_frozen_model(self.model)
        proba = self.calibrator.predict_proba(x)
        validate_probability_matrix(proba)
        assert_model_unchanged(before, snapshot_frozen_model(self.model))
        return proba

    def get_risk_thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def benchmark_latency(
        self,
        features: Any,
        *,
        repeats: int = 3,
    ) -> dict[str, float]:
        x = validate_and_prepare_batch(features, feature_names=self.feature_names)
        if x.shape[0] == 0:
            return {
                "latency_ms_mean": 0.0,
                "latency_ms_std": 0.0,
                "latency_ms_per_sample": 0.0,
            }

        # Warm-up
        self.predict_batch(x[:1])

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.predict_batch(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms / x.shape[0])

        return {
            "latency_ms_mean": float(np.mean(timings)),
            "latency_ms_std": float(np.std(timings)),
            "latency_ms_per_sample": float(np.mean(timings)),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model,
            "calibrator": self.calibrator,
            "phase3_metadata": self.phase3_metadata,
            "phase4_metadata": self.phase4_metadata,
            "model_snapshot": self._model_snapshot,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "InferenceService":
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            calibrator=payload["calibrator"],
            phase3_metadata=payload["phase3_metadata"],
            phase4_metadata=payload["phase4_metadata"],
        )
