"""Tests for frozen clean uncalibrated Phase 5 inference."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml.phase5.artifacts import load_clean_artifacts, load_risk_policy, metadata_references_canonical_clean_artifact
from ml.phase5.config import (
    EXPECTED_CLEAN_MODEL,
    LEGACY_PHASE3_MODEL_PATH,
    LEGACY_PHASE4_CALIBRATOR_PATH,
    MODEL_CALIBRATION,
    N_FEATURES,
    PHASE3_CLEAN_MODEL_PATH,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE5_CLEAN_METADATA_PATH,
    PHASE5_CLEAN_RISK_POLICY_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)
from ml.phase5.inference import InputValidationError
from ml.phase5.service import InferenceService


@pytest.fixture(scope="module")
def clean_service():
    if not PHASE5_CLEAN_SERVICE_PATH.exists():
        pytest.skip("Run: python -m ml.promotion.freeze_uncalibrated")
    return InferenceService.from_clean_artifacts()


def test_phase5_loads_clean_linear_svm(clean_service):
    assert isinstance(clean_service.model, Pipeline)
    assert isinstance(clean_service.model.named_steps["clf"], SVC)
    assert clean_service.phase3_metadata["model_name"] == EXPECTED_CLEAN_MODEL
    assert clean_service.model_calibration == MODEL_CALIBRATION


def test_phase5_does_not_load_legacy_phase3(clean_service):
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["uses_legacy_artifacts"] is False
    assert metadata_references_canonical_clean_artifact(
        meta["phase3_artifact_path"],
        PHASE3_CLEAN_MODEL_PATH,
    )
    assert "phase3/selected_model.joblib" not in meta["phase3_artifact_path"].replace("\\", "/")


def test_phase5_does_not_load_legacy_phase4_calibrator():
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["loads_legacy_phase4_calibrator"] is False
    assert meta["calibration_status"] == "NOT_USED_IN_FINAL_INFERENCE"
    bundle = load_clean_artifacts()
    assert bundle["risk_policy"] is not None
    assert not LEGACY_PHASE4_CALIBRATOR_PATH.samefile(PHASE5_CLEAN_RISK_POLICY_PATH)


def test_predictions_deterministic(clean_service, valid_sample):
    assert clean_service.predict(valid_sample) == clean_service.predict(valid_sample)


def test_predict_output_has_five_classes(clean_service, valid_sample):
    result = clean_service.predict(valid_sample)
    assert len(result["class_probabilities"]) == 5
    assert set(result["class_probabilities"].keys()) == {
        "benign", "fsk", "ook", "chirp", "tone",
    }


def test_probabilities_valid(clean_service, valid_batch):
    proba = clean_service.predict_proba(valid_batch)
    assert proba.shape == (valid_batch.shape[0], 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert not np.isnan(proba).any()
    assert not np.isinf(proba).any()


def test_requires_exactly_32_finite_features(clean_service, valid_sample):
    result = clean_service.predict(valid_sample)
    assert result["model_calibration"] == "uncalibrated"
    assert "calibrated_risk_score" not in result
    assert "threat_probability" not in result


def test_wrong_shape_fails(clean_service):
    with pytest.raises(InputValidationError):
        clean_service.predict(np.zeros(31))
    with pytest.raises(InputValidationError):
        clean_service.predict(np.zeros(33))


def test_nan_fails(clean_service, valid_sample):
    bad = valid_sample.copy()
    bad[0] = np.nan
    with pytest.raises(InputValidationError):
        clean_service.predict(bad)


def test_inf_fails(clean_service, valid_sample):
    bad = valid_sample.copy()
    bad[0] = np.inf
    with pytest.raises(InputValidationError):
        clean_service.predict(bad)


def test_artifact_reload_identical(clean_service, valid_batch):
    original = clean_service.predict_batch(valid_batch)
    reloaded = InferenceService.load(PHASE5_CLEAN_SERVICE_PATH)
    restored = reloaded.predict_batch(valid_batch)
    assert original == restored


def test_risk_policy_loads_without_phase4_calibrator():
    policy = load_risk_policy()
    assert policy.is_fitted
    assert policy.get_thresholds()["high_threshold"] >= policy.get_thresholds()["medium_threshold"]


def test_backend_frontend_untouched():
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["backend_modified"] is False
    assert meta["frontend_modified"] is False


def test_locked_test_metrics_recorded():
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    locked = meta["locked_test_metrics"]
    assert locked["benign_false_positive_rate"] == pytest.approx(0.0968, abs=0.001)
    assert locked["macro_f1"] == pytest.approx(0.8246, abs=0.001)


def test_legacy_model_not_used_by_clean_loader():
    legacy = joblib.load(LEGACY_PHASE3_MODEL_PATH)
    clean = joblib.load(PHASE3_CLEAN_MODEL_PATH)
    assert legacy.named_steps["clf"].__class__.__name__ != clean.named_steps["clf"].__class__.__name__ or (
        legacy.named_steps["clf"].coef_.shape != clean.named_steps["clf"].coef_.shape
    )
