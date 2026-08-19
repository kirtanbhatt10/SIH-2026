"""Phase 5 InferenceService tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.phase5.config import N_FEATURES, RISK_LEVELS
from ml.phase5.inference import InputValidationError
from ml.phase5.service import InferenceService


@pytest.fixture
def service():
    return InferenceService.from_artifacts()


def test_service_construction(service):
    assert service.feature_names is not None
    assert len(service.feature_names) == 32
    thresholds = service.get_risk_thresholds()
    assert "high_threshold" in thresholds
    assert "medium_threshold" in thresholds


def test_single_inference(service, valid_sample):
    result = service.predict(valid_sample)
    assert set(result.keys()) == {
        "predicted_class_id",
        "predicted_class",
        "confidence",
        "class_probabilities",
        "threat_score",
        "threat_probability",
        "calibrated_risk_score",
        "risk_level",
    }
    assert result["risk_level"] in RISK_LEVELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["threat_score"] <= 1.0
    assert 0.0 <= result["calibrated_risk_score"] <= 1.0


def test_batch_inference(service, valid_batch):
    results = service.predict_batch(valid_batch)
    assert len(results) == valid_batch.shape[0]
    for result in results:
        assert result["risk_level"] in RISK_LEVELS


def test_wrong_shape_rejected(service):
    with pytest.raises(InputValidationError):
        service.predict(np.zeros((2, N_FEATURES)))
    with pytest.raises(InputValidationError):
        service.predict_batch(np.zeros(N_FEATURES))


def test_nan_rejected(service, valid_sample):
    bad = valid_sample.copy()
    bad[0] = np.nan
    with pytest.raises(InputValidationError):
        service.predict(bad)


def test_inf_rejected(service, valid_sample):
    bad = valid_sample.copy()
    bad[0] = np.inf
    with pytest.raises(InputValidationError):
        service.predict(bad)


def test_probability_validity(service, valid_batch):
    proba = service.predict_proba(valid_batch)
    assert proba.shape == (valid_batch.shape[0], 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert not np.isnan(proba).any()
    assert not np.isinf(proba).any()


def test_deterministic_inference(service, valid_sample):
    a = service.predict(valid_sample)
    b = service.predict(valid_sample)
    assert a == b


def test_risk_mapping_uses_phase4_thresholds(service):
    thresholds = service.get_risk_thresholds()
    high = thresholds["high_threshold"]
    medium = thresholds["medium_threshold"]
    assert high >= medium


def test_save_load_consistency(service, valid_batch, tmp_path):
    path = tmp_path / "service.joblib"
    service.save(path)
    loaded = InferenceService.load(path)
    original = service.predict_batch(valid_batch)
    restored = loaded.predict_batch(valid_batch)
    assert original == restored


def test_frozen_model_not_mutated(service, valid_batch):
    from ml.phase5.artifacts import snapshot_frozen_model
    before = snapshot_frozen_model(service.model)
    service.predict_batch(valid_batch)
    after = snapshot_frozen_model(service.model)
    for key in before:
        assert np.array_equal(before[key], after[key])


def test_latency_benchmark(service, valid_batch):
    latency = service.benchmark_latency(valid_batch)
    assert latency["latency_ms_per_sample"] >= 0.0
