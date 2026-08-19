"""Phase 5 artifact loading and compatibility tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.phase5.artifacts import (
    ArtifactCompatibilityError,
    ArtifactValidationError,
    load_all_artifacts,
    load_phase3_metadata,
    load_phase3_model,
    load_phase4_metadata,
    load_risk_calibrator,
    validate_artifact_compatibility,
)
from ml.phase5.config import (
    CLASS_NAMES,
    FEATURE_NAMES,
    N_FEATURES,
    PHASE3_METADATA_PATH,
    PHASE3_MODEL_PATH,
    PHASE4_CALIBRATOR_PATH,
    PHASE4_METADATA_PATH,
)


def test_phase3_model_exists():
    assert PHASE3_MODEL_PATH.exists()


def test_phase4_calibrator_exists():
    assert PHASE4_CALIBRATOR_PATH.exists()


def test_load_phase3_model():
    model = load_phase3_model()
    assert hasattr(model, "predict_proba")


def test_load_phase3_metadata():
    meta = load_phase3_metadata()
    assert len(meta["feature_names"]) == N_FEATURES
    assert meta["model_name"] == "logistic_regression"


def test_load_phase4_metadata():
    meta = load_phase4_metadata()
    assert "risk_thresholds" in meta


def test_load_risk_calibrator():
    calibrator = load_risk_calibrator()
    assert calibrator.is_fitted


def test_feature_names_order():
    meta = load_phase3_metadata()
    assert meta["feature_names"] == FEATURE_NAMES


def test_class_mapping():
    meta = load_phase3_metadata()
    mapping = {int(k): v for k, v in meta["class_mapping"].items()}
    assert mapping == CLASS_NAMES


def test_phase3_phase4_compatibility():
    p3 = load_phase3_metadata()
    p4 = load_phase4_metadata()
    validate_artifact_compatibility(p3, p4)


def test_load_all_artifacts():
    bundle = load_all_artifacts()
    assert "model" in bundle
    assert "calibrator" in bundle
    assert bundle["calibrator"].is_fitted


def test_incompatible_class_mapping_rejected():
    p3 = load_phase3_metadata()
    p4 = load_phase4_metadata()
    bad = dict(p4)
    bad["class_mapping"] = {"0": "benign", "1": "fsk", "2": "ook", "3": "chirp", "4": "unknown"}
    with pytest.raises(ArtifactCompatibilityError):
        validate_artifact_compatibility(p3, bad)
