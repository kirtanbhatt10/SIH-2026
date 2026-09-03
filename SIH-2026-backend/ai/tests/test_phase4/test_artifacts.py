"""Phase 4 artifact loading and metadata validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from ml.phase4.artifacts import (
    ArtifactValidationError,
    load_phase3_metadata,
    load_phase3_model,
    validate_phase3_metadata,
)
from ml.phase4.config import FEATURE_NAMES, PHASE3_METADATA_PATH, PHASE3_MODEL_PATH


def test_load_real_phase3_model():
    model = load_phase3_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_load_real_phase3_metadata():
    metadata = load_phase3_metadata()
    assert len(metadata["feature_names"]) == 32
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["split_method"] == "stratified_grouped_by_generation_group"


def test_metadata_feature_order():
    metadata = load_phase3_metadata()
    assert metadata["feature_names"] == FEATURE_NAMES


def test_metadata_class_mapping():
    metadata = load_phase3_metadata()
    mapping = {int(k): v for k, v in metadata["class_mapping"].items()}
    assert mapping[0] == "benign"
    assert mapping[4] == "tone"


def test_invalid_feature_count_rejected():
    metadata = load_phase3_metadata()
    bad = dict(metadata)
    bad["feature_names"] = FEATURE_NAMES[:10]
    with pytest.raises(ArtifactValidationError):
        validate_phase3_metadata(bad)


def test_model_not_mutated_on_load():
    model = load_phase3_model()
    params_before = model.get_params(deep=True)
    _ = model.predict_proba([[0.0] * 32])
    assert model.get_params(deep=True) == params_before
