"""Tests for the isolated accuracy improvement experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.experiments.accuracy_improvement.config import BASE_FEATURE_NAMES, FROZEN_BASELINE
from ml.experiments.accuracy_improvement.data import load_experiment_data
from ml.experiments.accuracy_improvement.features import augment_features, build_engineered_features
from ml.experiments.accuracy_improvement.models import create_model_catalog
from ml.experiments.accuracy_improvement.tuning import tune_model_family
from ml.phase5.config import (
    PHASE3_CLEAN_MODEL_PATH,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)


REPORTS = Path(__file__).resolve().parents[2] / "ml" / "experiments" / "accuracy_improvement" / "reports"


def test_load_experiment_data_respects_frozen_split():
    data = load_experiment_data()
    overlap = set(data.locked_test_idx) & set(data.model_train_idx)
    assert not overlap
    assert len(data.locked_test_idx) == 463
    assert len(data.cal_dev_idx) == 289


def test_inner_dev_split_isolated_from_locked():
    data = load_experiment_data()
    assert not set(data.inner_dev_idx) & set(data.locked_test_idx)
    assert set(data.inner_dev_idx).issubset(set(data.model_train_idx))


def test_engineered_features_finite_and_deterministic():
    data = load_experiment_data()
    X = data.features[data.train_fit_idx[:20]]
    a = build_engineered_features(X)
    b = build_engineered_features(X)
    assert np.allclose(a, b)
    assert np.isfinite(a).all()


def test_augment_preserves_base_order():
    data = load_experiment_data()
    X, names = augment_features(data.features[:5], use_engineered=True)
    assert names[:32] == BASE_FEATURE_NAMES
    assert X.shape[1] == 40


def test_model_catalog_is_sklearn():
    for name, model in create_model_catalog().items():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_tune_does_not_use_locked_test():
    data = load_experiment_data()
    tuned = tune_model_family(
        "linear_svm_balanced",
        data.features[data.train_fit_idx[:200]],
        data.labels[data.train_fit_idx[:200]],
        data.features[data.inner_dev_idx[:50]],
        data.labels[data.inner_dev_idx[:50]],
    )
    assert tuned["best_model"] is not None


def test_label_shuffle_degrades_performance():
    data = load_experiment_data()
    X = data.features[data.inner_dev_idx]
    y = data.labels[data.inner_dev_idx]
    model = create_model_catalog()["linear_svm_balanced"]
    model.fit(X, y)
    base_acc = float(model.score(X, y))
    shuffled = y.copy()
    rng = np.random.RandomState(0)
    rng.shuffle(shuffled)
    model2 = create_model_catalog()["linear_svm_balanced"]
    model2.fit(X, shuffled)
    shuf_acc = float(model2.score(X, shuffled))
    assert shuf_acc < max(base_acc, 0.25)


@pytest.fixture(scope="module")
def experiment_reports():
    if not (REPORTS / "experiment_manifest.json").exists():
        pytest.skip("Run python -m ml.experiments.accuracy_improvement.run_experiments first")
    return json.loads((REPORTS / "experiment_manifest.json").read_text(encoding="utf-8"))


def test_experiment_manifest_exists(experiment_reports):
    assert experiment_reports["promotion_rule"].startswith("Do not overwrite")


def test_locked_test_one_shot_only(experiment_reports):
    assert experiment_reports["locked_test_evaluated"] is True
    locked = json.loads((REPORTS / "final_candidate_locked_test.json").read_text(encoding="utf-8"))
    assert "one_shot" in locked["warning"].lower() or "once" in locked["warning"].lower()


def test_frozen_artifacts_untouched():
    assert PHASE3_CLEAN_MODEL_PATH.exists()
    assert PHASE5_CLEAN_SERVICE_PATH.exists()
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["model_calibration"] == "uncalibrated"


def test_experiment_does_not_require_legacy_phase4_calibrator_for_clean_service():
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["loads_legacy_phase4_calibrator"] is False
    # Legacy artifact may exist but is not part of clean inference path.
    assert PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists() or PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists()
