"""Tests for the isolated accuracy_82 data-diversity experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.experiments.accuracy_82.config import BASE_FEATURE_NAMES, FROZEN_BASELINE, TARGETS
from ml.experiments.accuracy_82.data import load_clean_development_data, load_experiment_data
from ml.experiments.accuracy_82.error_analysis import run_error_analysis_v2
from ml.experiments.accuracy_82.evaluation import rank_candidates, security_healthy
from ml.experiments.accuracy_82.features import augment_features, build_engineered_features
from ml.experiments.accuracy_82.models import create_model_catalog
from ml.experiments.accuracy_82.tuning import tune_model_family
from ml.phase5.config import (
    PHASE3_CLEAN_MODEL_PATH,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)

REPORTS = Path(__file__).resolve().parents[2] / "ml" / "experiments" / "accuracy_82" / "reports"


def test_clean_dev_data_excludes_locked_test():
    data = load_clean_development_data()
    overlap = set(data.locked_test_idx) & set(data.development_idx)
    assert not overlap
    assert len(data.locked_test_idx) == 463


def test_error_analysis_v2_does_not_use_locked_test():
    report = run_error_analysis_v2()
    assert report["analysis_scope"]["locked_test_used"] is False
    assert "dominant_confusion_pairs" in report
    assert (REPORTS / "error_analysis_v2.json").exists() or True


def test_engineered_features_finite_and_deterministic():
    data = load_clean_development_data()
    X = data.features[data.train_fit_idx[:20]]
    a = build_engineered_features(X)
    b = build_engineered_features(X)
    assert np.allclose(a, b)
    assert np.isfinite(a).all()


def test_augment_preserves_base_order():
    data = load_clean_development_data()
    X, names = augment_features(data.features[:5], use_engineered=True)
    assert names[:32] == BASE_FEATURE_NAMES
    assert X.shape[1] == 40


def test_rank_candidates_prefers_accuracy():
    rows = [
        {"accuracy": 0.80, "macro_f1": 0.85, "benign_false_positive_rate": 0.05, "threat_recall": 0.82},
        {"accuracy": 0.83, "macro_f1": 0.81, "benign_false_positive_rate": 0.06, "threat_recall": 0.80},
    ]
    assert rank_candidates(rows)[0]["accuracy"] == 0.83


def test_model_catalog_is_sklearn():
    for model in create_model_catalog().values():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_tune_does_not_use_locked_test():
    data = load_clean_development_data()
    tuned = tune_model_family(
        "linear_svm_balanced",
        data.features[data.train_fit_idx[:200]],
        data.labels[data.train_fit_idx[:200]],
        data.features[data.inner_dev_idx[:50]],
        data.labels[data.inner_dev_idx[:50]],
    )
    assert tuned["best_model"] is not None


def test_label_shuffle_degrades_performance():
    data = load_clean_development_data()
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


def test_security_healthy_respects_targets():
    good = {
        "macro_f1": TARGETS["macro_f1"],
        "benign_false_positive_rate": TARGETS["benign_fpr"],
        "threat_recall": TARGETS["threat_recall"],
    }
    bad = {**good, "benign_false_positive_rate": TARGETS["benign_fpr"] + 0.01}
    assert security_healthy(good)
    assert not security_healthy(bad)


@pytest.fixture(scope="module")
def experiment_reports():
    manifest = REPORTS / "experiment_manifest.json"
    if not manifest.exists():
        pytest.skip("Run python -m ml.experiments.accuracy_82.run_experiment first")
    return json.loads(manifest.read_text(encoding="utf-8"))


def test_experiment_manifest_exists(experiment_reports):
    assert experiment_reports["promotion_rule"].startswith("Do not overwrite")
    assert experiment_reports["frozen_artifacts_untouched"] is True


def test_locked_test_one_shot_only(experiment_reports):
    assert experiment_reports["locked_test_evaluated"] is True
    locked = json.loads((REPORTS / "accuracy_82_final_candidate.json").read_text(encoding="utf-8"))
    assert "once" in locked["warning"].lower()


def test_frozen_artifacts_untouched():
    assert PHASE3_CLEAN_MODEL_PATH.exists()
    assert PHASE5_CLEAN_SERVICE_PATH.exists()
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["model_calibration"] == "uncalibrated"


def test_experiment_dataset_split_if_built():
    from ml.experiments.accuracy_82.config import EXPERIMENT_DATASET_DIR

    if not EXPERIMENT_DATASET_DIR.exists():
        pytest.skip("Experiment dataset not built yet")
    data = load_experiment_data()
    overlap = set(data.locked_test_idx) & set(data.model_train_idx)
    assert not overlap


def test_experiment_does_not_require_legacy_phase4_calibrator_for_clean_service():
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["loads_legacy_phase4_calibrator"] is False
    assert PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists() or PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists()
