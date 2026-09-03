"""Phase 4 calibrator behavior tests."""

from __future__ import annotations

import numpy as np

from ml.phase4.artifacts import assert_model_frozen, load_phase3_model, snapshot_model
from ml.phase4.calibrator import RiskCalibrator
from ml.phase3.data import grouped_train_test_split, load_dataset_v2


def test_calibration_fit_and_predict(mini_phase3_artifacts):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)

    before = snapshot_model(model)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(X_train, y_train)
    preds = calibrator.predict_risk(X_test)
    proba = calibrator.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert proba.shape == (len(X_test), 5)
    assert_model_frozen(before, snapshot_model(model))


def test_probability_validity(mini_phase3_artifacts):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)

    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(X_train, y_train)
    proba = calibrator.predict_proba(X_test)

    assert not np.isnan(proba).any()
    assert not np.isinf(proba).any()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_risk_levels_are_valid(mini_phase3_artifacts):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(
        dataset.features[split.train_idx],
        dataset.labels[split.train_idx],
    )
    preds = calibrator.predict_risk(dataset.features[split.test_idx])
    assert all(p.risk_level in {"LOW", "MEDIUM", "HIGH"} for p in preds)


def test_confidence_separate_from_threat_score(mini_phase3_artifacts):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(
        dataset.features[split.train_idx],
        dataset.labels[split.train_idx],
    )
    preds = calibrator.predict_risk(dataset.features[split.test_idx])
    for pred in preds:
        assert pred.confidence == pred.class_probability
        assert 0.0 <= pred.threat_score <= 1.0
        assert 0.0 <= pred.threat_probability <= 1.0


def test_deterministic_predictions(mini_phase3_artifacts):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)

    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]

    cal_a = RiskCalibrator(model, metadata)
    cal_a.calibrate(X_train, y_train)
    preds_a = cal_a.predict_risk(X_test)

    cal_b = RiskCalibrator(model, metadata)
    cal_b.calibrate(X_train, y_train)
    preds_b = cal_b.predict_risk(X_test)

    assert [p.calibrated_risk_score for p in preds_a] == [p.calibrated_risk_score for p in preds_b]


def test_save_load_consistency(mini_phase3_artifacts, tmp_path):
    dataset = mini_phase3_artifacts["dataset"]
    split = mini_phase3_artifacts["split"]
    model = load_phase3_model(mini_phase3_artifacts["model_path"])
    with mini_phase3_artifacts["metadata_path"].open(encoding="utf-8") as fh:
        import json
        metadata = json.load(fh)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(X_train, y_train)
    original = calibrator.predict_risk(X_test)

    path = tmp_path / "risk_calibrator.joblib"
    calibrator.save(path)
    loaded = RiskCalibrator.load(path)
    restored = loaded.predict_risk(X_test)

    assert [p.risk_level for p in original] == [p.risk_level for p in restored]
    assert np.allclose(
        [p.calibrated_risk_score for p in original],
        [p.calibrated_risk_score for p in restored],
    )
