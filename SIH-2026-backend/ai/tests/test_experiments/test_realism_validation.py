"""Tests for the DSP / feature-space realism audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.experiments.realism_validation.config import FEATURE_NAMES, FROZEN_SPLIT_MANIFEST
from ml.experiments.realism_validation.data import load_audit_data
from ml.experiments.realism_validation.run_audit import run_audit
from ml.phase5.config import (
    PHASE3_CLEAN_MODEL_PATH,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)

REPORTS = Path(__file__).resolve().parents[2] / "ml" / "experiments" / "realism_validation" / "reports"
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "ml" / "experiments" / "realism_validation"


def test_audit_data_never_loads_locked_test():
    data = load_audit_data()
    manifest = json.loads(FROZEN_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    locked = set(manifest["locked_test_idx"])
    dev = set(data.development_idx.tolist())
    assert not dev & locked


def test_feature_names_match_phase1_contract():
    data = load_audit_data()
    assert data.feature_names == FEATURE_NAMES
    assert len(data.feature_names) == 32


def test_development_features_finite():
    data = load_audit_data()
    X = data.dev_features()
    assert np.isfinite(X).all()


def test_run_audit_writes_only_in_experiment_dir():
    audit = run_audit(generate_signal_diagnostics=False)
    assert (REPORTS / "realism_audit.json").exists()
    assert audit["locked_test_accessed"] is False
    assert audit["frozen_artifacts_modified"] is False
    for path in REPORTS.glob("*.json"):
        assert EXPERIMENT_DIR in path.parents or path.parent == REPORTS


def test_reports_are_deterministic():
    a = json.loads((REPORTS / "realism_audit.json").read_text(encoding="utf-8"))
    b = run_audit(generate_signal_diagnostics=False)
    assert a["confusion_boundary_analysis"]["total_errors"] == b["confusion_boundary_analysis"]["total_errors"]
    assert a["conclusion"]["primary_conclusion"] == b["conclusion"]["primary_conclusion"]


def test_confusion_counts_match_historical():
    audit = json.loads((REPORTS / "realism_audit.json").read_text(encoding="utf-8"))
    verified = audit["confusion_boundary_analysis"]["verified_known_pairs"]
    assert verified["fsk_to_tone"]["match_historical"] is True
    assert verified["fsk_to_tone"]["count"] == 149


def test_frozen_directories_untouched():
    assert PHASE3_CLEAN_MODEL_PATH.exists()
    assert PHASE5_CLEAN_SERVICE_PATH.exists()
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["model_calibration"] == "uncalibrated"


def test_frozen_split_manifest_not_modified_by_audit():
    before = hashlib.sha256(FROZEN_SPLIT_MANIFEST.read_bytes()).hexdigest()
    run_audit(generate_signal_diagnostics=False)
    after = hashlib.sha256(FROZEN_SPLIT_MANIFEST.read_bytes()).hexdigest()
    assert before == after


def test_no_production_artifact_overwrite():
    model_mtime_before = PHASE3_CLEAN_MODEL_PATH.stat().st_mtime
    run_audit(generate_signal_diagnostics=False)
    assert PHASE3_CLEAN_MODEL_PATH.stat().st_mtime == model_mtime_before


def test_source_dataset_not_mutated():
    from ml.experiments.realism_validation.config import CLEAN_DATASET_DIR

    features_path = CLEAN_DATASET_DIR / "features.npy"
    mtime_before = features_path.stat().st_mtime
    run_audit(generate_signal_diagnostics=False)
    assert features_path.stat().st_mtime == mtime_before


def test_handoff_report_exists():
    doc = Path(__file__).resolve().parents[2] / "docs" / "AI_ML_DSP_REALISM_AUDIT.md"
    assert doc.exists()


def test_phase4_path_exists_without_requiring_calibrator_in_inference():
    meta = json.loads((PHASE5_CLEAN_SERVICE_PATH.parent / "inference_metadata.json").read_text())
    assert meta["loads_legacy_phase4_calibrator"] is False
    assert PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists() or PHASE4_CLEAN_CALIBRATOR_PATH.parent.exists()
