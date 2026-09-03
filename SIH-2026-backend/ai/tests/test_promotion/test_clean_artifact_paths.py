"""Tests for clean promotion artifact paths and legacy isolation."""

from __future__ import annotations

import json

import pytest

from ml.phase5.config import (
    LEGACY_PHASE3_MODEL_PATH,
    MODEL_CALIBRATION,
    PHASE3_CLEAN_METADATA_PATH,
    PHASE3_CLEAN_MODEL_PATH,
    PHASE4_CLEAN_METADATA_PATH,
    PHASE5_CLEAN_METADATA_PATH,
    PHASE5_CLEAN_RISK_POLICY_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)
from ml.phase5.service import InferenceService
from ml.phase5.artifacts import metadata_references_canonical_clean_artifact
from ml.promotion.config_clean import FROZEN_SPLIT_MANIFEST_PATH, LEGACY_MARKER


@pytest.fixture(scope="module")
def clean_artifacts_present() -> None:
    required = (
        PHASE3_CLEAN_MODEL_PATH,
        PHASE3_CLEAN_METADATA_PATH,
        PHASE4_CLEAN_METADATA_PATH,
        PHASE5_CLEAN_RISK_POLICY_PATH,
        PHASE5_CLEAN_SERVICE_PATH,
        PHASE5_CLEAN_METADATA_PATH,
        FROZEN_SPLIT_MANIFEST_PATH,
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(
            "Clean freeze artifacts not built yet. Run: "
            "python -m ml.promotion.freeze_uncalibrated"
        )


def test_frozen_split_manifest_is_locked(clean_artifacts_present):
    manifest = json.loads(FROZEN_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN"
    assert manifest["leakage_checks"]["passed"] is True
    assert manifest["partition_sizes"]["locked_test"] > 0


def test_phase5_clean_metadata_does_not_use_legacy_phase3(clean_artifacts_present):
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta.get("uses_legacy_artifacts") is False
    assert metadata_references_canonical_clean_artifact(
        meta["phase3_artifact_path"],
        PHASE3_CLEAN_MODEL_PATH,
    )
    assert meta["model_calibration"] == MODEL_CALIBRATION
    assert meta["calibration_status"] == "NOT_USED_IN_FINAL_INFERENCE"


def test_phase3_clean_metadata(clean_artifacts_present):
    meta = json.loads(PHASE3_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["artifact_lineage"] == "clean_uncalibrated_freeze_v1"
    assert meta["dataset_version"] == "dataset_v2_clean.0"
    assert meta["model_name"] == "linear_svm_balanced"
    assert meta["feature_count"] == 32
    assert len(meta["class_mapping"]) == 5


def test_phase5_clean_service_loads_clean_model(clean_artifacts_present):
    service = InferenceService.from_clean_artifacts()
    assert service.phase3_metadata["model_name"] == "linear_svm_balanced"
    assert service.model_calibration == "uncalibrated"


def test_legacy_dirs_marked(clean_artifacts_present):
    legacy_readme = LEGACY_PHASE3_MODEL_PATH.parent / "LEGACY_README.txt"
    assert legacy_readme.exists()
    assert LEGACY_MARKER in legacy_readme.read_text(encoding="utf-8")


def test_clean_artifacts_must_not_reference_legacy_paths(clean_artifacts_present):
    forbidden = "output/phase3/selected_model.joblib"
    for path in (
        PHASE3_CLEAN_METADATA_PATH,
        PHASE4_CLEAN_METADATA_PATH,
        PHASE5_CLEAN_METADATA_PATH,
    ):
        text = path.read_text(encoding="utf-8")
        assert forbidden not in text.replace("\\", "/")


def test_reproducibility_recorded(clean_artifacts_present):
    meta = json.loads(PHASE5_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    repro = meta["reproducibility"]
    assert repro["passed"] is True
