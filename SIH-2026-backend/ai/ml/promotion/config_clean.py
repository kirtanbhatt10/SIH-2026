"""Configuration for clean Dataset V2 promotion artifacts."""

from __future__ import annotations

from pathlib import Path

from ml.dataset_v2 import config_v2 as dataset_cfg

ML_DIR = Path(__file__).resolve().parent.parent
PROMOTION_DIR = Path(__file__).resolve().parent

CLEAN_DATASET_DIR = ML_DIR / dataset_cfg.CLEAN_OUTPUT_DIR_NAME
CLEAN_SPLITS_DIR = ML_DIR / "output" / "clean_splits"
FROZEN_SPLIT_MANIFEST_PATH = CLEAN_SPLITS_DIR / "frozen_split_manifest.json"

PHASE3_CLEAN_OUTPUT_DIR = ML_DIR / "output" / "phase3_clean"
PHASE4_CLEAN_OUTPUT_DIR = ML_DIR / "output" / "phase4_clean"
PHASE5_CLEAN_OUTPUT_DIR = ML_DIR / "output" / "phase5_clean"

LEGACY_PHASE3_DIR = ML_DIR / "output" / "phase3"
LEGACY_PHASE4_DIR = ML_DIR / "output" / "phase4"
LEGACY_PHASE5_DIR = ML_DIR / "output" / "phase5"

PHASE3_CLEAN_MODEL_PATH = PHASE3_CLEAN_OUTPUT_DIR / "selected_model.joblib"
PHASE3_CLEAN_METADATA_PATH = PHASE3_CLEAN_OUTPUT_DIR / "selected_model_metadata.json"
PHASE4_CLEAN_CALIBRATOR_PATH = PHASE4_CLEAN_OUTPUT_DIR / "risk_calibrator.joblib"
PHASE4_CLEAN_METADATA_PATH = PHASE4_CLEAN_OUTPUT_DIR / "calibration_metadata.json"
PHASE4_CLEAN_METRICS_PATH = PHASE4_CLEAN_OUTPUT_DIR / "calibration_metrics.json"
PHASE5_CLEAN_SERVICE_PATH = PHASE5_CLEAN_OUTPUT_DIR / "inference_service.joblib"
PHASE5_CLEAN_METADATA_PATH = PHASE5_CLEAN_OUTPUT_DIR / "inference_metadata.json"

SELECTED_MODEL_NAME = "linear_svm_balanced"
MODEL_VERSION = "clean_linear_svm_balanced_v1"
CALIBRATION_VERSION = "clean_isotonic_v1"
INFERENCE_VERSION = "clean_inference_v1"

SPLIT_SEED = 42
TEST_FRACTION = 0.2
CAL_DEV_SEED = 43
CAL_DEV_FRACTION = 0.15

DATASET_VERSION = dataset_cfg.CLEAN_DATASET_VERSION
GENERATOR_VERSION = dataset_cfg.GENERATOR_VERSION

LEGACY_MARKER = "LEGACY — OLD DATASET V2"
