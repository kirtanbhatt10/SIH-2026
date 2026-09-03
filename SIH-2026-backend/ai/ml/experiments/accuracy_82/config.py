"""Configuration for the 82% accuracy data-diversity experiment."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ML_DIR = EXPERIMENT_DIR.parents[1]
AI_DIR = ML_DIR.parent

CLEAN_DATASET_DIR = ML_DIR / "training_data_v2_clean"
EXPERIMENT_DATASET_DIR = ML_DIR / "training_data_v2_experiment"
FROZEN_CLEAN_MANIFEST = ML_DIR / "output" / "clean_splits" / "frozen_split_manifest.json"
EXPERIMENT_SPLIT_MANIFEST = EXPERIMENT_DIR / "splits" / "experiment_split_manifest.json"

REPORTS_DIR = EXPERIMENT_DIR / "reports"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"

FROZEN_BASELINE = {
    "model": "linear_svm_balanced",
    "calibration": "uncalibrated",
    "dataset_version": "dataset_v2_clean.0",
    "locked_test": {
        "accuracy": 0.7948,
        "macro_f1": 0.8246,
        "weighted_f1": 0.7973,
        "benign_precision": 0.8750,
        "benign_recall": 0.9032,
        "benign_fpr": 0.0968,
        "threat_precision": 0.8277,
        "threat_recall": 0.8046,
        "latency_ms_per_sample": 0.0285,
    },
}

TARGETS = {
    "accuracy": 0.82,
    "macro_f1": 0.82,
    "benign_fpr": 0.10,
    "threat_recall": 0.80,
}

OUTER_SPLIT_SEED = 42
CAL_DEV_SEED = 43
INNER_DEV_SEED = 44
INNER_DEV_FRACTION = 0.15

EXPERIMENT_DATASET_VERSION = "dataset_v2_experiment.0"
GENERATOR_VERSION = "dataset_v2_generator-1.2-experiment-diversity"

CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}
EXPERIMENT_ID = "accuracy_82_data_diversity_v1"

CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}
BENIGN_CLASS = 0
THREAT_CLASSES = (1, 2, 3, 4)

BASE_FEATURE_NAMES = [
    "ultrasonic_energy", "total_energy", "energy_ratio", "energy_db",
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "spectral_skewness", "spectral_kurtosis",
    "peak_frequency", "peak_magnitude", "peak_prominence", "num_peaks",
    "peak_spacing_mean", "peak_spacing_std",
    "tonal_prominence", "harmonic_ratio", "crest_factor", "zero_crossing_rate",
    "rms_amplitude", "amplitude_envelope_std", "onset_strength",
    "temporal_flatness", "duty_cycle", "bit_rate_estimate",
    "mean_magnitude", "std_magnitude", "max_magnitude", "min_magnitude",
    "magnitude_range", "coefficient_of_variation",
]

ENGINEERED_FEATURE_NAMES = [
    "log_ultrasonic_energy",
    "spectral_concentration",
    "peak_to_rms_ratio",
    "bandwidth_centroid_ratio",
    "flatness_rolloff_product",
    "threat_band_energy_ratio",
    "envelope_to_rms",
    "modulation_index",
]

DOMINANT_CONFUSION_PAIRS = [
    ("fsk", "tone"),
    ("ook", "benign"),
    ("ook", "tone"),
    ("tone", "benign"),
]
