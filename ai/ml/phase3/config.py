"""Phase 3 configuration — Dataset V2 ML baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
PHASE3_DIR = Path(__file__).resolve().parent
DATASET_V2_DIR = ML_DIR / "training_data_v2"
PHASE3_OUTPUT_DIR = ML_DIR / "output" / "phase3"

# Authoritative 32-feature contract (matches dsp.FeatureExtractor + dataset_info.json).
FEATURE_NAMES = [
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

N_FEATURES = 32
CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}
EXPECTED_CLASSES = (0, 1, 2, 3, 4)
THREAT_CLASSES = (1, 2, 3, 4)
BENIGN_CLASS = 0

SPLIT_SEED = 42
TEST_FRACTION = 0.2

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "random_state": 42,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 2000,
    "random_state": 42,
}

MODEL_NAMES = ("random_forest", "rbf_svm", "logistic_regression")

SYNTHETIC_DATA_LIMITATION = (
    "All metrics are measured on synthetic Dataset V2 only. "
    "They do not establish real-world or over-the-air detection performance."
)


def load_dataset_info() -> dict:
    path = DATASET_V2_DIR / "dataset_info.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
