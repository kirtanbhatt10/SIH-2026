"""Configuration for the DSP / feature-space realism audit."""

from __future__ import annotations

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ML_DIR = EXPERIMENT_DIR.parents[1]
AI_DIR = ML_DIR.parent

CLEAN_DATASET_DIR = ML_DIR / "training_data_v2_clean"
FROZEN_SPLIT_MANIFEST = ML_DIR / "output" / "clean_splits" / "frozen_split_manifest.json"
FROZEN_MODEL_PATH = ML_DIR / "output" / "phase3_clean" / "selected_model.joblib"

REPORTS_DIR = EXPERIMENT_DIR / "reports"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "diagnostic_samples"

FROZEN_BASELINE = {
    "model": "linear_svm_balanced",
    "dataset_version": "dataset_v2_clean.0",
    "locked_test_reference_only": {
        "accuracy": 0.7948,
        "macro_f1": 0.8246,
        "benign_fpr": 0.0968,
        "threat_recall": 0.8046,
        "note": "Reference numbers only — locked test NOT accessed during audit.",
    },
}

CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}
CLASS_ID = {v: k for k, v in CLASS_NAMES.items()}
BENIGN_CLASS = 0

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

CONFUSION_PAIRS = [
    ("fsk", "tone"),
    ("benign", "ook"),
    ("ook", "benign"),
    ("ook", "tone"),
    ("tone", "benign"),
]

HISTORICAL_DEV_ERRORS = {
    "fsk_to_tone": 149,
    "benign_to_ook": 57,
    "ook_to_benign": 52,
    "ook_to_tone": 50,
    "tone_to_benign": 23,
}

METADATA_NUMERIC = [
    "snr", "amplitude", "frequency", "frequency_deviation", "duty_cycle",
    "noise_level", "bit_rate", "peak_amplitude", "source_duration_sec",
    "crop_offset_sec", "crop_attempts",
]

METADATA_CATEGORICAL = ["noise_type", "benign_variant", "signal_type"]

AUDIT_SEED = 42
