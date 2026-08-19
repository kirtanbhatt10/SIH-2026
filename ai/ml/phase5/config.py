"""Phase 5 configuration."""

from __future__ import annotations

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
PHASE5_DIR = Path(__file__).resolve().parent
PHASE3_OUTPUT_DIR = ML_DIR / "output" / "phase3"
PHASE4_OUTPUT_DIR = ML_DIR / "output" / "phase4"
PHASE5_OUTPUT_DIR = ML_DIR / "output" / "phase5"
DATASET_V2_DIR = ML_DIR / "training_data_v2"

PHASE3_MODEL_PATH = PHASE3_OUTPUT_DIR / "selected_model.joblib"
PHASE3_METADATA_PATH = PHASE3_OUTPUT_DIR / "selected_model_metadata.json"
PHASE4_CALIBRATOR_PATH = PHASE4_OUTPUT_DIR / "risk_calibrator.joblib"
PHASE4_METADATA_PATH = PHASE4_OUTPUT_DIR / "calibration_metadata.json"
PHASE4_METRICS_PATH = PHASE4_OUTPUT_DIR / "calibration_metrics.json"

# Authoritative Phase-1 feature contract.
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
BENIGN_CLASS = 0
THREAT_CLASSES = (1, 2, 3, 4)
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

EXPECTED_PHASE3_MODEL = "logistic_regression"
PROB_SUM_TOLERANCE = 1e-4
SPLIT_SEED = 42

SYNTHETIC_DATA_LIMITATION = (
    "Phase 5 inference metrics on Dataset V2 are synthetic-data baselines only. "
    "They do not establish real-world or over-the-air detection performance."
)
