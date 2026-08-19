"""Phase 4 configuration."""

from __future__ import annotations

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
PHASE4_DIR = Path(__file__).resolve().parent
PHASE3_OUTPUT_DIR = ML_DIR / "output" / "phase3"
PHASE4_OUTPUT_DIR = ML_DIR / "output" / "phase4"

PHASE3_MODEL_PATH = PHASE3_OUTPUT_DIR / "selected_model.joblib"
PHASE3_METADATA_PATH = PHASE3_OUTPUT_DIR / "selected_model_metadata.json"

# Must match Phase 3 split policy exactly.
SPLIT_METHOD = "stratified_grouped_by_generation_group"
SPLIT_SEED = 42
TEST_FRACTION = 0.20

# Authoritative Phase-1 feature contract (32 features, fixed order).
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

# Probability calibration (fitted on train partition only; base model stays frozen).
CALIBRATION_METHOD = "isotonic"
CALIBRATION_SEED = 42

# Data-driven risk-band targets on the calibration (train) partition.
# HIGH: at most this fraction of benign samples may land in HIGH.
TARGET_HIGH_BENIGN_FPR = 0.05
# MEDIUM-or-above: at most this fraction of benign samples may land in MEDIUM+HIGH.
TARGET_MEDIUM_PLUS_BENIGN_FPR = 0.15

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

PROB_SUM_TOLERANCE = 1e-4

SYNTHETIC_DATA_LIMITATION = (
    "Calibration and risk metrics are measured on synthetic Dataset V2 only. "
    "They do not establish real-world or over-the-air detection performance."
)
