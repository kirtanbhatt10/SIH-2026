"""Shared configuration for the AI 2 (Detection / ML) subsystem.

Everything in this folder is owned by AI 2 and lives under ``ai/ml/``.
It does NOT touch anything under ``ai/dsp/`` (that code is owned by AI 1).

Naming/values follow the AI 1 handoff plan:
  * exactly 32 features, fixed order, never NaN/Inf
  * label map: 0=benign, 1=fsk, 2=ook, 3=chirp, 4=tone
  * ``confidence`` and ``suspicion_score`` must stay separate (audit point 9)
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ML_DIR = Path(__file__).resolve().parent
AI_DIR = ML_DIR.parent
REPO_ROOT = AI_DIR.parent
OUTPUT_DIR = Path(os.environ.get("ML_OUTPUT_DIR", ML_DIR / "output"))
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", AI_DIR / "training_data"))
MODEL_DIR = OUTPUT_DIR / "models"

# --------------------------------------------------------------------------- #
# Dataset contract (from AI 1 handoff plan §3.2, §4)
# --------------------------------------------------------------------------- #
N_FEATURES = 32
CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}
# A sample is "threat-like" if it is any of the modulation classes.
THREAT_CLASSES = (1, 2, 3, 4)

# EXACT names emitted by dsp.feature_extraction.FeatureExtractor.
# Verified against FeatureExtractor.get_feature_names() -- do not "tidy" these.
# The six spectral_* names previously read centroid/bandwidth/flatness/
# rolloff/skewness/kurtosis, which did not match the DSP layer.  Order and
# count were already correct, so nothing broke, but the mismatch made
# feature-importance reports impossible to cross-reference with AI 1's docs.
FEATURE_NAMES = [
    # Energy
    "ultrasonic_energy", "total_energy", "energy_ratio", "energy_db",
    # Spectral shape
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "spectral_skewness", "spectral_kurtosis",
    # Peak structure
    "peak_frequency", "peak_magnitude", "peak_prominence", "num_peaks",
    "peak_spacing_mean", "peak_spacing_std",
    # Tonal
    "tonal_prominence", "harmonic_ratio", "crest_factor", "zero_crossing_rate",
    # Temporal
    "rms_amplitude", "amplitude_envelope_std", "onset_strength",
    "temporal_flatness", "duty_cycle", "bit_rate_estimate",
    # Statistical
    "mean_magnitude", "std_magnitude", "max_magnitude", "min_magnitude",
    "magnitude_range", "coefficient_of_variation",
]

assert len(FEATURE_NAMES) == N_FEATURES, "feature list must be exactly 32"

# --------------------------------------------------------------------------- #
# Data provenance.  Every reported number must state which tier it came from.
# --------------------------------------------------------------------------- #
#   "dsp_synthetic" : synthetic AUDIO -> real dsp.FeatureExtractor  (preferred)
#   "fabricated"    : feature vectors invented in data_generation.py (fallback
#                     only, for when the dsp package is not importable)
#   "dsp_real"      : real over-the-air recordings -> real FeatureExtractor
SOURCE_DSP_SYNTHETIC = "dsp_synthetic"
SOURCE_FABRICATED = "fabricated"
SOURCE_DSP_REAL = "dsp_real"

# --------------------------------------------------------------------------- #
# Model defaults (Phase 1 baseline)
# --------------------------------------------------------------------------- #
BASELINE_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
}

# --------------------------------------------------------------------------- #
# Risk thresholds (Phase 4).
# These are the values AI 1 set as placeholders so Backend 1 was not blocked.
# Calibrating them against a measured FPR is explicitly AI 2's job.
# --------------------------------------------------------------------------- #
# Risk = 1 - P(benign), i.e. how threat-like the sample is.
DEFAULT_RISK_THRESHOLDS = {
    "HIGH": 0.75,
    "MEDIUM": 0.45,
    # anything below MEDIUM is LOW
}

# Target false-positive rate budget we calibrate the HIGH band to hit.
# "False positive" here = a benign sample promoted to HIGH risk.
TARGET_HIGH_FPR = 0.05   # 5% of benign samples may be flagged HIGH
TARGET_MEDIUM_FPR = 0.15

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
