"""Experiment diversity overrides — extends frozen config_v2 without modifying it."""

from __future__ import annotations

from ml.dataset_v2 import config_v2 as base

for _name in dir(base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(base, _name)

EXPERIMENT_DATASET_VERSION = "dataset_v2_experiment.0"
EXPERIMENT_OUTPUT_DIR_NAME = "training_data_v2_experiment"
GENERATOR_VERSION = "dataset_v2_generator-1.2-experiment-diversity"

# --- FSK: boundary-focused diversity within the frozen 18–21 kHz band ---
FSK_MARK_RANGE = (18000.0, 19500.0)
FSK_DEVIATION_RANGE = (200.0, 1500.0)
FSK_BAUD_RANGE = (10.0, 95.0)

# --- OOK: wider timing / carrier diversity within band ---
OOK_CARRIER_RANGE = (18000.0, 21000.0)
OOK_BAUD_RANGE = (20.0, 85.0)

# --- Tone: full-band carriers; overlap FSK/OOK decision space ---
TONE_FREQ_RANGE = (18000.0, 21000.0)

# --- Amplitude / SNR: more hard examples ---
AMPLITUDE_RANGE = (0.45, 1.0)
BENIGN_AMPLITUDE_RANGE = (0.12, 0.65)
SNR_TIER_WEIGHTS = [0.30, 0.30, 0.25, 0.15]

# --- Benign: richer ultrasonic noise profiles ---
BENIGN_VARIANT_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]
BENIGN_HF_BOOST_SNR_RANGE = (-8.0, 28.0)
BENIGN_STEADY_TONE_AMP_RANGE = (0.20, 0.60)
BENIGN_HISS_AMP_RANGE = (0.30, 0.80)

# --- Timing diversity ---
SOURCE_FACTOR_RANGE = (4.0, 12.0)
MAX_CROP_RESAMPLE_ATTEMPTS = 25
BENIGN_CROP_ATTEMPTS = 6
