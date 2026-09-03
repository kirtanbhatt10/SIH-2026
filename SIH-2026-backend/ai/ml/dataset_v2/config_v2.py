"""
Configuration for Synthetic Dataset V2.

Every range/value here was chosen to be consistent with facts traced from
the existing, frozen code (see ai/ml/dataset_v2/README.md "Audit" section):

  * dsp.SAMPLE_RATE   = 48000
  * dsp.CHUNK_SIZE    = dsp.N_FFT = 2048  ->  the FeatureExtractor always
    sees a fixed ~42.7 ms window. This is NOT a dataset-generation choice;
    it is how the runtime pipeline extracts features one chunk at a time.
    Dataset V2 therefore does not "vary duration" of the feature-extraction
    window itself (that would silently create train/serve skew). Instead
    it varies the length of the *source* waveform generated before a
    chunk-sized window is cropped from it (see generator.py), which is
    where genuine timing/phase diversity can come from without touching
    the frozen chunk size.
  * dsp.ULTRASONIC_LOW / HIGH = 18000 / 21000 Hz (detection band)
  * dsp.FILTER_LOW / HIGH     = 17500 / 21500 Hz (bandpass edges)

Nothing here changes SignalGenerator, AudioPreprocessor, or
FeatureExtractor. This file only configures how Dataset V2 calls them.
"""

DATASET_VERSION = "dataset_v2.0"
GENERATOR_VERSION = "dataset_v2_generator-1.2"

# Clean rebuild uses a separate on-disk dataset version (does not overwrite V2.0).
CLEAN_DATASET_VERSION = "dataset_v2_clean.0"
CLEAN_OUTPUT_DIR_NAME = "training_data_v2_clean"

# Phase-1 frozen detection band (must match dsp.ULTRASONIC_LOW / ULTRASONIC_HIGH).
# Phase-2 generation parameters are clamped to this band; do not change dsp/__init__.py.
DETECTION_BAND_LOW = 18000.0
DETECTION_BAND_HIGH = 21000.0

CLASS_NAMES = {0: "benign", 1: "fsk", 2: "ook", 3: "chirp", 4: "tone"}

# --------------------------------------------------------------------------- #
# Dataset size (Task 8: diversity > raw count; keep it runnable in seconds)
# --------------------------------------------------------------------------- #
DEFAULT_SAMPLES_PER_CLASS = 500

# --------------------------------------------------------------------------- #
# Source-waveform length, before cropping to the fixed 2048-sample window.
# Expressed as a multiple of the fixed chunk duration (~42.7 ms). A longer
# source gives more bit periods / sweeps to crop a window out of, which is
# how Dataset V2 gets timing diversity without changing the chunk size.
# --------------------------------------------------------------------------- #
SOURCE_FACTOR_RANGE = (5.0, 10.0)   # -> ~213 ms - 427 ms of source audio
                                     # (wider than v0 to give OOK enough "on"
                                     # bits across a random-crop window)

# --------------------------------------------------------------------------- #
# Frequency ranges (Hz). Clamped to the frozen 18–21 kHz detection band.
# FILTER_LOW/HIGH (17500–21500) are wider passband edges; teachable energy
# must stay inside DETECTION_BAND_LOW/HIGH so FeatureExtractor metrics and
# runtime detection see the same band the labels describe.
# --------------------------------------------------------------------------- #
TONE_FREQ_RANGE = (18300.0, DETECTION_BAND_HIGH)

FSK_MARK_RANGE = (18200.0, 19800.0)
FSK_DEVIATION_RANGE = (400.0, 2200.0)   # freq_space = freq_mark + deviation
FSK_MAX_SPACE_FREQ = DETECTION_BAND_HIGH  # hard cap regardless of deviation
FSK_BAUD_RANGE = (15.0, 90.0)           # bits/sec

OOK_CARRIER_RANGE = (18300.0, DETECTION_BAND_HIGH)
OOK_BAUD_RANGE = (30.0, 70.0)       # min 30 bps -> ~33 ms/bit, guarantees at
                                     # least 1-2 bit transitions per 42.7 ms
                                     # crop window, preventing degenerate
                                     # all-off crops at very low baud rates

CHIRP_START_RANGE = (DETECTION_BAND_LOW, 19300.0)
CHIRP_END_RANGE = (20200.0, DETECTION_BAND_HIGH)
CHIRP_SWEEPS_RANGE = (2, 8)             # integer, inclusive

# --------------------------------------------------------------------------- #
# Amplitude (post-generation scale factor applied before noise mixing)
# --------------------------------------------------------------------------- #
AMPLITUDE_RANGE = (0.55, 1.0)
BENIGN_AMPLITUDE_RANGE = (0.15, 0.6)

# --------------------------------------------------------------------------- #
# SNR / noise. Stratified into named tiers so the dataset is guaranteed to
# cover "very noisy" through "clean" rather than relying on chance from a
# single uniform draw (Task 5).
# --------------------------------------------------------------------------- #
SNR_TIERS = {
    "very_noisy": (-5.0, 2.0),
    "noisy": (2.0, 10.0),
    "moderate": (10.0, 20.0),
    "clean": (20.0, 30.0),
}
SNR_TIER_NAMES = list(SNR_TIERS.keys())
SNR_TIER_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

NOISE_TYPES = ["white", "ambient"]
NOISE_TYPE_WEIGHTS = [0.5, 0.5]

# --------------------------------------------------------------------------- #
# Benign variants. All are legitimately "not communication", but they cover
# different broadband/high-frequency energy profiles so the model does not
# learn "any ultrasonic energy = threat" (Task 3).
#   ambient           - realistic room/office noise (pink + hum + spikes)
#   ambient_hf_boost  - ambient noise with extra broadband energy mixed in,
#                        at a random SNR, to simulate a noisier environment
#                        with strong non-communication high-frequency content
#   white             - pure broadband hiss, no structure at all
# --------------------------------------------------------------------------- #
BENIGN_VARIANTS = [
    "steady_ultrasonic_hum",
    "band_limited_hiss",
    "ambient_hf_boost",
    "ambient",
    "white",
]
BENIGN_VARIANT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
BENIGN_HF_BOOST_SNR_RANGE = (-5.0, 25.0)
BENIGN_ULTRASONIC_TONE_RANGE = (18200.0, DETECTION_BAND_HIGH)
BENIGN_STEADY_TONE_AMP_RANGE = (0.25, 0.55)
BENIGN_HISS_AMP_RANGE = (0.35, 0.75)

# Regenerate samples that collapse to the preprocessor squelch fingerprint.
MAX_SAMPLE_REGEN_ATTEMPTS = 12
COLLAPSED_RMS_FEATURE_INDEX = 20  # rms_amplitude in FeatureExtractor order

# Benign crop retries: pick the loudest post-preprocess window from several
# random offsets to reduce identical all-zero feature rows without faking
# communication structure.
BENIGN_CROP_ATTEMPTS = 5

# --------------------------------------------------------------------------- #
# Validation thresholds
# --------------------------------------------------------------------------- #
MIN_RMS_AMPLITUDE = 1e-6   # below this, treat a sample as suspiciously silent
MAX_EXACT_FEATURE_DUPLICATES = 0
MAX_CROSS_SPLIT_EXACT_FEATURE_MATCHES = 0

# AudioPreprocessor.normalize() (dsp/audio_preprocessing.py, frozen/real code)
# explicitly returns an all-zero chunk when the post-bandpass peak is below
# this floor ("Returns zeros if chunk is silent or out-of-band suppressed").
# Traced from the actual code, not documented anywhere else. Dataset V2 uses
# this to decide when a communication-class crop window landed on a
# genuinely inaudible slice (e.g. an OOK "off" bit at high SNR) and should
# be re-cropped rather than silently kept as a degenerate all-zero row.
AUDIO_PREPROCESSOR_SQUELCH_FLOOR = 0.05
MAX_CROP_RESAMPLE_ATTEMPTS = 20
