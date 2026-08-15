"""
Acoustic Cybersecurity System — DSP Module
================================
Ultrasonic signal processing pipeline for covert acoustic exfiltration detection.

Modules:
    audio_preprocessing  — Clean raw audio, isolate ultrasonic band
    feature_extraction   — Extract 32 ML-ready features per audio chunk
    spectrogram_generator— Generate spectrograms for visualization & analysis
    frequency_analyzer   — Detect peaks, classify modulation schemes (FSK/OOK/Chirp)
    segmentation         — Onset/offset detection, event durations
    stft                 — Explicit STFT with configurable overlap
    recording_quality    — Rejects dead microphone captures
    utils                — Synthetic test signal generators
    dsp_api              — Single-call wrapper for Backend integration

Author: AI 1 — DSP / Signal Processing Lead
"""

from .audio_preprocessing import AudioPreprocessor
from .feature_extraction import FeatureExtractor
from .spectrogram_generator import SpectrogramGenerator
from .frequency_analyzer import FrequencyAnalyzer
from .utils import SignalGenerator
from .segmentation import segment_signal, StreamingSegmenter
from .stft import stft, istft, spectrogram_db

# ── Shared Constants (single source of truth for the whole team) ──────────
SAMPLE_RATE = 48000          # Hz — required for 24 kHz Nyquist ceiling
CHUNK_SIZE = 2048            # samples per processing frame (~42.7 ms)
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 2048            # samples between successive spectrogram frames.
                             # Equals CHUNK_SIZE: SpectrogramGenerator appends
                             # one FFT frame per incoming chunk, so frames do
                             # not overlap. INTEGRATION_GUIDE.md previously
                             # documented 512, which existed nowhere in code.

# ── Detection band ───────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH. These were previously contradicted in three places:
#   INTEGRATION_GUIDE.md      said 18-22 kHz, FILTER_HIGH 23000, HOP 512
#   dsp/__init__.py           said FILTER_HIGH 21500
#   audio_preprocessing.py    docstring said 17.5-23 kHz, default 21000
# Backend 2 building a simulator at 21.5 kHz would have gone undetected.
ULTRASONIC_LOW = 18000.0     # Hz — lower edge of detection band
ULTRASONIC_HIGH = 21000.0    # Hz — upper edge of detection band
FILTER_LOW = 17500.0         # Hz — bandpass lower cutoff (500 Hz margin)
FILTER_HIGH = 21500.0        # Hz — bandpass upper cutoff (500 Hz margin)


def __getattr__(name):
    if name == "DSPPipeline":
        from .dsp_api import DSPPipeline
        return DSPPipeline

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "AudioPreprocessor",
    "FeatureExtractor",
    "SpectrogramGenerator",
    "FrequencyAnalyzer",
    "SignalGenerator",
    "segment_signal",
    "StreamingSegmenter",
    "stft",
    "istft",
    "spectrogram_db",
    "DSPPipeline",
    "SAMPLE_RATE",
    "CHUNK_SIZE",
    "N_FFT",
    "HOP_LENGTH",
    "ULTRASONIC_LOW",
    "ULTRASONIC_HIGH",
    "FILTER_LOW",
    "FILTER_HIGH",
]
