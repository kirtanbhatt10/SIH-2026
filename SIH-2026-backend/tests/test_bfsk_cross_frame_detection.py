"""Cross-frame sequential BFSK carrier aggregation tests."""

from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import pytest
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from dsp import DSPPipeline  # noqa: E402

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.payload_service import encode_text_to_signal, save_signal_to_wav

CHUNK = 2048
SAMPLE_WAV = os.path.join(_REPO_ROOT, "samples", "backend2", "sample.wav")


def _run_audio_through_dsp(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    pipeline = DSPPipeline(sample_rate=sample_rate)
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pipeline.process(audio[i : i + CHUNK])
    return pipeline.to_threat_event()


def _run_wav_through_dsp(path: str) -> dict:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _run_audio_through_dsp(audio, sample_rate=sr)


def _generate_backend2_wav(payload: str = "SIH_PC1_PC2_TEST") -> str:
    signal = encode_text_to_signal(
        payload,
        freq_0=FREQ_0,
        freq_1=FREQ_1,
        bit_duration=BIT_DURATION,
        sample_rate=SAMPLE_RATE,
    )
    os.makedirs(STORAGE_DIR, exist_ok=True)
    wav_path = os.path.join(STORAGE_DIR, f"test_{uuid.uuid4().hex[:8]}.wav")
    save_signal_to_wav(wav_path, signal, SAMPLE_RATE)
    return wav_path


class TestBfskCrossFrameDetection:
    def test_synthetic_fsk_remains_fsk(self):
        pipeline = DSPPipeline()
        audio = pipeline.generate_attack_signal("fsk", duration_sec=3.0)
        event = _run_audio_through_dsp(audio)
        assert event["detected"] is True
        assert event["pattern"] == "fsk"
        assert len(event["carrier_freqs"]) >= 2

    def test_backend2_generated_wav_is_fsk(self):
        wav_path = _generate_backend2_wav()
        try:
            event = _run_wav_through_dsp(wav_path)
            assert event["detected"] is True
            assert event["pattern"] == "fsk"
            assert len(event["carrier_freqs"]) >= 2
            assert event["carrier_freqs"][0] == pytest.approx(18500, abs=250)
            assert event["carrier_freqs"][1] == pytest.approx(20500, abs=250)
        finally:
            if os.path.isfile(wav_path):
                os.remove(wav_path)

    @pytest.mark.parametrize("freq_hz", [18500, 20500])
    def test_single_steady_tone_stays_tone(self, freq_hz: int):
        duration_sec = 2.0
        t = np.arange(int(SAMPLE_RATE * duration_sec)) / SAMPLE_RATE
        audio = np.sin(2 * np.pi * freq_hz * t)
        event = _run_audio_through_dsp(audio)
        assert event["pattern"] == "tone"
        assert len(event["carrier_freqs"]) <= 1

    def test_silence_remains_non_detected(self):
        audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float64)
        event = _run_audio_through_dsp(audio)
        assert event["detected"] is False
        assert event["pattern"] == "none"
        assert event["carrier_freqs"] == []
        assert event["risk"] == "LOW"

    @pytest.mark.skipif(not os.path.isfile(SAMPLE_WAV), reason="reference sample missing")
    def test_backend2_reference_sample_is_fsk(self):
        event = _run_wav_through_dsp(SAMPLE_WAV)
        assert event["detected"] is True
        assert event["pattern"] == "fsk"
        assert len(event["carrier_freqs"]) >= 2
