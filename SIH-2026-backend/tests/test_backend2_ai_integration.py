"""Verify Backend 2 reference sample passes the actual AI DSP pipeline."""

import json
import os
import sys

import numpy as np
import pytest
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from dsp import DSPPipeline  # noqa: E402

SAMPLE_WAV = os.path.join(_REPO_ROOT, "samples", "backend2", "sample.wav")
CHUNK = 2048


def _run_wav_through_dsp(path: str) -> dict:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    pipeline = DSPPipeline(sample_rate=sr)
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pipeline.process(audio[i : i + CHUNK])
    return pipeline.to_threat_event()


@pytest.fixture(scope="module")
def sample_exists():
    if not os.path.isfile(SAMPLE_WAV):
        pytest.skip(f"Reference sample missing: {SAMPLE_WAV}")


class TestBackend2AiIntegration:
    def test_sample_wav_exists(self, sample_exists):
        assert os.path.isfile(SAMPLE_WAV)

    def test_ai_dsp_detects_backend2_sample(self, sample_exists):
        event = _run_wav_through_dsp(SAMPLE_WAV)
        assert event["schema_version"] == "1.0.0-dsp"
        assert event["detected"] is True, json.dumps(event, indent=2)
        assert event["risk"] in ("MEDIUM", "HIGH")
        assert event["suspicion_score"] >= 0.45
        assert event["pattern"] == "fsk"
        assert len(event["carrier_freqs"]) >= 2
        assert event["carrier_freqs"][0] == pytest.approx(18500, abs=250)
        assert event["carrier_freqs"][1] == pytest.approx(20500, abs=250)

    def test_ai_output_is_json_serializable(self, sample_exists):
        event = _run_wav_through_dsp(SAMPLE_WAV)
        json.dumps(event)
