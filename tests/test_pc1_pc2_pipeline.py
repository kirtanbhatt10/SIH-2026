"""
PC1 → PC2 pipeline tests — simulator, DSP, and backend API integration.

Uses real DSPPipeline (not fake ThreatEvents).
"""

from __future__ import annotations

import os
import sys
import uuid
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.main import app
from backend.services.payload_service import (
    decode_signal_to_text,
    encode_text_to_signal,
    save_signal_to_wav,
)
from backend.services.threat_service import _threats
from dsp import DSPPipeline  # noqa: E402
from integration.backend_client import BackendThreatClient  # noqa: E402

CHUNK = 2048
TEST_PAYLOAD = "SIH_PC1_PC2_TEST"


@pytest.fixture(autouse=True)
def clear_threats():
    _threats.clear()
    yield
    _threats.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def generated_wav(tmp_path):
    payload_id = str(uuid.uuid4())[:8]
    signal = encode_text_to_signal(TEST_PAYLOAD)
    path = os.path.join(tmp_path, f"{payload_id}.wav")
    save_signal_to_wav(path, signal, SAMPLE_RATE)
    return path, payload_id, signal


def _run_dsp_on_wav(path: str) -> dict:
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


class TestBackendHealth:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_system_status(self, client):
        r = client.get("/api/system-status")
        assert r.status_code == 200
        assert r.json()["status"] == "online"


class TestSimulatorGenerate:
    def test_simulator_generate_api(self, client):
        r = client.post(
            "/simulate/generate",
            json={
                "text": TEST_PAYLOAD,
                "freq_0_hz": FREQ_0,
                "freq_1_hz": FREQ_1,
                "bit_duration_ms": BIT_DURATION * 1000,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "payload_id" in data
        assert data["duration_sec"] > 0
        assert data["bit_count"] > 0


class TestWavValidation:
    def test_wav_exists(self, generated_wav):
        path, _, _ = generated_wav
        assert os.path.isfile(path)

    def test_wav_non_zero(self, generated_wav):
        path, _, signal = generated_wav
        assert np.max(np.abs(signal)) > 1e-6

    def test_wav_format_valid(self, generated_wav):
        path, _, _ = generated_wav
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0


class TestSimulatorRoundTrip:
    def test_encode_decode_roundtrip(self):
        signal = encode_text_to_signal(TEST_PAYLOAD)
        result = decode_signal_to_text(signal)
        assert result.success is True
        assert result.text == TEST_PAYLOAD


class TestAiDspPipeline:
    def test_dsp_processes_wav(self, generated_wav):
        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        assert event["schema_version"] == "1.0.0-dsp"
        assert event["chunks_analyzed"] > 0

    def test_threat_event_generated(self, generated_wav):
        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        required = {
            "detected",
            "confidence",
            "risk",
            "suspicion_score",
            "pattern",
            "snr",
            "chunks_analyzed",
            "timestamp",
            "schema_version",
        }
        assert required.issubset(event.keys())


class TestBackendIntegration:
    def test_post_analyze(self, client, generated_wav):
        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        r = client.post("/api/analyze", json=event)
        assert r.status_code == 200
        assert r.json()["status"] == "received"

    def test_get_threats_current(self, client, generated_wav):
        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        client.post("/api/analyze", json=event)
        r = client.get("/api/threats/current")
        assert r.status_code == 200
        current = r.json()["current"]
        assert current is not None
        assert current == event

    def test_get_threats(self, client, generated_wav):
        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        client.post("/api/analyze", json=event)
        r = client.get("/api/threats")
        assert r.status_code == 200
        threats = r.json()["threats"]
        assert len(threats) >= 1
        assert any(t == event for t in threats)

    def test_backend_client_live_roundtrip(self, generated_wav):
        import httpx

        path, _, _ = generated_wav
        event = _run_dsp_on_wav(path)
        test_client = TestClient(app)

        def post_fn(url: str, payload: dict) -> httpx.Response:
            path_only = "/" + url.split("/", 3)[-1] if "://" in url else url
            if not path_only.startswith("/"):
                path_only = "/api/analyze"
            response = test_client.post(path_only, json=payload)
            return httpx.Response(
                response.status_code,
                request=httpx.Request("POST", url),
                content=response.content,
                headers=dict(response.headers),
            )

        bc = BackendThreatClient(base_url="http://test.local", post_fn=post_fn)
        result = bc.submit_threat_event(event)
        assert result.success is True

        current_r = test_client.get("/api/threats/current")
        assert current_r.status_code == 200
        assert current_r.json()["current"] == event


class TestReferenceSample:
    def test_backend2_sample_dsp_detects(self):
        sample = os.path.join(_REPO_ROOT, "samples", "backend2", "sample.wav")
        if not os.path.isfile(sample):
            pytest.skip("Reference sample missing")
        event = _run_dsp_on_wav(sample)
        assert event["detected"] is True
