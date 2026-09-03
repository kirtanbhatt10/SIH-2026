"""Integration tests: simulator virtual feed -> real ThreatEvent."""

import asyncio
import os

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.config import SAMPLE_RATE, STREAM_CONFIG
from backend.main import app
from backend.services.monitoring_pipeline import MonitoringPipeline
from backend.services.payload_service import encode_text_to_signal
from backend.services.threat_service import get_threats


@pytest.fixture
def sample_wav():
    path = os.path.abspath(os.path.join("samples", "backend2", "sample.wav"))
    if not os.path.isfile(path):
        pytest.skip("sample.wav not found")
    return path


def test_direct_pipeline_generated_signal_produces_threat():
    async def _run():
        signal = encode_text_to_signal("SIH2026", freq_0=18500, freq_1=20500)
        before = len(get_threats())
        pipe = MonitoringPipeline(sample_rate=SAMPLE_RATE)
        await pipe.start()
        pipe.reset_accumulator()
        win = int(SAMPLE_RATE * STREAM_CONFIG.window_sec)
        hop = int(SAMPLE_RATE * STREAM_CONFIG.hop_sec)
        for start in range(0, max(1, len(signal) - win), hop):
            chunk = signal[start : start + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            await pipe.process_chunk(chunk.astype(np.float32))
        await pipe.stop()
        after = len(get_threats())
        assert after > before
        threat = get_threats()[-1]
        assert threat.detected is True
        assert threat.risk in ("MEDIUM", "HIGH")
        assert any(18000 < f < 21000 for f in threat.carrier_freqs)

    asyncio.run(_run())


def test_direct_pipeline_ambient_noise_no_threat():
    async def _run():
        before = len(get_threats())
        pipe = MonitoringPipeline(sample_rate=SAMPLE_RATE)
        await pipe.start()
        rng = np.random.default_rng(42)
        win = int(SAMPLE_RATE * STREAM_CONFIG.window_sec)
        hop = int(SAMPLE_RATE * STREAM_CONFIG.hop_sec)
        noise = rng.normal(0, 0.01, win * 4).astype(np.float32)
        for start in range(0, len(noise) - win, hop):
            await pipe.process_chunk(noise[start : start + win])
        await pipe.stop()
        assert len(get_threats()) == before

    asyncio.run(_run())


def test_virtual_simulator_api_generates_threat(sample_wav):
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
            await client.post(f"/stream/start?source=file&file_path={sample_wav}")
            await asyncio.sleep(0.5)
            gen = await client.post(
                "/simulate/generate",
                json={
                    "text": "SIH2026",
                    "freq_0_hz": 18500,
                    "freq_1_hz": 20500,
                    "bit_duration_ms": 50,
                },
            )
            assert gen.status_code == 200
            payload_id = gen.json()["payload_id"]
            tr = await client.post("/simulate/transmit", json={"payload_id": payload_id, "mode": "virtual"})
            assert tr.status_code == 200
            body = tr.json()
            assert body["state"] == "complete"
            assert body["threats_after"] > body["threats_before"]
            assert body["dsp_detected"] is True

            current = await client.get("/api/threats/current")
            assert current.json()["current"] is not None
            assert current.json()["current"]["detected"] is True

            await client.post("/stream/stop")

    asyncio.run(_run())


def test_simulator_diagnostics_endpoint():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/simulate/diagnostics")
            assert r.status_code == 200
            body = r.json()
            assert body["backend_online"] is True
            assert "pipeline_object_id" in body
            assert "mic_device" in body

    asyncio.run(_run())
