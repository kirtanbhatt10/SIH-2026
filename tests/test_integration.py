"""End-to-end integration test.

Tests the full flow: simulator generates payload → detect characteristics →
submit to /api/analyze → verify via /api/threats.

NOTE: The AI/detector components are not yet implemented, so this test uses
a clearly marked stub to bridge the gap. The stub does NOT fabricate real
AI measurements — it constructs a ThreatEvent from known simulator metadata.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.payload_service import encode_text_to_signal, decode_signal_to_text
from backend.services.threat_service import _threats
from backend.core.config import FREQ_0, FREQ_1, SAMPLE_RATE


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_threats():
    """Clear threat state before each test."""
    _threats.clear()
    yield
    _threats.clear()



def _stub_ai_detection(decode_result, freq_0: int, freq_1: int) -> dict:
    """STUB: Simulates what the AI/detector pipeline would eventually produce.

    This is NOT real AI detection. It constructs a ThreatEvent from known
    simulator parameters for integration testing purposes only.

    This stub will be replaced when the AI team delivers their detection
    specification (see docs/integration-contract.md Section 5).
    """
    return {
        "detected": decode_result.success,
        "confidence": decode_result.confidence,
        "risk": "HIGH" if decode_result.confidence > 0.7 else "LOW",
        "frequency": {
            "min": float(min(freq_0, freq_1)),
            "max": float(max(freq_0, freq_1)),
        },
        "duration": decode_result.bit_count * 0.05,  # bit_duration default
        "pattern": "BFSK",
    }


class TestEndToEndIntegration:
    """Full pipeline: simulator → (stub detection) → backend API → threat storage."""

    def test_full_pipeline(self):
        # Step 1: Simulator generates encoded signal
        payload_text = "INTEGRATION TEST"
        signal = encode_text_to_signal(payload_text)

        # Step 2: Decoder recovers the payload (simulates detector + partial AI)
        decode_result = decode_signal_to_text(signal)
        assert decode_result.success is True
        assert decode_result.text == payload_text

        # Step 3: Stub AI produces a ThreatEvent
        threat_event = _stub_ai_detection(decode_result, FREQ_0, FREQ_1)

        # Step 4: Submit to Backend 1's /api/analyze
        response = client.post("/api/analyze", json=threat_event)
        assert response.status_code == 200
        assert response.json()["status"] == "received"

        # Step 5: Verify threat is stored and retrievable
        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 1
        assert threats[0]["detected"] is True
        assert threats[0]["pattern"] == "BFSK"

        # Step 6: Verify /api/threats/current returns this threat
        response = client.get("/api/threats/current")
        current = response.json()["current"]
        assert current is not None
        assert current["frequency"]["min"] == float(FREQ_0)
        assert current["frequency"]["max"] == float(FREQ_1)

    def test_pipeline_with_different_payloads(self):
        """Run the pipeline with multiple distinct payloads."""
        payloads = ["SIH2026", "COVERT", "EXFIL"]

        for text in payloads:
            signal = encode_text_to_signal(text)
            result = decode_signal_to_text(signal)
            assert result.success is True
            assert result.text == text

            threat = _stub_ai_detection(result, FREQ_0, FREQ_1)
            response = client.post("/api/analyze", json=threat)
            assert response.status_code == 200

        # All three should be stored
        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 3

    def test_simulator_api_generate(self):
        """Test the simulator REST API directly."""
        response = client.post(
            "/simulate/generate",
            json={"text": "API TEST", "freq_0_hz": 18500, "freq_1_hz": 21000, "bit_duration_ms": 50.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "payload_id" in data
        assert data["bit_count"] > 0
        assert data["duration_sec"] > 0

    def test_system_status_during_operation(self):
        """System status should remain online during active operation."""
        # Generate a payload
        client.post(
            "/simulate/generate",
            json={"text": "STATUS CHECK"},
        )

        # System status should still be online
        response = client.get("/api/system-status")
        assert response.status_code == 200
        assert response.json()["status"] == "online"

        # Health should also be good
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
