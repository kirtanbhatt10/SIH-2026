"""End-to-end integration test.

Tests the full flow: simulator generates payload → decode self-test →
submit 13-field ThreatEvent to /api/analyze → verify via /api/threats.

The ThreatEvent submitted to Backend 1 uses the final ``1.0.0-dsp`` contract.
Simulator encode/decode steps validate Backend 2; threat intake validates Backend 1.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.payload_service import decode_signal_to_text, encode_text_to_signal
from backend.services.threat_service import _threats

client = TestClient(app)

VALID_THREAT_EVENT = {
    "schema_version": "1.0.0-dsp",
    "detected": True,
    "confidence": 0.94,
    "risk": "HIGH",
    "suspicion_score": 0.91,
    "frequency_start": 19800.0,
    "frequency_end": 21200.0,
    "carrier_freqs": [19800.0, 21200.0],
    "duration": 3.2,
    "pattern": "fsk",
    "snr": 18.5,
    "chunks_analyzed": 75,
    "timestamp": 1786621450.25,
}

THREAT_EVENT_FIELDS = frozenset(VALID_THREAT_EVENT.keys())


@pytest.fixture(autouse=True)
def clear_threats():
    """Clear threat state before each test."""
    _threats.clear()
    yield
    _threats.clear()


class TestEndToEndIntegration:
    """Full pipeline: simulator → ThreatEvent → backend API → threat storage."""

    def test_full_pipeline(self):
        # Step 1: Simulator generates encoded signal
        payload_text = "INTEGRATION TEST"
        signal = encode_text_to_signal(payload_text)

        # Step 2: Decoder recovers the payload (Backend 2 self-test)
        decode_result = decode_signal_to_text(signal)
        assert decode_result.success is True
        assert decode_result.text == payload_text

        # Step 3: Submit final 13-field ThreatEvent to Backend 1
        threat_event = dict(VALID_THREAT_EVENT)
        response = client.post("/api/analyze", json=threat_event)
        assert response.status_code == 200
        assert response.json()["status"] == "received"

        # Step 4: Verify threat is stored and retrievable
        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 1
        assert threats[0] == VALID_THREAT_EVENT
        assert threats[0]["detected"] is True
        assert threats[0]["pattern"] == "fsk"
        assert threats[0]["carrier_freqs"] == [19800.0, 21200.0]

        # Step 5: Verify /api/threats/current returns this threat unchanged
        response = client.get("/api/threats/current")
        current = response.json()["current"]
        assert current is not None
        assert frozenset(current.keys()) == THREAT_EVENT_FIELDS
        assert current["frequency_start"] == 19800.0
        assert current["frequency_end"] == 21200.0
        assert current == VALID_THREAT_EVENT

    def test_pipeline_with_different_payloads(self):
        """Run simulator round-trips, then store a ThreatEvent per payload."""
        payloads = ["SIH2026", "COVERT", "EXFIL"]

        for i, text in enumerate(payloads):
            signal = encode_text_to_signal(text)
            result = decode_signal_to_text(signal)
            assert result.success is True
            assert result.text == text

            threat = {**VALID_THREAT_EVENT, "timestamp": VALID_THREAT_EVENT["timestamp"] + i}
            response = client.post("/api/analyze", json=threat)
            assert response.status_code == 200

        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 3
        assert all(t["schema_version"] == "1.0.0-dsp" for t in threats)
        assert all(t["pattern"] == "fsk" for t in threats)

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
        client.post(
            "/simulate/generate",
            json={"text": "STATUS CHECK"},
        )

        response = client.get("/api/system-status")
        assert response.status_code == 200
        assert response.json()["status"] == "online"

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
