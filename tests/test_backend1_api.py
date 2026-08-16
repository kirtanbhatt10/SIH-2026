"""Tests for Backend 1 API endpoints.

Verifies all threat management endpoints work correctly:
- GET  /api/system-status
- POST /api/analyze
- GET  /api/threats
- GET  /api/threats/current
- GET  /docs (Swagger)

ThreatEvent contract: 13-field AI schema (1.0.0-dsp).
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

from backend.main import app
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

NO_THREAT_EVENT = {
    "schema_version": "1.0.0-dsp",
    "detected": False,
    "confidence": 0.0,
    "risk": "LOW",
    "suspicion_score": 0.0,
    "frequency_start": None,
    "frequency_end": None,
    "carrier_freqs": [],
    "duration": 0.0,
    "pattern": "none",
    "snr": 0.0,
    "chunks_analyzed": 0,
    "timestamp": 1786621450.25,
}

THREAT_EVENT_FIELDS = frozenset(VALID_THREAT_EVENT.keys())


@pytest.fixture(autouse=True)
def clear_threats():
    """Clear threat state before each test."""
    _threats.clear()
    yield
    _threats.clear()


def _post(event: dict):
    return client.post("/api/analyze", json=event)


class TestSystemStatus:
    def test_system_status_returns_200(self):
        response = client.get("/api/system-status")
        assert response.status_code == 200

    def test_system_status_fields(self):
        response = client.get("/api/system-status")
        data = response.json()
        assert data["status"] == "online"
        assert data["service"] == "acoustic-shield-backend"
        assert "version" in data


class TestThreatEventValidation:
    """Contract validation for POST /api/analyze (13-field ThreatEvent)."""

    def test_valid_full_13_field_event(self):
        response = _post(VALID_THREAT_EVENT)
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_valid_no_threat_event(self):
        response = _post(NO_THREAT_EVENT)
        assert response.status_code == 200
        assert response.json()["event"]["detected"] is False
        assert response.json()["event"]["carrier_freqs"] == []

    def test_confidence_zero(self):
        event = {**VALID_THREAT_EVENT, "confidence": 0.0}
        assert _post(event).status_code == 200

    def test_confidence_one(self):
        event = {**VALID_THREAT_EVENT, "confidence": 1.0}
        assert _post(event).status_code == 200

    def test_suspicion_score_zero(self):
        event = {**VALID_THREAT_EVENT, "suspicion_score": 0.0}
        assert _post(event).status_code == 200

    def test_suspicion_score_one(self):
        event = {**VALID_THREAT_EVENT, "suspicion_score": 1.0}
        assert _post(event).status_code == 200

    def test_null_frequency_start(self):
        event = {**VALID_THREAT_EVENT, "frequency_start": None}
        assert _post(event).status_code == 200

    def test_null_frequency_end(self):
        event = {**VALID_THREAT_EVENT, "frequency_end": None}
        assert _post(event).status_code == 200

    def test_empty_carrier_freqs(self):
        event = {**NO_THREAT_EVENT, "detected": True, "pattern": "tone"}
        assert _post(event).status_code == 200

    def test_multiple_carrier_frequencies(self):
        event = {**VALID_THREAT_EVENT, "carrier_freqs": [18500.0, 19000.0, 20500.0]}
        response = _post(event)
        assert response.status_code == 200
        assert response.json()["event"]["carrier_freqs"] == [18500.0, 19000.0, 20500.0]

    def test_invalid_confidence(self):
        event = {**VALID_THREAT_EVENT, "confidence": 1.5}
        assert _post(event).status_code == 422

    def test_invalid_suspicion_score(self):
        event = {**VALID_THREAT_EVENT, "suspicion_score": -0.1}
        assert _post(event).status_code == 422

    def test_invalid_risk(self):
        event = {**VALID_THREAT_EVENT, "risk": "CRITICAL"}
        assert _post(event).status_code == 422

    def test_invalid_pattern(self):
        event = {**VALID_THREAT_EVENT, "pattern": "FSK-like"}
        assert _post(event).status_code == 422

    def test_negative_duration(self):
        event = {**VALID_THREAT_EVENT, "duration": -1.0}
        assert _post(event).status_code == 422

    def test_negative_chunks_analyzed(self):
        event = {**VALID_THREAT_EVENT, "chunks_analyzed": -1}
        assert _post(event).status_code == 422

    def test_negative_frequency_start(self):
        event = {**VALID_THREAT_EVENT, "frequency_start": -100.0}
        assert _post(event).status_code == 422

    def test_negative_frequency_end(self):
        event = {**VALID_THREAT_EVENT, "frequency_end": -50.0}
        assert _post(event).status_code == 422

    def test_negative_carrier_freq(self):
        event = {**VALID_THREAT_EVENT, "carrier_freqs": [19000.0, -1.0]}
        assert _post(event).status_code == 422

    def test_missing_required_fields(self):
        assert _post({"detected": True}).status_code == 422


class TestThreatEventStorage:
    def test_stores_values_without_recalculation(self):
        _post(VALID_THREAT_EVENT)
        stored = client.get("/api/threats/current").json()["current"]
        assert stored == VALID_THREAT_EVENT
        assert stored["confidence"] == 0.94
        assert stored["suspicion_score"] == 0.91
        assert stored["confidence"] != stored["suspicion_score"]

    def test_current_returns_all_13_fields_unchanged(self):
        _post(VALID_THREAT_EVENT)
        current = client.get("/api/threats/current").json()["current"]
        assert frozenset(current.keys()) == THREAT_EVENT_FIELDS
        for key, value in VALID_THREAT_EVENT.items():
            assert current[key] == value

    def test_analyze_accepts_ai_generated_payload(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ai_root = os.path.join(repo_root, "ai")
        if ai_root not in sys.path:
            sys.path.insert(0, ai_root)

        from dsp import DSPPipeline  # noqa: E402

        pipeline = DSPPipeline()
        audio = pipeline.generate_attack_signal("fsk", duration_sec=2.0)
        chunk = 2048
        for i in range(0, len(audio) - chunk, chunk):
            pipeline.process(audio[i : i + chunk])
        ai_event = pipeline.to_threat_event()

        response = _post(ai_event)
        assert response.status_code == 200, response.json()
        assert response.json()["event"]["schema_version"] == "1.0.0-dsp"
        assert frozenset(response.json()["event"].keys()) == THREAT_EVENT_FIELDS

        current = client.get("/api/threats/current").json()["current"]
        assert current["detected"] == ai_event["detected"]
        assert current["confidence"] == ai_event["confidence"]
        assert current["suspicion_score"] == ai_event["suspicion_score"]
        assert current["carrier_freqs"] == ai_event["carrier_freqs"]


class TestThreats:
    def test_threats_empty_initially(self):
        response = client.get("/api/threats")
        assert response.status_code == 200
        assert response.json()["threats"] == []

    def test_threats_current_empty_initially(self):
        response = client.get("/api/threats/current")
        assert response.status_code == 200
        data = response.json()
        assert data["current"] is None
        assert "message" in data

    def test_threats_returns_stored_events(self):
        _post(VALID_THREAT_EVENT)
        threats = client.get("/api/threats").json()["threats"]
        assert len(threats) == 1
        assert threats[0]["risk"] == "HIGH"

    def test_threats_current_returns_latest(self):
        _post(VALID_THREAT_EVENT)
        second = {**VALID_THREAT_EVENT, "risk": "MEDIUM", "confidence": 0.5, "suspicion_score": 0.55}
        _post(second)

        current = client.get("/api/threats/current").json()["current"]
        assert current["risk"] == "MEDIUM"
        assert current["confidence"] == 0.5
        assert current["suspicion_score"] == 0.55

    def test_threats_preserves_order(self):
        for i in range(3):
            event = {**VALID_THREAT_EVENT, "confidence": round((i + 1) * 0.1, 2)}
            _post(event)

        threats = client.get("/api/threats").json()["threats"]
        assert len(threats) == 3
        assert threats[0]["confidence"] == pytest.approx(0.1)
        assert threats[2]["confidence"] == pytest.approx(0.3)


class TestSwaggerDocs:
    def test_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        paths = list(data["paths"].keys())
        assert "/api/system-status" in paths
        assert "/api/analyze" in paths
        assert "/api/threats" in paths
        assert "/api/threats/current" in paths


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_fields(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "sample_rate" in data
        assert "freq_0" in data
        assert "freq_1" in data
