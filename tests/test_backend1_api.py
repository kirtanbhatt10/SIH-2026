"""Tests for Backend 1 API endpoints.

Verifies all threat management endpoints work correctly:
- GET  /api/system-status
- POST /api/analyze
- GET  /api/threats
- GET  /api/threats/current
- GET  /docs (Swagger)
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.threat_service import _threats


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_threats():
    """Clear threat state before each test."""
    _threats.clear()
    yield
    _threats.clear()



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


class TestAnalyze:
    VALID_EVENT = {
        "detected": True,
        "confidence": 0.94,
        "risk": "HIGH",
        "frequency": {"min": 19800, "max": 21200},
        "duration": 3.2,
        "pattern": "FSK-like",
    }

    def test_analyze_accepts_valid_event(self):
        response = client.post("/api/analyze", json=self.VALID_EVENT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["event"]["detected"] is True
        assert data["event"]["confidence"] == 0.94

    def test_analyze_rejects_missing_fields(self):
        response = client.post("/api/analyze", json={"detected": True})
        assert response.status_code == 422  # Pydantic validation error

    def test_analyze_rejects_invalid_confidence(self):
        bad_event = {**self.VALID_EVENT, "confidence": 1.5}
        response = client.post("/api/analyze", json=bad_event)
        assert response.status_code == 422

    def test_analyze_rejects_negative_duration(self):
        bad_event = {**self.VALID_EVENT, "duration": -1.0}
        response = client.post("/api/analyze", json=bad_event)
        assert response.status_code == 422

    def test_analyze_stores_event(self):
        client.post("/api/analyze", json=self.VALID_EVENT)
        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 1
        assert threats[0]["pattern"] == "FSK-like"


class TestThreats:
    VALID_EVENT = {
        "detected": True,
        "confidence": 0.8,
        "risk": "MEDIUM",
        "frequency": {"min": 18000, "max": 20000},
        "duration": 2.0,
        "pattern": "tone-burst",
    }

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
        client.post("/api/analyze", json=self.VALID_EVENT)
        response = client.get("/api/threats")
        threats = response.json()["threats"]
        assert len(threats) == 1
        assert threats[0]["risk"] == "MEDIUM"

    def test_threats_current_returns_latest(self):
        # Post two events
        client.post("/api/analyze", json=self.VALID_EVENT)
        second_event = {**self.VALID_EVENT, "risk": "CRITICAL", "confidence": 0.99}
        client.post("/api/analyze", json=second_event)

        response = client.get("/api/threats/current")
        current = response.json()["current"]
        assert current["risk"] == "CRITICAL"
        assert current["confidence"] == 0.99

    def test_threats_preserves_order(self):
        for i in range(3):
            event = {**self.VALID_EVENT, "confidence": round((i + 1) * 0.1, 2)}
            client.post("/api/analyze", json=event)

        response = client.get("/api/threats")
        threats = response.json()["threats"]
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
