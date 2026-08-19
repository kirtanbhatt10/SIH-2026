"""Tests for AI/DSP → Backend 1 HTTP integration client."""

from __future__ import annotations

import httpx
import pytest

from integration.backend_client import ANALYZE_PATH, BackendThreatClient, SubmitResult

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


def _make_response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", f"http://test.local{ANALYZE_PATH}")
    content = b""
    headers = {"content-type": "application/json"}
    if json_body is not None:
        import json

        content = json.dumps(json_body).encode("utf-8")
    return httpx.Response(status_code, request=request, content=content, headers=headers)


class TestBackendThreatClient:
    def test_valid_threat_event_is_posted_to_analyze_endpoint(self):
        posted: list[tuple[str, dict]] = []

        def fake_post(url: str, payload: dict) -> httpx.Response:
            posted.append((url, payload))
            return _make_response(200, {"status": "received", "event": payload})

        client = BackendThreatClient(base_url="http://test.local", post_fn=fake_post)
        result = client.submit_threat_event(VALID_THREAT_EVENT)

        assert result.success is True
        assert result.stored is True
        assert len(posted) == 1
        assert posted[0][0] == f"http://test.local{ANALYZE_PATH}"

    def test_exact_13_field_payload_preserved(self):
        captured: dict = {}

        def fake_post(url: str, payload: dict) -> httpx.Response:
            captured.update(payload)
            return _make_response(200, {"status": "received", "event": payload})

        client = BackendThreatClient(base_url="http://test.local", post_fn=fake_post)
        client.submit_threat_event(VALID_THREAT_EVENT)

        assert captured == VALID_THREAT_EVENT

    def test_no_payload_mutation(self):
        original = dict(VALID_THREAT_EVENT)
        snapshot = dict(original)

        def fake_post(url: str, payload: dict) -> httpx.Response:
            return _make_response(200, {"status": "received", "event": payload})

        BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            original
        )
        assert original == snapshot

    def test_backend_success_response_handled(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            return _make_response(200, {"status": "received", "event": payload})

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.status_code == 200
        assert result.response_body["status"] == "received"
        assert result.error is None

    def test_backend_422_handled(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            return _make_response(422, {"detail": [{"msg": "validation error"}]})

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.success is False
        assert result.stored is False
        assert result.status_code == 422
        assert "422" in (result.error or "")

    def test_backend_500_handled(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            return _make_response(500, {"detail": "internal error"})

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.success is False
        assert result.status_code == 500

    def test_backend_unavailable_handled(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.success is False
        assert result.stored is False
        assert result.status_code is None
        assert "unavailable" in (result.error or "").lower()

    def test_backend_timeout_handled(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=httpx.Request("POST", url))

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_invalid_success_body_not_treated_as_stored(self):
        def fake_post(url: str, payload: dict) -> httpx.Response:
            return _make_response(200, {"status": "ok"})

        result = BackendThreatClient(base_url="http://test.local", post_fn=fake_post).submit_threat_event(
            VALID_THREAT_EVENT
        )
        assert result.success is False
        assert result.stored is False


class TestBackendThreatClientLive:
    """In-process Backend 1 via TestClient (no separate server)."""

    @pytest.fixture
    def live_client(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.services.threat_service import _threats

        _threats.clear()
        test_client = TestClient(app)

        def post_fn(url: str, payload: dict) -> httpx.Response:
            # Map integration client URL to TestClient path
            path = url.split("://", 1)[-1]
            if "/" in path:
                path = "/" + path.split("/", 1)[1]
            else:
                path = ANALYZE_PATH
            response = test_client.post(path, json=payload)
            return httpx.Response(
                response.status_code,
                request=httpx.Request("POST", url),
                content=response.content,
                headers=dict(response.headers),
            )

        yield BackendThreatClient(base_url="http://test.local", post_fn=post_fn)
        _threats.clear()

    def test_dsp_to_backend_round_trip(self, live_client):
        import os
        import sys

        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ai_root = os.path.join(repo, "ai")
        if ai_root not in sys.path:
            sys.path.insert(0, ai_root)

        from dsp import DSPPipeline

        pipeline = DSPPipeline()
        audio = pipeline.generate_attack_signal("fsk", duration_sec=2.0)
        chunk = 2048
        for i in range(0, len(audio) - chunk, chunk):
            pipeline.process(audio[i : i + chunk])
        event = pipeline.to_threat_event()

        result = live_client.submit_threat_event(event)
        assert result.success is True, result.error

        from backend.services.threat_service import get_latest_threat

        stored = get_latest_threat()
        assert stored is not None
        assert stored.model_dump() == event
