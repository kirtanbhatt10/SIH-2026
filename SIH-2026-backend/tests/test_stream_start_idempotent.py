"""POST /stream/start must not tear down an active mic stream on duplicate calls."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


def test_duplicate_mic_start_reuses_active_stream():
    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
            first = await client.post("/stream/start?source=mic&device_id=1")
            if first.status_code != 200:
                pytest.skip(f"mic stream unavailable: {first.text}")
            second = await client.post("/stream/start?source=mic&device_id=1")
            assert second.status_code == 200
            assert first.json()["stream_id"] == second.json()["stream_id"]
            status = await client.get("/stream/status")
            assert status.json()["active"] is True
            await client.post("/stream/stop")

    asyncio.run(_run())
