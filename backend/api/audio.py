import asyncio
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from backend.api.system_status import get_active_streamer, set_active_streamer
from backend.core.config import STREAM_CONFIG
from backend.services.audio_service import AudioStreamer, FileReplaySource, LiveMicSource, list_input_devices

router = APIRouter(prefix="/stream", tags=["audio"])

_stream_task: asyncio.Task | None = None


@router.websocket("/audio")
async def stream_audio(ws: WebSocket):
    await ws.accept()
    streamer = get_active_streamer()
    if streamer is None:
        await ws.close(code=1011, reason="No active stream - start one via /stream/start")
        return
    streamer.subscribers.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if streamer and ws in streamer.subscribers:
            streamer.subscribers.remove(ws)


@router.websocket("/forensics")
async def stream_forensics(ws: WebSocket):
    await ws.accept()
    streamer = get_active_streamer()
    if streamer is None:
        await ws.close(code=1011, reason="No active stream - start one via /stream/start")
        return
    streamer.forensic_subscribers.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if streamer and ws in streamer.forensic_subscribers:
            streamer.forensic_subscribers.remove(ws)


@router.get("/devices")
def list_devices():
    return list_input_devices()


@router.post("/start")
async def start_stream(
    source: str = Query("mic", description="'mic' or 'file'"),
    file_path: Optional[str] = Query(None, description="Path to WAV file if source is 'file'"),
    device_id: Optional[int] = Query(None, description="Optional audio device ID for mic"),
):
    global _stream_task

    streamer = get_active_streamer()
    if streamer:
        streamer.stop()

    if source == "file":
        if not file_path:
            return {"error": "file_path parameter is required when source='file'"}
        src = FileReplaySource(file_path)
    else:
        src = LiveMicSource(STREAM_CONFIG.sample_rate, device=device_id)

    new_streamer = AudioStreamer(src)
    set_active_streamer(new_streamer)
    _stream_task = asyncio.create_task(new_streamer.run())

    return {
        "status": "started",
        "stream_id": new_streamer.stream_id,
        "source": source,
        "sample_rate": STREAM_CONFIG.sample_rate,
        "window_sec": STREAM_CONFIG.window_sec,
    }


@router.post("/stop")
async def stop_stream():
    streamer = get_active_streamer()
    if streamer:
        streamer.stop()
        set_active_streamer(None)
        return {"status": "stopped"}
    return {"status": "no active stream"}


@router.get("/status")
def stream_status():
    streamer = get_active_streamer()
    if streamer:
        return {
            "active": True,
            "stream_id": streamer.stream_id,
            "subscribers": len(streamer.subscribers),
            "forensic_subscribers": len(streamer.forensic_subscribers),
            "latest_forensics": [e.model_dump() for e in streamer.latest_forensic_events[-5:]],
        }
    return {"active": False, "subscribers": 0, "latest_forensics": []}
