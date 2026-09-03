import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

DEVICE_ENUM_TIMEOUT_SEC = 5.0

from backend.api.system_status import get_active_streamer, set_active_streamer
from backend.core.config import MIC_DEVICE_ID, STREAM_CONFIG
from backend.services.stream_manager import get_stream_task, set_stream_task
from backend.services.audio_service import AudioStreamer, FileReplaySource, LiveMicSource, list_input_devices

router = APIRouter(prefix="/stream", tags=["audio"])


async def _finish_stream_task() -> None:
    task = get_stream_task()
    if task is not None:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        set_stream_task(None)


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
async def list_devices():
    logger.info("Starting audio input device enumeration")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(list_input_devices),
            timeout=DEVICE_ENUM_TIMEOUT_SEC,
        )
        count = len(result.get("input_devices", []))
        if result.get("error"):
            logger.warning("Device enumeration returned error: %s", result.get("error"))
        else:
            logger.info("Device enumeration succeeded: %d input device(s)", count)
        return result
    except asyncio.TimeoutError:
        logger.warning("Device enumeration timed out after %.1fs", DEVICE_ENUM_TIMEOUT_SEC)
        return {"error": "device enumeration timed out", "input_devices": []}
    except Exception as exc:
        logger.exception("Unexpected error during device enumeration: %s", exc)
        return {"error": str(exc), "input_devices": []}


def _active_mic_stream_response(streamer: AudioStreamer, source: str) -> dict:
    return {
        "status": "started",
        "stream_id": streamer.stream_id,
        "source": source,
        "sample_rate": STREAM_CONFIG.sample_rate,
        "window_sec": STREAM_CONFIG.window_sec,
    }


@router.post("/start")
async def start_stream(
    source: str = Query("mic", description="'mic' or 'file'"),
    file_path: Optional[str] = Query(None, description="Path to WAV file if source is 'file'"),
    device_id: Optional[int] = Query(None, description="Optional audio device ID for mic"),
):
    requested_device = device_id if device_id is not None else MIC_DEVICE_ID
    streamer = get_active_streamer()
    if streamer and streamer.is_running and source == "mic":
        existing_source = streamer.source
        if isinstance(existing_source, LiveMicSource):
            existing_device = existing_source.device if existing_source.device is not None else MIC_DEVICE_ID
            if existing_device == requested_device:
                logger.info(
                    "[STREAM START] reusing active mic stream stream_id=%s subscribers=%d",
                    streamer.stream_id,
                    len(streamer.subscribers),
                )
                return _active_mic_stream_response(streamer, source)

    if streamer:
        streamer.stop()
        set_active_streamer(None)
        await _finish_stream_task()

    if source == "file":
        if not file_path:
            return {"error": "file_path parameter is required when source='file'"}
        src = FileReplaySource(file_path)
    else:
        src = LiveMicSource(STREAM_CONFIG.sample_rate, device=device_id if device_id is not None else MIC_DEVICE_ID)

    new_streamer = AudioStreamer(src)
    set_active_streamer(new_streamer)
    set_stream_task(asyncio.create_task(new_streamer.run()))
    logger.info(
        "[STREAM START] stream_id=%s source=%s streamer_id=%s monitoring_id=pending",
        new_streamer.stream_id,
        source,
        id(new_streamer),
    )

    return _active_mic_stream_response(new_streamer, source)


@router.post("/stop")
async def stop_stream():
    streamer = get_active_streamer()
    if streamer:
        streamer.stop()
        set_active_streamer(None)
        await _finish_stream_task()
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
