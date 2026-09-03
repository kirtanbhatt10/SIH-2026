"""Authoritative active stream / monitoring pipeline registry (single instance)."""

from __future__ import annotations

import asyncio
from typing import Optional

from backend.services.audio_service import AudioStreamer

_active_streamer: Optional[AudioStreamer] = None
_stream_task: Optional[asyncio.Task] = None


def get_active_streamer() -> Optional[AudioStreamer]:
    return _active_streamer


def set_active_streamer(streamer: Optional[AudioStreamer]) -> None:
    global _active_streamer
    _active_streamer = streamer


def get_stream_task() -> Optional[asyncio.Task]:
    return _stream_task


def set_stream_task(task: Optional[asyncio.Task]) -> None:
    global _stream_task
    _stream_task = task


def get_active_monitoring():
    """Return the MonitoringPipeline attached to the active stream, if any."""
    streamer = _active_streamer
    if streamer is None or not streamer.is_running:
        return None
    return streamer.monitoring
