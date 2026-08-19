import asyncio
import base64
import collections
import uuid
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import WebSocket

from backend.core.config import STREAM_CONFIG
from backend.models.data_schemas import AudioChunkMessage, DecodedPayloadEvent, StreamConfig
from backend.services.payload_service import decode_signal_to_text


class FileReplaySource:
    def __init__(self, path: str, realtime: bool = True):
        import wave

        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        self.sample_rate = sr
        self.signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.realtime = realtime

    async def chunks(self, window_sec: float, hop_sec: float) -> AsyncGenerator[np.ndarray, None]:
        win = int(self.sample_rate * window_sec)
        hop = int(self.sample_rate * hop_sec)
        for start in range(0, max(1, len(self.signal) - win), hop):
            chunk = self.signal[start : start + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            yield chunk
            if self.realtime:
                await asyncio.sleep(hop_sec)


class LiveMicSource:
    def __init__(self, sample_rate: int = 48000, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._running = True

    def stop(self):
        self._running = False

    async def chunks(self, window_sec: float, hop_sec: float) -> AsyncGenerator[np.ndarray, None]:
        import sounddevice as sd

        win = int(self.sample_rate * window_sec)
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            if status:
                pass
            try:
                loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())
            except Exception:
                pass

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=win,
            device=self.device,
            callback=callback,
        ):
            while self._running:
                chunk = await queue.get()
                yield chunk


class AudioStreamer:
    def __init__(self, source, config: StreamConfig = STREAM_CONFIG):
        self.source = source
        self.config = config
        self.stream_id = str(uuid.uuid4())[:8]
        self.subscribers: list[WebSocket] = []
        self.forensic_subscribers: list[WebSocket] = []
        self._seq = 0
        self._sliding_buffer = collections.deque(maxlen=int(config.sample_rate * 6))
        self.latest_forensic_events: list[DecodedPayloadEvent] = []
        self.is_running = True

    def _to_message(self, chunk: np.ndarray) -> AudioChunkMessage:
        b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode("ascii")
        msg = AudioChunkMessage(
            stream_id=self.stream_id,
            seq=self._seq,
            sample_rate=self.config.sample_rate,
            samples_b64=b64,
            duration_sec=self.config.window_sec,
        )
        self._seq += 1
        return msg

    async def run(self):
        async for chunk in self.source.chunks(self.config.window_sec, self.config.hop_sec):
            if not self.is_running:
                break
            msg = self._to_message(chunk)
            await self._broadcast_audio(msg)

            self._sliding_buffer.extend(chunk)
            if len(self._sliding_buffer) >= int(self.config.sample_rate * 2):
                buf_array = np.array(self._sliding_buffer, dtype=np.float32)
                result = decode_signal_to_text(buf_array, sample_rate=self.config.sample_rate)
                if result.success and result.text:
                    if (
                        not self.latest_forensic_events
                        or self.latest_forensic_events[-1].recovered_text != result.text
                    ):
                        event = DecodedPayloadEvent(
                            event_type="payload_decoded",
                            stream_id=self.stream_id,
                            recovered_text=result.text,
                            confidence=result.confidence,
                            bit_count=result.bit_count,
                            preamble_found=result.preamble_found,
                        )
                        self.latest_forensic_events.append(event)
                        if len(self.latest_forensic_events) > 50:
                            self.latest_forensic_events.pop(0)
                        await self._broadcast_forensic(event)

    async def _broadcast_audio(self, msg: AudioChunkMessage):
        dead = []
        for ws in self.subscribers:
            try:
                await ws.send_json(msg.model_dump())
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.subscribers:
                self.subscribers.remove(ws)

    async def _broadcast_forensic(self, event: DecodedPayloadEvent):
        dead = []
        for ws in self.forensic_subscribers:
            try:
                await ws.send_json(event.model_dump())
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.forensic_subscribers:
                self.forensic_subscribers.remove(ws)

    def stop(self):
        self.is_running = False
        if hasattr(self.source, "stop"):
            self.source.stop()


def decode_chunk_message(msg: AudioChunkMessage) -> np.ndarray:
    raw = base64.b64decode(msg.samples_b64)
    return np.frombuffer(raw, dtype=np.float32)


def list_input_devices() -> dict:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devs = []
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                input_devs.append(
                    {
                        "id": idx,
                        "name": d.get("name"),
                        "hostapi": d.get("hostapi"),
                        "max_input_channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate"),
                    }
                )
        return {"input_devices": input_devs, "default_input": sd.default.device[0]}
    except Exception as e:
        return {"error": str(e), "input_devices": []}
