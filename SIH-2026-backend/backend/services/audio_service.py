import asyncio
import base64
import collections
import logging
import queue as thread_queue
import time
import uuid
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import WebSocket

from backend.core.config import STREAM_CONFIG
from backend.models.data_schemas import AudioChunkMessage, DecodedPayloadEvent, StreamConfig
from backend.services.monitoring_pipeline import MonitoringPipeline
from backend.services.payload_service import decode_signal_to_text
from backend.services.stream_debug import RateLimitedLogger, chunk_stats

logger = logging.getLogger(__name__)


async def feed_signal_to_active_monitoring(
    signal: np.ndarray,
    monitoring: MonitoringPipeline,
    *,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    realtime: bool = False,
    streamer: Optional["AudioStreamer"] = None,
) -> dict:
    """Replay a generated payload through an already-running monitoring pipeline."""
    win = int(sample_rate * window_sec)
    hop = max(1, int(sample_rate * hop_sec))
    chunks_sent = 0
    logger.info(
        "[SIM] VIRTUAL FEED START samples=%d sr=%d window=%.2fs hop=%.2fs pipeline_id=%s dsp_id=%s",
        len(signal),
        sample_rate,
        window_sec,
        hop_sec,
        id(monitoring),
        monitoring.dsp_object_id,
    )
    if streamer is not None:
        streamer.simulator_feed_active = True
    monitoring.reset_accumulator()
    try:
        total_chunks = max(1, (max(1, len(signal) - win) + hop - 1) // hop)
        for start in range(0, max(1, len(signal) - win), hop):
            chunk = signal[start : start + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            await monitoring.process_chunk(chunk.astype(np.float32, copy=False))
            chunks_sent += 1
            stats = chunk_stats(chunk)
            logger.info(
                "[SIM] VIRTUAL FEED CHUNK %d/%d samples=%d rms=%.6f peak=%.6f",
                chunks_sent,
                total_chunks,
                int(stats["count"]),
                stats["rms"],
                stats["peak"],
            )
            if realtime:
                await asyncio.sleep(hop_sec)
    finally:
        if streamer is not None:
            streamer.simulator_feed_active = False
    logger.info("[SIM] VIRTUAL FEED COMPLETE chunks=%d", chunks_sent)
    return {
        "chunks_processed": chunks_sent,
        "dsp_detected": bool(monitoring.last_dsp_snapshot.get("detected")),
        "dsp_pattern": monitoring.last_dsp_snapshot.get("pattern"),
        "ml_class": (monitoring._last_ml or {}).get("predicted_class"),  # noqa: SLF001
        "ml_risk": (monitoring._last_ml or {}).get("risk_level"),  # noqa: SLF001
    }


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
    _STOP_SENTINEL = object()

    def __init__(self, sample_rate: int = 48000, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._running = True
        self._block_queue: Optional[thread_queue.Queue] = None

    def stop(self):
        self._running = False
        if self._block_queue is not None:
            try:
                self._block_queue.put_nowait(self._STOP_SENTINEL)
            except thread_queue.Full:
                try:
                    self._block_queue.get_nowait()
                except thread_queue.Empty:
                    pass
                try:
                    self._block_queue.put_nowait(self._STOP_SENTINEL)
                except Exception:
                    pass
            except Exception:
                pass

    async def chunks(self, window_sec: float, hop_sec: float) -> AsyncGenerator[np.ndarray, None]:
        if not self._running:
            return

        import sounddevice as sd

        win = int(self.sample_rate * window_sec)
        hop = max(1, int(self.sample_rate * hop_sec))
        blocksize = min(hop, 4096)
        block_queue: thread_queue.Queue = thread_queue.Queue(maxsize=200)
        self._block_queue = block_queue
        pcm_buffer = np.array([], dtype=np.float32)
        mic_log = RateLimitedLogger("MIC")
        chunks_yielded = 0
        status_warnings = 0
        dropped_blocks = 0

        logger.info(
            "[MIC START] device_id=%s sample_rate=%s channels=1 window=%s hop=%s blocksize=%s",
            self.device,
            self.sample_rate,
            win,
            hop,
            blocksize,
        )

        def enqueue_chunk(block):
            nonlocal dropped_blocks
            try:
                block_queue.put_nowait(block)
            except thread_queue.Full:
                dropped_blocks += 1
                try:
                    block_queue.get_nowait()
                    block_queue.put_nowait(block)
                except thread_queue.Empty:
                    pass
                if dropped_blocks <= 3 or dropped_blocks % 20 == 0:
                    mic_log.log(
                        "MIC BLOCK DROPPED queue full (consumer lag) dropped=%d",
                        dropped_blocks,
                    )

        def callback(indata, frames, time_info, status):
            nonlocal status_warnings
            if not self._running:
                return
            if status:
                status_warnings += 1
                if status_warnings <= 3 or status_warnings % 20 == 0:
                    logger.warning("[MIC] InputStream status: %s", status)
            try:
                enqueue_chunk(indata[:, 0].copy())
            except Exception:
                logger.exception("[MIC] callback failed")

        def open_and_start():
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=blocksize,
                device=self.device,
                callback=callback,
            )
            stream.start()
            return stream

        def stop_and_close(stream):
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        input_stream = None
        try:
            input_stream = await asyncio.to_thread(open_and_start)
            while self._running:
                block = await asyncio.to_thread(block_queue.get)
                if block is self._STOP_SENTINEL:
                    break
                stats = chunk_stats(block)
                mic_log.log(
                    "MIC BLOCK RECEIVED samples=%d rms=%.6f peak=%.6f min=%.6f max=%.6f status_warns=%d",
                    int(stats["count"]),
                    stats["rms"],
                    stats["peak"],
                    stats["min"],
                    stats["max"],
                    status_warnings,
                )
                pcm_buffer = np.concatenate([pcm_buffer, block.astype(np.float32, copy=False)])
                while len(pcm_buffer) >= win:
                    chunk = pcm_buffer[:win].copy()
                    chunks_yielded += 1
                    stats = chunk_stats(chunk)
                    mic_log.log(
                        "MIC CHUNK YIELDED #%d samples=%d rms=%.6f peak=%.6f min=%.6f max=%.6f",
                        chunks_yielded,
                        int(stats["count"]),
                        stats["rms"],
                        stats["peak"],
                        stats["min"],
                        stats["max"],
                    )
                    yield chunk
                    pcm_buffer = pcm_buffer[hop:]
        finally:
            self._running = False
            if input_stream is not None:
                await asyncio.to_thread(stop_and_close, input_stream)
            self._block_queue = None
            logger.info(
                "[MIC STOP] device_id=%s chunks_yielded=%d dropped_blocks=%d status_warns=%d",
                self.device,
                chunks_yielded,
                dropped_blocks,
                status_warnings,
            )


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
        self.monitoring: MonitoringPipeline | None = None
        self._last_forensic_decode_ts = 0.0
        self.simulator_feed_active = False

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
        monitoring = MonitoringPipeline(sample_rate=self.config.sample_rate)
        self.monitoring = monitoring
        logger.info(
            "[STREAM RUN] streamer_id=%s monitoring_id=%s stream_id=%s",
            id(self),
            id(monitoring),
            self.stream_id,
        )
        try:
            await monitoring.start()
            async for chunk in self.source.chunks(self.config.window_sec, self.config.hop_sec):
                if not self.is_running:
                    break
                msg = self._to_message(chunk)
                await self._broadcast_audio(msg)

                if not self.simulator_feed_active:
                    await monitoring.process_chunk(chunk)

                self._sliding_buffer.extend(chunk)
                now = time.monotonic()
                if (
                    not self.simulator_feed_active
                    and len(self._sliding_buffer) >= int(self.config.sample_rate * 2)
                    and now - self._last_forensic_decode_ts >= 2.0
                ):
                    self._last_forensic_decode_ts = now
                    buf_array = np.array(self._sliding_buffer, dtype=np.float32)
                    result = await asyncio.to_thread(decode_signal_to_text, buf_array)
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
        finally:
            while self.simulator_feed_active:
                await asyncio.sleep(0.05)
            await monitoring.stop()
            self.monitoring = None

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
