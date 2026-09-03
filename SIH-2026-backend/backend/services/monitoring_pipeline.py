"""
D1.1 — Live mic stream → DSP (2048 frames) → ML inference → ThreatEvent storage.

Emission rule (documented):
- Accumulate 2048-sample frames from the live stream (48000-sample chunks subdivided).
- DSPPipeline.process() runs per frame; InferenceService.predict() on each frame's 32 features.
- A ThreatEvent is emitted only when ALL hold:
  1. DSP accumulated verdict reports detected=True (to_threat_event, min_chunks >= MIN_DSP_CHUNKS).
  2. Latest ML prediction class is not "benign".
  3. ML risk_level is MEDIUM or HIGH (not LOW).
- Duplicate suppression: skip if same (pattern, carrier_freqs) emitted within EMIT_COOLDOWN_SEC.
- No ThreatEvent is fabricated when thresholds are not met.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from dsp import CHUNK_SIZE, DSPPipeline  # noqa: E402
from ml.phase5.service import InferenceService  # noqa: E402

from backend.models.data_schemas import ThreatEvent
from backend.services.threat_service import add_threat

from backend.services.stream_debug import RateLimitedLogger, chunk_stats

logger = logging.getLogger(__name__)

MIN_DSP_CHUNKS = 47  # ~2.0 s at 2048 samples / 48000 Hz
EMIT_COOLDOWN_SEC = 5.0

ML_CLASS_TO_PATTERN = {
    "benign": "none",
    "fsk": "fsk",
    "ook": "ook",
    "chirp": "chirp",
    "tone": "tone",
}

_inference_service: Optional[InferenceService] = None
_inference_lock = asyncio.Lock()


async def load_inference_service() -> InferenceService:
    """Load frozen Phase 5 InferenceService once (thread-safe)."""
    global _inference_service
    if _inference_service is not None:
        return _inference_service
    async with _inference_lock:
        if _inference_service is None:
            _inference_service = await asyncio.to_thread(InferenceService.from_artifacts)
    return _inference_service


def _process_frames_sync(
    frames: list[np.ndarray],
    dsp: DSPPipeline,
    ml: InferenceService,
) -> tuple[Optional[dict[str, Any]], bool]:
    """Run DSP+ML on a batch of 2048-sample frames (worker thread)."""
    last_ml: Optional[dict[str, Any]] = None
    event_closed = False
    for frame in frames:
        dsp_result = dsp.process(frame)
        feature_values = dsp_result["features"]["feature_values"]
        last_ml = ml.predict(feature_values)
        if dsp_result.get("event_closed"):
            event_closed = True
    return last_ml, event_closed


class MonitoringPipeline:
    """Buffers live audio, subdivides to DSP frames, runs ML, emits ThreatEvents."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate
        self._dsp = DSPPipeline(sample_rate=sample_rate)
        self._ml: Optional[InferenceService] = None
        self._remainder = np.array([], dtype=np.float32)
        self._active = False
        self._last_ml: Optional[dict[str, Any]] = None
        self._last_emit_time = 0.0
        self._last_emit_signature: Optional[tuple[str, tuple[float, ...]]] = None
        self._pipeline_log = RateLimitedLogger("PIPELINE")
        self._chunks_received = 0
        self._chunk_lock = asyncio.Lock()
        self.last_dsp_snapshot: dict[str, Any] = {}
        self.last_chunk_ts = 0.0

    async def start(self) -> None:
        self._ml = await load_inference_service()
        self._dsp.reset()
        self._remainder = np.array([], dtype=np.float32)
        self._last_ml = None
        self._last_emit_time = 0.0
        self._last_emit_signature = None
        self._active = True
        logger.info("[D1.1] AI pipeline started (sample_rate=%d, frame_size=%d)", self.sample_rate, CHUNK_SIZE)

    async def stop(self) -> None:
        self._active = False
        self._remainder = np.array([], dtype=np.float32)
        self._dsp.reset()
        self._last_ml = None
        logger.info("[D1.1] AI pipeline stopped")

    def reset_accumulator(self) -> None:
        """Clear DSP frame accumulation for a clean simulator injection window."""
        self._dsp.reset()
        self._remainder = np.array([], dtype=np.float32)
        logger.info(
            "[SIM] RESET pipeline_id=%s dsp_id=%s DSP_RESET=OK",
            id(self),
            id(self._dsp),
        )

    @property
    def dsp_object_id(self) -> int:
        return id(self._dsp)

    async def process_chunk(self, chunk: np.ndarray) -> None:
        if not self._active or self._ml is None:
            return

        async with self._chunk_lock:
            samples = np.asarray(chunk, dtype=np.float32).ravel()
            if samples.size == 0:
                return

            self._chunks_received += 1
            self.last_chunk_ts = time.time()
            stats = chunk_stats(samples)
            self._pipeline_log.log(
                "PIPELINE CHUNK RECEIVED #%d samples=%d rms=%.6f peak=%.6f frames_pending=%d",
                self._chunks_received,
                int(stats["count"]),
                stats["rms"],
                stats["peak"],
                len(self._remainder),
            )

            combined = np.concatenate([self._remainder, samples])
            n_frames = len(combined) // CHUNK_SIZE
            if n_frames == 0:
                self._remainder = combined
                return

            frames = [
                combined[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE].astype(np.float64, copy=False)
                for i in range(n_frames)
            ]
            self._remainder = combined[n_frames * CHUNK_SIZE :]

            last_ml, event_closed = await asyncio.to_thread(
                _process_frames_sync,
                frames,
                self._dsp,
                self._ml,
            )

            for _ in frames:
                logger.debug("[D1.1] DSP frame processed")

            if last_ml is not None:
                self._last_ml = last_ml
                logger.info(
                    "[ML RESULT] prediction=%s risk=%s confidence=%.3f",
                    last_ml.get("predicted_class"),
                    last_ml.get("risk_level"),
                    float(last_ml.get("confidence", 0.0)),
                )

            dsp_snapshot = self._dsp.to_threat_event(min_chunks=MIN_DSP_CHUNKS)
            self.last_dsp_snapshot = dsp_snapshot
            self._pipeline_log.log(
                "DSP RESULT detected=%s pattern=%s freq=%s-%s snr=%.2f chunks=%s ml=%s risk=%s",
                dsp_snapshot.get("detected"),
                dsp_snapshot.get("pattern"),
                dsp_snapshot.get("frequency_start"),
                dsp_snapshot.get("frequency_end"),
                float(dsp_snapshot.get("snr") or 0.0),
                dsp_snapshot.get("chunks_analyzed"),
                (self._last_ml or {}).get("predicted_class"),
                (self._last_ml or {}).get("risk_level"),
            )

            await self._maybe_emit_threat_event(force_segment_close=event_closed)

    async def _maybe_emit_threat_event(self, force_segment_close: bool = False) -> None:
        if self._last_ml is None:
            return

        dsp_event = self._dsp.to_threat_event(min_chunks=MIN_DSP_CHUNKS)
        if not dsp_event.get("detected"):
            return

        ml = self._last_ml
        predicted_class = ml.get("predicted_class", "benign")
        risk_level = ml.get("risk_level", "LOW")

        if predicted_class == "benign" or risk_level == "LOW":
            return

        if not force_segment_close and dsp_event.get("chunks_analyzed", 0) < MIN_DSP_CHUNKS:
            return

        pattern = ML_CLASS_TO_PATTERN.get(predicted_class, dsp_event.get("pattern", "none"))
        carriers = tuple(float(c) for c in dsp_event.get("carrier_freqs", []))
        signature = (pattern, carriers)
        now = time.time()
        if (
            self._last_emit_signature == signature
            and (now - self._last_emit_time) < EMIT_COOLDOWN_SEC
        ):
            logger.debug("[D1.1] ThreatEvent suppressed (duplicate within cooldown)")
            return

        threat = ThreatEvent(
            schema_version=dsp_event.get("schema_version", "1.0.0-dsp"),
            detected=True,
            confidence=round(float(ml["confidence"]), 4),
            risk=risk_level,
            suspicion_score=round(float(ml["calibrated_risk_score"]), 4),
            frequency_start=dsp_event.get("frequency_start"),
            frequency_end=dsp_event.get("frequency_end"),
            carrier_freqs=[float(c) for c in dsp_event.get("carrier_freqs", [])],
            duration=float(dsp_event.get("duration", 0.0)),
            pattern=pattern if pattern in ("none", "tone", "fsk", "ook", "chirp") else "fsk",
            snr=float(dsp_event.get("snr", 0.0)),
            chunks_analyzed=int(dsp_event.get("chunks_analyzed", 0)),
            timestamp=now,
        )
        add_threat(threat)
        self._last_emit_time = now
        self._last_emit_signature = signature
        logger.info(
            "[THREAT GENERATED] threat_id=TE-%d pattern=%s risk=%s confidence=%.4f carriers=%s",
            int(threat.timestamp * 1000),
            threat.pattern,
            threat.risk,
            threat.confidence,
            threat.carrier_freqs,
        )
