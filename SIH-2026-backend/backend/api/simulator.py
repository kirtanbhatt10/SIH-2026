import asyncio
import logging
import os
import socket
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api.system_status import (
    get_last_simulator_diag,
    get_payloads,
    set_last_simulator_diag,
)
from backend.core.config import (
    APP_VERSION,
    MIC_DEVICE_ID,
    SAMPLE_RATE,
    SPEAKER_DEVICE_ID,
    STORAGE_DIR,
    STREAM_CONFIG,
)
from backend.services.audio_service import feed_signal_to_active_monitoring
from backend.services.payload_service import (
    encode_text_to_signal,
    get_audio_device_info,
    play_ultrasonic_signal,
    save_signal_to_wav,
)
from backend.services.stream_manager import get_active_monitoring, get_active_streamer
from backend.services.threat_service import get_threats
from backend.models.data_schemas import (
    AcousticExfiltrateRequest,
    AcousticExfiltrateResponse,
    GeneratePayloadRequest,
    GeneratePayloadResponse,
    TransmitRequest,
    TransmitStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulate", tags=["simulator"])


def _build_transmit_status(
    payload_id: str,
    state: str,
    progress_pct: float,
    *,
    monitoring=None,
    feed_result: dict | None = None,
    threats_before: int = 0,
    threats_after: int = 0,
    reason: str | None = None,
) -> TransmitStatus:
    ml_class = None
    ml_risk = None
    dsp_detected = None
    chunks_processed = None
    if feed_result:
        ml_class = feed_result.get("ml_class")
        ml_risk = feed_result.get("ml_risk")
        dsp_detected = feed_result.get("dsp_detected")
        chunks_processed = feed_result.get("chunks_processed")
    elif monitoring is not None:
        dsp_detected = bool(monitoring.last_dsp_snapshot.get("detected"))
        ml_class = (monitoring._last_ml or {}).get("predicted_class")  # noqa: SLF001
        ml_risk = (monitoring._last_ml or {}).get("risk_level")  # noqa: SLF001

    return TransmitStatus(
        payload_id=payload_id,
        state=state,
        progress_pct=progress_pct,
        chunks_processed=chunks_processed,
        dsp_detected=dsp_detected,
        ml_class=ml_class,
        ml_risk=ml_risk,
        threats_before=threats_before,
        threats_after=threats_after,
        pipeline_object_id=id(monitoring) if monitoring else None,
        dsp_object_id=monitoring.dsp_object_id if monitoring else None,
        reason=reason,
    )


def _log_generated_signal(
    *,
    payload_id: str,
    path: str,
    signal: np.ndarray,
    freq_0: int,
    freq_1: int,
) -> None:
    duration_sec = len(signal) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(signal * signal))) if len(signal) else 0.0
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    logger.info(
        "[SIM] GENERATE COMPLETE payload_id=%s sr=%d duration=%.2fs samples=%d rms=%.6f peak=%.6f f0=%d f1=%d path=%s",
        payload_id,
        SAMPLE_RATE,
        duration_sec,
        len(signal),
        rms,
        peak,
        freq_0,
        freq_1,
        path,
    )
    set_last_simulator_diag(
        last_generate_payload_id=payload_id,
        last_generate_path=path,
        last_generate_duration_sec=duration_sec,
        last_generate_samples=len(signal),
        last_generate_rms=rms,
        last_generate_peak=peak,
        last_generate_freq_0=freq_0,
        last_generate_freq_1=freq_1,
    )


@router.post("/generate", response_model=GeneratePayloadResponse)
def generate_payload(req: GeneratePayloadRequest):
    logger.info(
        "[SIM] GENERATE START text_len=%d freq_0=%d freq_1=%d bit_ms=%s",
        len(req.text),
        req.freq_0_hz,
        req.freq_1_hz,
        req.bit_duration_ms,
    )
    if not req.text.strip():
        raise HTTPException(400, "Payload text must not be empty")
    payloads = get_payloads()
    payload_id = str(uuid.uuid4())[:8]
    signal = encode_text_to_signal(
        req.text,
        freq_0=req.freq_0_hz,
        freq_1=req.freq_1_hz,
        bit_duration=req.bit_duration_ms / 1000.0,
        sample_rate=SAMPLE_RATE,
    )
    if signal.size == 0:
        raise HTTPException(500, "Signal generation produced empty array")
    path = os.path.abspath(os.path.join(STORAGE_DIR, f"{payload_id}.wav"))
    save_signal_to_wav(path, signal, SAMPLE_RATE)
    _log_generated_signal(
        payload_id=payload_id,
        path=path,
        signal=signal,
        freq_0=req.freq_0_hz,
        freq_1=req.freq_1_hz,
    )

    payloads[payload_id] = {
        "signal": signal,
        "path": path,
        "state": "ready",
        "text": req.text,
        "freq_0": req.freq_0_hz,
        "freq_1": req.freq_1_hz,
    }

    return GeneratePayloadResponse(
        payload_id=payload_id,
        duration_sec=len(signal) / SAMPLE_RATE,
        bit_count=len(req.text) * 8 + 8,
        wav_url=f"/files/{payload_id}.wav",
    )


@router.post("/transmit", response_model=TransmitStatus)
async def transmit_payload(req: TransmitRequest):
    payloads = get_payloads()
    if req.payload_id not in payloads:
        raise HTTPException(404, "Unknown payload_id - call /simulate/generate first")

    entry = payloads[req.payload_id]
    signal = entry["signal"]
    if signal is None or len(signal) == 0:
        raise HTTPException(400, "Payload signal is empty — regenerate payload")

    streamer = get_active_streamer()
    monitoring = get_active_monitoring()
    threats_before = len(get_threats())

    logger.info(
        "[SIM] TRANSMIT START payload_id=%s mode=%s streamer_id=%s monitoring_id=%s dsp_id=%s threats_before=%d",
        req.payload_id,
        req.mode,
        id(streamer) if streamer else None,
        id(monitoring) if monitoring else None,
        monitoring.dsp_object_id if monitoring else None,
        threats_before,
    )

    if streamer and monitoring and streamer.monitoring is not monitoring:
        logger.error(
            "[SIM] INSTANCE MISMATCH streamer.monitoring_id=%s lookup_monitoring_id=%s",
            id(streamer.monitoring),
            id(monitoring),
        )

    set_last_simulator_diag(
        last_transmit_payload_id=req.payload_id,
        last_transmit_mode=req.mode,
        stream_active=bool(streamer and streamer.is_running),
        monitoring_attached=monitoring is not None,
        threats_before=threats_before,
    )

    feed_result: dict | None = None

    if req.mode == "virtual":
        if monitoring is None:
            entry["state"] = "failed"
            raise HTTPException(
                409,
                "No active monitoring pipeline. Start /monitor (POST /stream/start?source=mic&device_id=1) before Acoustic Simulation.",
            )
        entry["state"] = "processing"
        feed_result = await feed_signal_to_active_monitoring(
            signal,
            monitoring,
            sample_rate=SAMPLE_RATE,
            window_sec=STREAM_CONFIG.window_sec,
            hop_sec=STREAM_CONFIG.hop_sec,
            realtime=False,
            streamer=streamer,
        )
        threats_after = len(get_threats())
        detected = threats_after > threats_before
        if detected:
            entry["state"] = "complete"
            reason = None
            logger.info(
                "[SIM] TRANSMIT COMPLETE detection=YES threats_before=%d threats_after=%d",
                threats_before,
                threats_after,
            )
        else:
            entry["state"] = "completed_no_detection"
            reason = (
                f"DSP detected={feed_result.get('dsp_detected')}, "
                f"ML={feed_result.get('ml_class')}/{feed_result.get('ml_risk')}, "
                f"chunks={feed_result.get('chunks_processed')}"
            )
            logger.warning(
                "[SIM] TRANSMIT COMPLETE detection=NO threats_before=%d threats_after=%d %s",
                threats_before,
                threats_after,
                reason,
            )
        set_last_simulator_diag(
            last_virtual_feed="complete",
            last_virtual_chunks=feed_result.get("chunks_processed"),
            threats_after=threats_after,
            detection=detected,
        )
        return _build_transmit_status(
            req.payload_id,
            entry["state"],
            100.0,
            monitoring=monitoring,
            feed_result=feed_result,
            threats_before=threats_before,
            threats_after=threats_after,
            reason=reason,
        )

    if req.mode == "audio":
        if streamer is None or not streamer.is_running:
            logger.warning("[SIM] transmit mode=audio without active mic stream")
        if monitoring is not None:
            monitoring.reset_accumulator()
        entry["state"] = "playing"
        logger.info("[SIM] SPEAKER DEVICE=%s PLAYBACK START", SPEAKER_DEVICE_ID)
        success = await asyncio.to_thread(
            play_ultrasonic_signal,
            signal,
            SAMPLE_RATE,
            True,
            None,
        )
        if not success:
            entry["state"] = "failed"
            raise HTTPException(500, "Speaker playback failed on device %s" % SPEAKER_DEVICE_ID)
        logger.info("[SIM] SPEAKER DEVICE=%s PLAYBACK FINISHED", SPEAKER_DEVICE_ID)
        if monitoring is not None and streamer is not None:
            entry["state"] = "processing"
            feed_result = await feed_signal_to_active_monitoring(
                signal,
                monitoring,
                sample_rate=SAMPLE_RATE,
                window_sec=STREAM_CONFIG.window_sec,
                hop_sec=STREAM_CONFIG.hop_sec,
                realtime=False,
                streamer=streamer,
            )
        threats_after = len(get_threats())
        detected = threats_after > threats_before
        if detected:
            entry["state"] = "complete"
            reason = None
        else:
            entry["state"] = "completed_no_detection"
            reason = (
                "Speaker playback succeeded but no ThreatEvent was created. "
                f"DSP={feed_result.get('dsp_detected') if feed_result else 'n/a'}, "
                f"ML={feed_result.get('ml_class') if feed_result else 'n/a'}"
            )
        set_last_simulator_diag(last_audio_playback="finished", threats_after=threats_after, detection=detected)
        return _build_transmit_status(
            req.payload_id,
            entry["state"],
            100.0,
            monitoring=monitoring,
            feed_result=feed_result,
            threats_before=threats_before,
            threats_after=threats_after,
            reason=reason,
        )

    raise HTTPException(422, f"Unsupported transmit mode: {req.mode}")


@router.post("/acoustic-exfiltrate", response_model=AcousticExfiltrateResponse)
async def acoustic_exfiltrate(req: AcousticExfiltrateRequest):
    payloads = get_payloads()

    if req.source_type == "sysinfo":
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "127.0.0.1"
        data_text = f"SYSINFO:{hostname}|IP:{ip}"
    elif req.source_type == "keylog":
        data_text = "KEYLOG:admin_pass_2026!#sudo"
    else:
        data_text = req.custom_text or "COVERT_ACOUSTIC_SIGNAL_SIH2026"

    payload_id = str(uuid.uuid4())[:8]
    signal = encode_text_to_signal(
        data_text,
        freq_0=req.freq_0_hz,
        freq_1=req.freq_1_hz,
        bit_duration=req.bit_duration_ms / 1000.0,
        sample_rate=SAMPLE_RATE,
    )

    path = os.path.join(STORAGE_DIR, f"{payload_id}.wav")
    save_signal_to_wav(path, signal, SAMPLE_RATE)
    payloads[payload_id] = {"signal": signal, "path": path, "state": "ready", "text": data_text}

    if req.emit_audio:
        await asyncio.to_thread(play_ultrasonic_signal, signal, SAMPLE_RATE, True, None)
        status_msg = "Ultrasonic tone emitted through PC speaker"
    else:
        status_msg = "WAV generated without audio emission"

    return AcousticExfiltrateResponse(
        status=status_msg,
        text_length=len(data_text),
        duration_sec=len(signal) / SAMPLE_RATE,
        bit_count=len(data_text) * 8 + 8,
        payload_id=payload_id,
        wav_url=f"/files/{payload_id}.wav",
        preview_text=data_text[:50] + ("..." if len(data_text) > 50 else ""),
    )


@router.get("/status/{payload_id}", response_model=TransmitStatus)
def get_status(payload_id: str):
    payloads = get_payloads()
    if payload_id not in payloads:
        raise HTTPException(404, "Unknown payload_id")
    entry = payloads[payload_id]
    state = entry["state"]
    if state in ("complete", "failed", "completed_no_detection"):
        progress = 100.0
    elif state == "processing":
        progress = 75.0
    elif state == "playing":
        progress = 50.0
    elif state == "ready":
        progress = 0.0
    else:
        progress = 50.0
    monitoring = get_active_monitoring()
    return _build_transmit_status(
        payload_id,
        state,
        progress,
        monitoring=monitoring,
        threats_before=entry.get("threats_before"),
        threats_after=len(get_threats()),
    )


@router.get("/diagnostics")
def simulator_diagnostics():
    streamer = get_active_streamer()
    monitoring = get_active_monitoring()
    mic_info = get_audio_device_info(MIC_DEVICE_ID, "input")
    spk_info = get_audio_device_info(SPEAKER_DEVICE_ID, "output")
    diag = get_last_simulator_diag()
    threats = get_threats()
    last_threat = threats[-1] if threats else None
    last_dsp = monitoring.last_dsp_snapshot if monitoring else {}
    return {
        "backend_version": APP_VERSION,
        "backend_online": True,
        "stream_active": bool(streamer and streamer.is_running),
        "stream_id": streamer.stream_id if streamer else None,
        "stream_source": type(streamer.source).__name__ if streamer else None,
        "monitoring_active": bool(monitoring and monitoring._active),  # noqa: SLF001
        "monitoring_attached": monitoring is not None,
        "pipeline_object_id": id(monitoring) if monitoring else None,
        "dsp_object_id": monitoring.dsp_object_id if monitoring else None,
        "streamer_object_id": id(streamer) if streamer else None,
        "streamer_monitoring_object_id": id(streamer.monitoring) if streamer and streamer.monitoring else None,
        "simulator_feed_active": bool(streamer and streamer.simulator_feed_active),
        "chunks_analyzed": last_dsp.get("chunks_analyzed"),
        "threats_count": len(threats),
        "last_detection": last_threat.model_dump() if last_threat else None,
        "last_simulation": diag,
        "mic_device": mic_info,
        "speaker_device": spk_info,
        "last_dsp": last_dsp,
        "last_ml": monitoring._last_ml if monitoring else None,  # noqa: SLF001
    }


@router.get("/list")
def list_payloads():
    payloads = get_payloads()
    return [
        {
            "payload_id": pid,
            "text": v["text"],
            "state": v["state"],
            "wav_url": f"/files/{pid}.wav",
        }
        for pid, v in payloads.items()
    ]
