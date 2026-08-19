import os
import socket
import uuid

from fastapi import APIRouter, HTTPException

from backend.api.system_status import get_payloads
from backend.core.config import SAMPLE_RATE, STORAGE_DIR
from backend.models.data_schemas import (
    AcousticExfiltrateRequest,
    AcousticExfiltrateResponse,
    GeneratePayloadRequest,
    GeneratePayloadResponse,
    TransmitRequest,
    TransmitStatus,
)
from backend.services.payload_service import encode_text_to_signal, play_ultrasonic_signal, save_signal_to_wav

router = APIRouter(prefix="/simulate", tags=["simulator"])


@router.post("/generate", response_model=GeneratePayloadResponse)
def generate_payload(req: GeneratePayloadRequest):
    payloads = get_payloads()
    payload_id = str(uuid.uuid4())[:8]
    signal = encode_text_to_signal(
        req.text,
        freq_0=req.freq_0_hz,
        freq_1=req.freq_1_hz,
        bit_duration=req.bit_duration_ms / 1000.0,
        sample_rate=SAMPLE_RATE,
    )
    path = os.path.join(STORAGE_DIR, f"{payload_id}.wav")
    save_signal_to_wav(path, signal, SAMPLE_RATE)

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
def transmit_payload(req: TransmitRequest):
    payloads = get_payloads()
    if req.payload_id not in payloads:
        raise HTTPException(404, "Unknown payload_id - call /simulate/generate first")

    entry = payloads[req.payload_id]

    if req.mode == "audio":
        success = play_ultrasonic_signal(entry["signal"], sample_rate=SAMPLE_RATE, blocking=False)
        if not success:
            raise HTTPException(
                500,
                "Speaker playback failed. Check audio output device or fall back to mode='virtual'.",
            )
        entry["state"] = "complete"
    else:
        entry["state"] = "complete"

    return TransmitStatus(payload_id=req.payload_id, state=entry["state"], progress_pct=100.0)


@router.post("/acoustic-exfiltrate", response_model=AcousticExfiltrateResponse)
def acoustic_exfiltrate(req: AcousticExfiltrateRequest):
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
        play_ultrasonic_signal(signal, sample_rate=SAMPLE_RATE, blocking=False)
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
    return TransmitStatus(payload_id=payload_id, state=entry["state"], progress_pct=100.0)


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
