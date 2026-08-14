import time
from typing import Optional

from pydantic import BaseModel, Field


class AudioChunkMessage(BaseModel):
    stream_id: str
    seq: int
    sample_rate: int = 48000
    samples_b64: str
    timestamp: float = Field(default_factory=time.time)
    duration_sec: float = 1.0


class StreamConfig(BaseModel):
    sample_rate: int = 48000
    window_sec: float = 1.0
    hop_sec: float = 0.5
    band_low_hz: int = 17000
    band_high_hz: int = 24000


class GeneratePayloadRequest(BaseModel):
    text: str = Field(..., max_length=256, description="Payload text to encode")
    freq_0_hz: int = 18500
    freq_1_hz: int = 21000
    bit_duration_ms: float = 50.0


class GeneratePayloadResponse(BaseModel):
    payload_id: str
    duration_sec: float
    bit_count: int
    wav_url: str


class TransmitRequest(BaseModel):
    payload_id: str
    mode: str = Field(
        "virtual",
        description="'virtual' = inject into pipeline; 'audio' = play through speaker",
    )


class TransmitStatus(BaseModel):
    payload_id: str
    state: str
    progress_pct: float = 0.0


class DecodedPayloadEvent(BaseModel):
    event_type: str = "payload_decoded"
    stream_id: str
    recovered_text: Optional[str]
    confidence: float
    bit_count: int
    preamble_found: bool
    timestamp: float = Field(default_factory=time.time)


class FeatureVectorMessage(BaseModel):
    stream_id: str
    seq: int
    features_b64: str
    feature_shape: list[int]
    timestamp: float = Field(default_factory=time.time)


class AcousticExfiltrateRequest(BaseModel):
    source_type: str = Field("custom", description="'custom' | 'keylog' | 'sysinfo'")
    custom_text: Optional[str] = Field(None, description="Text to exfiltrate if source_type is 'custom'")
    freq_0_hz: int = Field(18500, description="Frequency representing bit 0 (Hz)")
    freq_1_hz: int = Field(21000, description="Frequency representing bit 1 (Hz)")
    bit_duration_ms: float = Field(50.0, description="Duration per bit in milliseconds")
    emit_audio: bool = Field(True, description="Whether to play tone through PC speaker")


class AcousticExfiltrateResponse(BaseModel):
    status: str
    text_length: int
    duration_sec: float
    bit_count: int
    payload_id: str
    wav_url: Optional[str] = None
    preview_text: str


class ReverseShellCommandRequest(BaseModel):
    target_id: Optional[str] = None
    command: str = Field(..., description="Reverse shell command or acoustic trigger")


class ReverseShellStatus(BaseModel):
    connected: bool
    target_ip: Optional[str] = None
    last_seen: Optional[float] = None
    admin_privileges: Optional[bool] = None
    active_keylogger: bool = False


# ---------------------------------------------------------------------------
# Backend 1 — Threat Management Models (TEMPORARY contract, see integration-contract.md)
# ---------------------------------------------------------------------------


class FrequencyRange(BaseModel):
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)


class ThreatEvent(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk: str
    frequency: FrequencyRange
    duration: float = Field(..., ge=0)
    pattern: str
