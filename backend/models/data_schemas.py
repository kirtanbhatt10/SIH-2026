import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    freq_1_hz: int = 20500
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
    freq_1_hz: int = Field(20500, description="Frequency representing bit 1 (Hz)")
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
    """Legacy stub — not used by Backend 2 simulator API. See isolated/README.md."""

    target_id: Optional[str] = None
    command: str = Field(..., description="Reverse shell command or acoustic trigger")


class ReverseShellStatus(BaseModel):
    """Legacy stub — not used by Backend 2 simulator API. See isolated/README.md."""
    connected: bool
    target_ip: Optional[str] = None
    last_seen: Optional[float] = None
    admin_privileges: Optional[bool] = None
    active_keylogger: bool = False


# ---------------------------------------------------------------------------
# Backend 1 — Threat Management Models (AI contract schema 1.0.0-dsp)
# ---------------------------------------------------------------------------

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
PatternType = Literal["none", "tone", "fsk", "ook", "chirp"]


class ThreatEvent(BaseModel):
    """AI → Backend detection event. Backend validates and stores; no inference."""

    schema_version: str = Field(..., description='Expected current value: "1.0.0-dsp"')
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Modulation-classifier certainty")
    risk: RiskLevel
    suspicion_score: float = Field(..., ge=0.0, le=1.0, description="Threat-likeness heuristic")
    frequency_start: Optional[float] = Field(None, ge=0, description="Hz; null when no carrier band")
    frequency_end: Optional[float] = Field(None, ge=0, description="Hz; null when no carrier band")
    carrier_freqs: list[float] = Field(
        default_factory=list,
        description="Discrete detected carriers (preserves FSK structure)",
    )
    duration: float = Field(..., ge=0, description="Event-level duration in seconds")
    pattern: PatternType
    snr: float = Field(..., description="Peak SNR in dB")
    chunks_analyzed: int = Field(..., ge=0)
    timestamp: float = Field(..., description="Unix epoch seconds")

    @field_validator("carrier_freqs")
    @classmethod
    def carrier_freqs_non_negative(cls, values: list[float]) -> list[float]:
        for freq in values:
            if freq < 0:
                raise ValueError("carrier_freqs values must be >= 0")
        return values
