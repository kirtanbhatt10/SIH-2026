import logging
import os
import wave
from dataclasses import dataclass

import numpy as np

from backend.core.config import (
    BIT_DURATION,
    FREQ_0,
    FREQ_1,
    PREAMBLE,
    SAMPLE_RATE,
    SPEAKER_DEVICE_ID,
)

logger = logging.getLogger(__name__)


def get_audio_device_info(device_id: int, kind: str = "output") -> dict:
    """Return PortAudio device metadata for diagnostics."""
    try:
        import sounddevice as sd

        dev = sd.query_devices(device_id)
        return {
            "index": device_id,
            "name": dev.get("name"),
            "max_input_channels": dev.get("max_input_channels"),
            "max_output_channels": dev.get("max_output_channels"),
            "default_samplerate": dev.get("default_samplerate"),
            "kind": kind,
            "available": True,
        }
    except Exception as exc:
        return {"index": device_id, "kind": kind, "available": False, "error": str(exc)}


# Reject preamble bit-pattern matches whose tone energy is negligible vs the
# file's strongest tone window (~1% of peak). PC2 forensic noise false-positives
# sit at ~0.03–0.14% of peak; clean TX preambles are near 100% of peak.
PREAMBLE_MIN_PEAK_FRACTION = 0.01


@dataclass
class DecodeResult:
    success: bool
    text: str | None
    bits: str
    confidence: float
    preamble_found: bool
    bit_count: int


def encode_text_to_signal(
    text: str,
    pad_silence_sec: float = 0.5,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    bit_duration: float = BIT_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    bits = PREAMBLE + _text_to_bits(text)
    tone = _bits_to_tone(bits, freq_0=freq_0, freq_1=freq_1, bit_duration=bit_duration, sample_rate=sample_rate)
    pad = np.zeros(int(sample_rate * pad_silence_sec), dtype=np.float32)
    return np.concatenate([pad, tone, pad]).astype(np.float32)


def decode_signal_to_text(
    signal: np.ndarray,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    bit_duration: float = BIT_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> DecodeResult:
    samples_per_bit = int(sample_rate * bit_duration)
    if len(signal) < samples_per_bit * len(PREAMBLE):
        return DecodeResult(False, None, "", 0.0, False, 0)

    start_idx, _ = _find_preamble(
        signal, samples_per_bit, freq_0=freq_0, freq_1=freq_1, sample_rate=sample_rate
    )
    if start_idx is None:
        return DecodeResult(False, None, "", 0.0, False, 0)

    bits = []
    confidences = []
    idx = start_idx + samples_per_bit * len(PREAMBLE)
    while idx + samples_per_bit <= len(signal):
        window = signal[idx : idx + samples_per_bit]
        bit, conf = _classify_bit(window, freq_0=freq_0, freq_1=freq_1, sample_rate=sample_rate)
        bits.append(bit)
        confidences.append(conf)
        idx += samples_per_bit
        if len(bits) % 8 == 0 and conf < 0.15:
            bits = bits[: len(bits) - (len(bits) % 8) or len(bits)]
            break

    bitstring = "".join(bits)
    usable_len = (len(bitstring) // 8) * 8
    bitstring = bitstring[:usable_len]

    text = _bits_to_text(bitstring) if bitstring else None
    if text:
        text = text.rstrip("?")
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return DecodeResult(
        success=bool(text and len(text) > 0),
        text=text,
        bits=bitstring,
        confidence=avg_conf,
        preamble_found=True,
        bit_count=len(bitstring),
    )


def play_ultrasonic_signal(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    blocking: bool = True,
    output_device: int | None = None,
) -> bool:
    device = SPEAKER_DEVICE_ID if output_device is None else output_device
    stats = {
        "samples": len(signal),
        "sample_rate": sample_rate,
        "duration_sec": len(signal) / sample_rate if sample_rate else 0.0,
        "rms": float(np.sqrt(np.mean(signal * signal))) if len(signal) else 0.0,
        "peak": float(np.max(np.abs(signal))) if len(signal) else 0.0,
    }
    logger.info(
        "[SIM] playback request device=%s samples=%d sr=%d duration=%.2fs rms=%.6f peak=%.6f blocking=%s",
        device,
        stats["samples"],
        stats["sample_rate"],
        stats["duration_sec"],
        stats["rms"],
        stats["peak"],
        blocking,
    )
    try:
        import sounddevice as sd

        logger.info("[SIM] playback started device=%s", device)
        sd.play(signal, samplerate=sample_rate, device=device)
        if blocking:
            sd.wait()
            logger.info("[SIM] playback finished device=%s", device)
        return True
    except Exception as e:
        logger.warning("[SIM] sounddevice speaker playback failed: %s", e)
        try:
            import tempfile
            import winsound

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            save_signal_to_wav(tmp_path, signal, sample_rate)
            flags = winsound.SND_FILENAME if blocking else (winsound.SND_FILENAME | winsound.SND_ASYNC)
            winsound.PlaySound(tmp_path, flags)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            logger.info("[SIM] fallback winsound playback device=default blocking=%s", blocking)
            return True
        except Exception as e2:
            logger.error("[SIM] fallback winsound playback failed: %s", e2)
            return False


def encode_and_emit(text: str, pad_silence_sec: float = 0.3, blocking: bool = True) -> bool:
    signal = encode_text_to_signal(text, pad_silence_sec=pad_silence_sec)
    return play_ultrasonic_signal(signal, sample_rate=SAMPLE_RATE, blocking=blocking)


def save_signal_to_wav(path: str, signal: np.ndarray, sample_rate: int = SAMPLE_RATE):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def decode_audio_file(path: str) -> DecodeResult:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return decode_signal_to_text(signal, sample_rate=sr)


def _text_to_bits(text: str) -> str:
    return "".join(f"{ord(c):08b}" for c in text)


def _bits_to_text(bits: str) -> str:
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i : i + 8]
        code = int(byte, 2)
        if 32 <= code <= 126 or code in (10, 13):
            chars.append(chr(code))
        else:
            chars.append("?")
    return "".join(chars)


def _bits_to_tone(
    bits: str,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    bit_duration: float = BIT_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    samples_per_bit = int(sample_rate * bit_duration)
    t = np.linspace(0, bit_duration, samples_per_bit, endpoint=False)
    signal = []
    ramp = max(1, int(samples_per_bit * 0.1))
    for bit in bits:
        freq = freq_1 if bit == "1" else freq_0
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        envelope = np.ones_like(tone)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        signal.append(tone * envelope)
    return np.concatenate(signal)


def _goertzel_power(window: np.ndarray, freq: float, sample_rate: int = SAMPLE_RATE) -> float:
    n = len(window)
    if n == 0:
        return 0.0
    k = int(0.5 + n * freq / sample_rate)
    w = 2 * np.pi * k / n
    cos_w = np.cos(w)
    coeff = 2 * cos_w
    s_prev, s_prev2 = 0.0, 0.0
    for sample in window:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return float(max(0.0, power))


def _symbol_tone_power(
    window: np.ndarray,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    p0 = _goertzel_power(window, freq_0, sample_rate)
    p1 = _goertzel_power(window, freq_1, sample_rate)
    return max(p0, p1)


def _classify_bit(
    window: np.ndarray,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, float]:
    p0 = _goertzel_power(window, freq_0, sample_rate)
    p1 = _goertzel_power(window, freq_1, sample_rate)
    total = p0 + p1 + 1e-9
    if p1 > p0:
        return "1", float(p1 / total)
    return "0", float(p0 / total)


def _find_preamble(
    signal: np.ndarray,
    samples_per_bit: int,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    sample_rate: int = SAMPLE_RATE,
):
    # Forensic PC2 captures are often 15–20 s with the transmission starting
    # well after t=0 (e.g. ~11–19 s). A fixed 3 s window misses late preambles.
    search_span = len(signal)
    hop = max(1, samples_per_bit // 4)
    best_start, best_conf = None, 0.0

    peak_tone_power = 0.0
    for start in range(0, search_span - samples_per_bit, hop):
        window = signal[start : start + samples_per_bit]
        peak_tone_power = max(
            peak_tone_power,
            _symbol_tone_power(window, freq_0=freq_0, freq_1=freq_1, sample_rate=sample_rate),
        )
    min_preamble_tone = peak_tone_power * PREAMBLE_MIN_PEAK_FRACTION

    for start in range(0, search_span - samples_per_bit * len(PREAMBLE), hop):
        bits = []
        confs = []
        tone_powers = []
        idx = start
        ok = True
        for _ in range(len(PREAMBLE)):
            if idx + samples_per_bit > len(signal):
                ok = False
                break
            window = signal[idx : idx + samples_per_bit]
            bit, conf = _classify_bit(
                window,
                freq_0=freq_0,
                freq_1=freq_1,
                sample_rate=sample_rate,
            )
            bits.append(bit)
            confs.append(conf)
            tone_powers.append(
                _symbol_tone_power(window, freq_0=freq_0, freq_1=freq_1, sample_rate=sample_rate)
            )
            idx += samples_per_bit
        if ok and "".join(bits) == PREAMBLE:
            mean_tone_power = float(np.mean(tone_powers))
            # Bit classification can match PREAMBLE on noise; require real tone energy.
            if mean_tone_power < min_preamble_tone:
                continue
            avg_conf = float(np.mean(confs))
            if avg_conf > best_conf:
                best_conf, best_start = avg_conf, start

    return best_start, best_conf
