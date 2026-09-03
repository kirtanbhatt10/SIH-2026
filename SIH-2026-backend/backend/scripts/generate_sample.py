"""Generate the official Backend 2 reference WAV sample and metadata.

Reproduction:
    python backend/scripts/generate_sample.py

Output (default):
    samples/backend2/sample.wav
    samples/backend2/sample.json
"""

import argparse
import json
import os
import sys
import wave

import numpy as np

# Allow running as script from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, PREAMBLE, SAMPLE_RATE
from backend.services.payload_service import (
    decode_audio_file,
    encode_text_to_signal,
    save_signal_to_wav,
)

DEFAULT_PAYLOAD = "SIH 2026 BACKEND 2"
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "samples", "backend2")


def _text_to_bits(text: str) -> str:
    return "".join(f"{ord(c):08b}" for c in text)


def _analyze_wav(path: str) -> dict:
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        signal = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0

    duration = n_frames / sample_rate
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    has_nan = bool(np.any(np.isnan(signal)))
    has_inf = bool(np.any(np.isinf(signal)))
    clipped = bool(np.any(signal >= 1.0) or np.any(signal <= -1.0))

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_count": n_frames,
        "duration_seconds": round(duration, 6),
        "peak_amplitude": round(peak, 6),
        "rms_amplitude": round(rms, 6),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "clipped_at_full_scale": clipped,
        "non_zero_signal": bool(rms > 1e-9),
    }


def build_metadata(
    payload: str,
    wav_path: str,
    freq_0: int,
    freq_1: int,
    bit_duration: float,
    pad_silence_sec: float,
    sample_rate: int,
) -> dict:
    wav_info = _analyze_wav(wav_path)
    encoded_bits = PREAMBLE + _text_to_bits(payload)

    return {
        "sample_file": os.path.basename(wav_path),
        "payload_text": payload,
        "payload_description": "controlled Backend 2 reference test payload",
        "encoding": {
            "format": "UTF-8 text to 8-bit ASCII bits (MSB first per byte)",
            "preamble_bits": PREAMBLE,
            "preamble_description": "8-bit alternating pattern for sync",
            "checksum": "not_applicable",
            "encoded_bit_count": len(encoded_bits),
            "encoded_bits_preview": encoded_bits[:32] + ("..." if len(encoded_bits) > 32 else ""),
        },
        "modulation": "BFSK",
        "carrier_frequency_0_hz": freq_0,
        "carrier_frequency_1_hz": freq_1,
        "bit_0_frequency_hz": freq_0,
        "bit_1_frequency_hz": freq_1,
        "symbol_duration_seconds": bit_duration,
        "bit_duration_ms": round(bit_duration * 1000, 3),
        "amplitude_tone_peak": 0.5,
        "amplitude_envelope": "10% linear ramp per symbol",
        "pad_silence_seconds": pad_silence_sec,
        "sample_rate": wav_info["sample_rate"],
        "channels": wav_info["channels"],
        "sample_width_bytes": wav_info["sample_width_bytes"],
        "duration_seconds": wav_info["duration_seconds"],
        "sample_count": wav_info["sample_count"],
        "peak_amplitude": wav_info["peak_amplitude"],
        "rms_amplitude": wav_info["rms_amplitude"],
        "audio_quality": {
            "has_nan": wav_info["has_nan"],
            "has_inf": wav_info["has_inf"],
            "clipped_at_full_scale": wav_info["clipped_at_full_scale"],
            "non_zero_signal": wav_info["non_zero_signal"],
        },
        "reproduction_command": "python backend/scripts/generate_sample.py",
        "decoder_self_test": "not_applicable",
        "hardware_validation": "not_applicable",
        "notes": (
            "Frequencies and timing are software simulator parameters. "
            "Speaker/microphone hardware behavior is TO BE MEASURED separately."
        ),
    }


def generate(
    payload: str = DEFAULT_PAYLOAD,
    out_dir: str = DEFAULT_OUT_DIR,
    freq_0: int = FREQ_0,
    freq_1: int = FREQ_1,
    bit_duration: float = BIT_DURATION,
    pad_silence_sec: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "sample.wav")
    json_path = os.path.join(out_dir, "sample.json")

    signal = encode_text_to_signal(
        payload,
        pad_silence_sec=pad_silence_sec,
        freq_0=freq_0,
        freq_1=freq_1,
        bit_duration=bit_duration,
        sample_rate=sample_rate,
    )
    save_signal_to_wav(wav_path, signal, sample_rate)

    decode_result = decode_audio_file(wav_path)
    metadata = build_metadata(
        payload, wav_path, freq_0, freq_1, bit_duration, pad_silence_sec, sample_rate
    )
    metadata["decoder_self_test"] = {
        "original_payload": payload,
        "decoded_payload": decode_result.text,
        "success": decode_result.success,
        "confidence": round(decode_result.confidence, 6),
        "preamble_found": decode_result.preamble_found,
        "bit_count": decode_result.bit_count,
        "payload_match": decode_result.text == payload,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return wav_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Generate Backend 2 reference WAV sample")
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD, help="Payload text to encode")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--freq-0", type=int, default=FREQ_0, help="Hz for bit 0")
    parser.add_argument("--freq-1", type=int, default=FREQ_1, help="Hz for bit 1")
    parser.add_argument("--bit-duration-ms", type=float, default=BIT_DURATION * 1000)
    parser.add_argument("--pad-silence-sec", type=float, default=0.5)
    args = parser.parse_args()

    wav_path, json_path = generate(
        payload=args.payload,
        out_dir=args.out_dir,
        freq_0=args.freq_0,
        freq_1=args.freq_1,
        bit_duration=args.bit_duration_ms / 1000.0,
        pad_silence_sec=args.pad_silence_sec,
    )
    print(f"WAV: {wav_path}")
    print(f"Metadata: {json_path}")


if __name__ == "__main__":
    main()
