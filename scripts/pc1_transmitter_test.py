#!/usr/bin/env python3
"""
PC1 Acoustic Transmitter — generate encoded BFSK WAV using existing payload service.

Example:
    python scripts/pc1_transmitter_test.py --payload "SIH_PC1_PC2_TEST"
    python scripts/pc1_transmitter_test.py --payload "HELLO" --play
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.payload_service import (
    encode_text_to_signal,
    save_signal_to_wav,
    _text_to_bits,
)
from backend.models.data_schemas import GeneratePayloadRequest

PREAMBLE = "10101010"


def _play_wav_file(
    wav_path: str,
    device: int | None,
) -> tuple[int, str]:
    """Load WAV with soundfile, play via sounddevice, wait for completion."""
    import sounddevice as sd
    import soundfile as sf

    data, sample_rate = sf.read(wav_path, dtype="float32")
    wav_channels = 1 if data.ndim == 1 else int(data.shape[1])

    if device is not None:
        dev_info = sd.query_devices(device)
        device_line = f"{device} - {dev_info['name']}"
    else:
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0:
            dev_info = sd.query_devices(default_out)
            device_line = f"{default_out} - {dev_info['name']}"
        else:
            device_line = "default"

    print(f"WAV Channels : {wav_channels}")
    print(f"Output Device: {device_line}")
    print("Playback       : STARTED")

    if device is not None:
        sd.play(data, sample_rate, device=device)
    else:
        sd.play(data, sample_rate)
    sd.wait()

    return wav_channels, device_line


def main() -> int:
    parser = argparse.ArgumentParser(description="PC1 acoustic transmitter test")
    parser.add_argument("--payload", default="SIH_PC1_PC2_TEST", help="Text payload to encode")
    parser.add_argument("--freq-0", type=int, default=FREQ_0, help="BFSK frequency for bit 0 (Hz)")
    parser.add_argument("--freq-1", type=int, default=FREQ_1, help="BFSK frequency for bit 1 (Hz)")
    parser.add_argument(
        "--bit-duration-ms",
        type=float,
        default=BIT_DURATION * 1000.0,
        help="Bit duration in milliseconds",
    )
    parser.add_argument("--play", action="store_true", help="Play WAV through default speaker")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Output device ID for sounddevice playback (optional)",
    )
    parser.add_argument(
        "--out-dir",
        default=STORAGE_DIR,
        help="Directory for generated WAV files",
    )
    args = parser.parse_args()

    # Validate like API (max 256 chars)
    req = GeneratePayloadRequest(
        text=args.payload,
        freq_0_hz=args.freq_0,
        freq_1_hz=args.freq_1,
        bit_duration_ms=args.bit_duration_ms,
    )

    encoding_status = "PASS"
    bfsk_status = "PASS"
    wav_status = "PASS"
    speaker_status = "NOT_AVAILABLE"
    playback_status: str | None = None
    device_line: str | None = None
    wav_channels: int | None = None
    playback_error: str | None = None

    try:
        bit_duration_sec = req.bit_duration_ms / 1000.0
        signal = encode_text_to_signal(
            req.text,
            freq_0=req.freq_0_hz,
            freq_1=req.freq_1_hz,
            bit_duration=bit_duration_sec,
            sample_rate=SAMPLE_RATE,
        )
    except Exception as e:
        encoding_status = f"FAIL ({e})"
        bfsk_status = "FAIL"
        wav_status = "FAIL"
        signal = None

    payload_id = str(uuid.uuid4())[:8]
    wav_path = os.path.join(args.out_dir, f"{payload_id}.wav")

    if signal is not None:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            save_signal_to_wav(wav_path, signal, SAMPLE_RATE)
            if not os.path.isfile(wav_path):
                wav_status = "FAIL (file not created)"
        except Exception as e:
            wav_status = f"FAIL ({e})"

    if args.play and signal is not None and wav_status == "PASS":
        try:
            wav_channels, device_line = _play_wav_file(wav_path, args.device)
            speaker_status = "PASS"
            playback_status = "COMPLETE"
        except Exception as e:
            speaker_status = "FAIL"
            playback_status = "FAILED"
            playback_error = str(e)

    bit_count = len(PREAMBLE + _text_to_bits(req.text))
    duration_sec = len(signal) / SAMPLE_RATE if signal is not None else 0.0

    print("========================================")
    print("PC1 ACOUSTIC TRANSMITTER")
    print("========================================")
    print(f"Payload        : {req.text}")
    print(f"Payload ID     : {payload_id}")
    print(f"Encoding       : {encoding_status}")
    print(f"BFSK Modulation: {bfsk_status}")
    print(f"WAV Generation : {wav_status}")
    print(f"Speaker        : {speaker_status}")
    if args.play:
        if wav_channels is not None:
            print(f"WAV Channels   : {wav_channels}")
        if device_line is not None:
            print(f"Output Device  : {device_line}")
        if playback_status is not None:
            print(f"Playback       : {playback_status}")
        if playback_error is not None:
            print(f"Error          : {playback_error}")
    print(f"WAV Path       : {os.path.abspath(wav_path)}")
    print(f"Sample Rate    : {SAMPLE_RATE}")
    print(f"Duration (s)   : {duration_sec:.3f}")
    print(f"Frequency 0    : {req.freq_0_hz} Hz")
    print(f"Frequency 1    : {req.freq_1_hz} Hz")
    print(f"Bit duration   : {req.bit_duration_ms} ms")
    print(f"Bit count      : {bit_count}")
    print("========================================")

    if encoding_status != "PASS" or wav_status != "PASS":
        return 1
    if args.play and speaker_status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
