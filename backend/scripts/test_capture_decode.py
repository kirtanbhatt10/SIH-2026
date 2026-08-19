#!/usr/bin/env python3
"""Concise decode diagnostic for transmitter WAVs and PC2 forensic captures."""

from __future__ import annotations

import argparse
import os
import sys
import wave

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import BIT_DURATION, PREAMBLE, SAMPLE_RATE  # noqa: E402
from backend.services.payload_service import (  # noqa: E402
    _find_preamble,
    decode_audio_file,
)


def _duration_sec(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _preamble_start_sec(path: str) -> tuple[float | None, float]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    import numpy as np

    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    samples_per_bit = int(sr * BIT_DURATION)
    start_idx, conf = _find_preamble(signal, samples_per_bit, sample_rate=sr)
    if start_idx is None:
        return None, conf
    return start_idx / sr, conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode diagnostic for capture WAV files")
    parser.add_argument("wav", help="Path to mono WAV file")
    args = parser.parse_args()

    path = os.path.abspath(args.wav)
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}")
        return 1

    duration = _duration_sec(path)
    preamble_t, preamble_conf = _preamble_start_sec(path)
    result = decode_audio_file(path)

    print(f"file              : {path}")
    print(f"duration_sec      : {duration:.3f}")
    print(f"preamble_start_sec: {preamble_t if preamble_t is not None else 'NOT FOUND'}")
    print(f"preamble_conf     : {preamble_conf:.4f}")
    print(f"preamble_found    : {result.preamble_found}")
    print(f"success           : {result.success}")
    print(f"bit_count         : {result.bit_count}")
    print(f"confidence        : {result.confidence:.4f}")
    print(f"text              : {result.text!r}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
