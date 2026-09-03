#!/usr/bin/env python3
"""Silence-only microphone capture to verify sounddevice raw recording pipeline.

Usage:
  python backend/scripts/test_mic_raw_capture.py
  python backend/scripts/test_mic_raw_capture.py --input-device 1 --sample-rate 44100
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import sounddevice as sd

DEFAULT_INPUT_DEVICE = 1
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_SEC = 3.0


def _raw_stats(raw: np.ndarray) -> dict:
    arr = np.asarray(raw).reshape(-1)
    finite_mask = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite_mask))
    non_finite_count = int(arr.size - finite_count)
    if finite_count == 0:
        return {
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
            "raw_min": float("nan"),
            "raw_max": float("nan"),
            "raw_mean": float("nan"),
            "raw_std": float("nan"),
            "raw_rms": float("nan"),
            "finite_count": finite_count,
            "non_finite_count": non_finite_count,
            "sane": False,
            "reason": "no finite samples",
        }

    vals = arr[finite_mask].astype(np.float64, copy=False)
    peak = float(np.max(np.abs(vals)))
    sane = peak <= 1.5 and non_finite_count == 0
    reason = "ok" if sane else (
        f"peak {peak:.3e} outside normalized float32 range [-1, 1]"
        if peak > 1.5
        else "non-finite samples present"
    )
    return {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "raw_min": float(np.min(vals)),
        "raw_max": float(np.max(vals)),
        "raw_mean": float(np.mean(vals)),
        "raw_std": float(np.std(vals)),
        "raw_rms": float(np.sqrt(np.mean(np.square(vals)))),
        "finite_count": finite_count,
        "non_finite_count": non_finite_count,
        "sane": sane,
        "reason": reason,
    }


def _print_stats(stats: dict) -> None:
    print(f"dtype          : {stats['dtype']}")
    print(f"shape          : {stats['shape']}")
    print(f"min            : {stats['raw_min']:.6e}")
    print(f"max            : {stats['raw_max']:.6e}")
    print(f"mean           : {stats['raw_mean']:.6e}")
    print(f"std            : {stats['raw_std']:.6e}")
    print(f"RMS            : {stats['raw_rms']:.6e}")
    print(f"finite count   : {stats['finite_count']}")
    print(f"non-finite cnt : {stats['non_finite_count']}")
    print(f"sane           : {stats['sane']} ({stats['reason']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Silence mic capture pipeline check")
    parser.add_argument("--input-device", type=int, default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument(
        "--save",
        default="diagnostic_silence.wav",
        help="WAV path (written only when recording is sane)",
    )
    args = parser.parse_args()

    frames = int(args.duration * args.sample_rate)
    print("MIC RAW CAPTURE (silence, no playback)")
    print("====================================")
    print(f"Input device : {args.input_device}")
    print(f"Sample rate  : {args.sample_rate} Hz")
    print(f"Duration     : {args.duration:.1f} s")
    print(f"Recording {args.duration:.1f} s ...", flush=True)

    try:
        buf = sd.rec(
            frames,
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=args.input_device,
        )
        sd.wait()
        raw = np.asarray(buf, dtype=np.float32).reshape(-1).copy()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stats = _raw_stats(raw)
    print()
    _print_stats(stats)

    if stats["sane"]:
        try:
            from scipy.io.wavfile import write

            pcm = raw.astype(np.float32)
            write(args.save, args.sample_rate, pcm)
            print(f"saved          : {os.path.abspath(args.save)}")
        except Exception as exc:
            print(f"WARN: could not save WAV: {exc}", file=sys.stderr)
            return 1
    else:
        print("WAV not saved - raw recording is not in a sane normalized range.")

    return 0 if stats["sane"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
