#!/usr/bin/env python3
"""Diagnose whether PC1 speaker output reproduces ultrasonic test frequencies.

Records from the local microphone while playing each test tone so we can see
whether the speaker→air→mic path carries 18.5 kHz / 20.5 kHz energy.

Usage examples:
  python backend/scripts/test_speaker_ultrasonic_output.py
  python backend/scripts/test_speaker_ultrasonic_output.py --device 4 --input-device 1
  python backend/scripts/test_speaker_ultrasonic_output.py --device 6 --input-device 2
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import welch

SAMPLE_RATE = 44100
TONE_DURATION_SEC = 5.0
TONE_AMPLITUDE = 0.4
PRE_ROLL_SEC = 0.3
POST_ROLL_SEC = 0.5
WELCH_NPERSEG = 16384

TEST_FREQUENCIES = (18500, 20500)


def _generate_tone(freq_hz: float) -> np.ndarray:
    n = int(SAMPLE_RATE * TONE_DURATION_SEC)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (TONE_AMPLITUDE * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _record_while_playing(
    tone: np.ndarray,
    input_device: int,
    output_device: int,
) -> np.ndarray:
    total_sec = PRE_ROLL_SEC + TONE_DURATION_SEC + POST_ROLL_SEC
    n_samples = int(total_sec * SAMPLE_RATE)
    recorded = np.zeros((n_samples, 1), dtype=np.float32)
    rec_done = threading.Event()

    def _record() -> None:
        nonlocal recorded
        recorded = sd.rec(
            n_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            device=input_device,
            dtype="float32",
        )
        sd.wait()
        rec_done.set()

    rec_thread = threading.Thread(target=_record, daemon=True)
    rec_thread.start()
    time.sleep(0.1)
    time.sleep(PRE_ROLL_SEC)
    sd.play(tone, samplerate=SAMPLE_RATE, device=output_device)
    sd.wait()
    rec_done.wait(timeout=total_sec + 5.0)
    return np.asarray(recorded, dtype=np.float64).reshape(-1)


def _prepare_recording_for_analysis(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert to float64, report non-finite samples, and sanitize for RMS/PSD."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    non_finite_mask = ~np.isfinite(signal)
    non_finite_count = int(np.count_nonzero(non_finite_mask))
    if non_finite_count:
        signal = signal.copy()
        signal[non_finite_mask] = 0.0
    signal = np.clip(signal, -1.0, 1.0)
    return signal, non_finite_count


def _band_peak(f: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> tuple[float, float]:
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0, (f_lo + f_hi) / 2.0
    band_f = f[mask]
    band_psd = psd[mask]
    idx = int(np.argmax(band_psd))
    return float(band_psd[idx]), float(band_f[idx])


def _psd_at_target(f: np.ndarray, psd: np.ndarray, target_hz: float) -> float:
    idx = int(np.argmin(np.abs(f - target_hz)))
    return float(psd[idx])


def _analyze_recording(signal: np.ndarray, target_hz: int) -> dict:
    signal, non_finite_count = _prepare_recording_for_analysis(signal)
    f, psd = welch(signal, fs=SAMPLE_RATE, nperseg=WELCH_NPERSEG)
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

    if target_hz == 18500:
        target_lo, target_hi = 18000.0, 19000.0
        other_lo, other_hi = 20000.0, 21000.0
    else:
        target_lo, target_hi = 20000.0, 21000.0
        other_lo, other_hi = 18000.0, 19000.0

    target_psd = _psd_at_target(f, psd, float(target_hz))
    band_max, detected_hz = _band_peak(f, psd, target_lo, target_hi)
    other_max, _ = _band_peak(f, psd, other_lo, other_hi)
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    band_diff = target_psd - other_max

    # Target tone is present when its band dominates the other ultrasonic band.
    detected = target_psd > other_max * 10.0 and band_max > other_max * 10.0

    return {
        "target_hz": target_hz,
        "detected_hz": detected_hz,
        "target_psd": target_psd,
        "band_max": band_max,
        "other_max": other_max,
        "band_diff": band_diff,
        "rms": rms,
        "detected": detected,
        "non_finite_count": non_finite_count,
    }


def _print_result(
    metrics: dict,
    input_device: int,
    output_device: int,
    target_lo: int,
    target_hi: int,
    other_lo: int,
    other_hi: int,
) -> None:
    target_khz = metrics["target_hz"] / 1000.0
    print(f"=== {target_khz:g} kHz TEST ===")
    print(f"Input device : {input_device}")
    print(f"Output device: {output_device}")
    print(f"Target       : {metrics['target_hz']} Hz")
    print(f"Detected     : {metrics['detected_hz']:.0f} Hz")
    if metrics["non_finite_count"]:
        print(f"Non-finite   : {metrics['non_finite_count']} samples replaced for analysis")
    print(f"Target PSD   : {metrics['target_psd']:.2e}")
    print(f"{target_lo//1000}-{target_hi//1000}k max   : {metrics['band_max']:.2e}")
    print(f"{other_lo//1000}-{other_hi//1000}k max   : {metrics['other_max']:.2e}")
    print(f"RMS          : {metrics['rms']:.2e}")
    print(f"Band diff    : {metrics['band_diff']:.2e}")
    print(f"RESULT       : {'DETECTED' if metrics['detected'] else 'NOT DETECTED'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play ultrasonic tones and verify speaker output via local microphone"
    )
    parser.add_argument("--device", type=int, default=4, help="Output device ID (default: 4)")
    parser.add_argument(
        "--input-device", type=int, default=1, help="Input device ID (default: 1)"
    )
    args = parser.parse_args()

    print("Speaker ultrasonic output diagnostic")
    print(f"Sample rate  : {SAMPLE_RATE} Hz")
    print(f"Tone duration: {TONE_DURATION_SEC:.1f} s (amplitude {TONE_AMPLITUDE})")
    print(f"Pre-roll     : {PRE_ROLL_SEC:.1f} s before playback")
    print()

    all_detected = True
    for target_hz in TEST_FREQUENCIES:
        tone = _generate_tone(float(target_hz))
        print(f"Recording + playing {target_hz} Hz ...", flush=True)
        try:
            recording = _record_while_playing(tone, args.input_device, args.device)
        except Exception as exc:
            print(f"ERROR during {target_hz} Hz test: {exc}", file=sys.stderr)
            return 1

        metrics = _analyze_recording(recording, target_hz)
        if target_hz == 18500:
            target_lo, target_hi, other_lo, other_hi = 18000, 19000, 20000, 21000
        else:
            target_lo, target_hi, other_lo, other_hi = 20000, 21000, 18000, 19000

        _print_result(metrics, args.input_device, args.device, target_lo, target_hi, other_lo, other_hi)
        all_detected = all_detected and metrics["detected"]

    print("=== CONCLUSION ===")
    if all_detected:
        print("Both test frequencies were DETECTED on the local microphone.")
    else:
        print("At least one test frequency was NOT DETECTED.")
        print("Check output device, volume, speaker capability, and mic placement.")
    return 0 if all_detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
