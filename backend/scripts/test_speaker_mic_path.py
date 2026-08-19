#!/usr/bin/env python3
"""Speaker -> microphone acoustic path diagnostic for PC1 hardware verification.

Uses sd.playrec() so playback and recording share one synchronized PortAudio
stream. Concurrent sd.rec()+sd.play() in separate threads can corrupt buffers
and produce int16-scale or garbage sample values.

Usage:
  python backend/scripts/test_speaker_mic_path.py
  python backend/scripts/test_speaker_mic_path.py --input-device 1 --output-device 4
  python backend/scripts/test_speaker_mic_path.py --input-device 1 --output-device 4 --sample-rate 48000
  python backend/scripts/test_speaker_mic_path.py --list-devices
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd
from scipy.signal import welch

DEFAULT_INPUT_DEVICE = 1
DEFAULT_OUTPUT_DEVICE = 4
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_SEC = 5.0
DEFAULT_VOLUME = 0.4
PRE_ROLL_SEC = 0.3
POST_ROLL_SEC = 0.5
WELCH_NPERSEG = 16384
MIN_BAND_RATIO = 10.0
MIN_RMS_RATIO = 3.0
NORMALIZED_PEAK_LIMIT = 1.5

REF_FREQ_HZ = 1000
FREQ_18500_HZ = 18500
FREQ_20500_HZ = 20500


def _list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("=== HOST APIs ===")
    for i, api in enumerate(hostapis):
        print(f"API {i}: {api['name']}")
    print("\n=== SOUND DEVICES ===")
    for i, dev in enumerate(sd.query_devices()):
        api_name = hostapis[dev["hostapi"]]["name"]
        print(
            f"[{i:2d}] {dev['name']:<50s} | "
            f"In: {dev['max_input_channels']} | Out: {dev['max_output_channels']} | "
            f"Default SR: {dev['default_samplerate']:.0f} Hz | HostAPI: {api_name}"
        )
    print(f"\nDefault devices (input, output): {sd.default.device}")


def _generate_tone(freq_hz: float, duration_sec: float, volume: float, sample_rate: int) -> np.ndarray:
    n = int(sample_rate * duration_sec)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (volume * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _raw_recording_stats(raw: np.ndarray) -> dict:
    """Stats on the preserved raw buffer immediately after sd.wait().copy()."""
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
            "raw_sane": False,
            "raw_reason": "no finite samples",
        }

    vals = arr[finite_mask].astype(np.float64, copy=False)
    peak = float(np.max(np.abs(vals)))
    if non_finite_count:
        sane, reason = False, f"{non_finite_count} non-finite samples"
    elif peak > NORMALIZED_PEAK_LIMIT:
        sane = False
        if peak > 100.0:
            reason = f"peak {peak:.3e} looks corrupted (expected normalized float32 in [-1, 1])"
        else:
            reason = f"peak {peak:.3e} outside normalized float32 range"
    else:
        sane, reason = True, "normalized float32 range"

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
        "raw_sane": sane,
        "raw_reason": reason,
    }


def _print_raw_stats(stats: dict) -> None:
    print("   RAW recording (immediately after capture):")
    print(f"      dtype       : {stats['dtype']}")
    print(f"      shape       : {stats['shape']}")
    print(f"      min         : {stats['raw_min']:.6e}")
    print(f"      max         : {stats['raw_max']:.6e}")
    print(f"      mean        : {stats['raw_mean']:.6e}")
    print(f"      std         : {stats['raw_std']:.6e}")
    print(f"      RMS         : {stats['raw_rms']:.6e}")
    print(f"      finite cnt  : {stats['finite_count']}")
    print(f"      non-finite  : {stats['non_finite_count']}")
    print(f"      pipeline ok : {stats['raw_sane']} ({stats['raw_reason']})")


def _record_while_playing(
    tone: np.ndarray,
    input_device: int,
    output_device: int,
    sample_rate: int,
    duration_sec: float,
) -> tuple[np.ndarray, dict]:
    """Play tone through speaker while recording mic on one synchronized playrec stream."""
    pre_samples = int(PRE_ROLL_SEC * sample_rate)
    post_samples = int(POST_ROLL_SEC * sample_rate)
    playback = np.concatenate(
        [
            np.zeros(pre_samples, dtype=np.float32),
            np.asarray(tone, dtype=np.float32).reshape(-1),
            np.zeros(post_samples, dtype=np.float32),
        ]
    ).reshape(-1, 1)

    recorded = sd.playrec(
        playback,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=(input_device, output_device),
    )
    sd.wait()

    raw = np.asarray(recorded, dtype=np.float64).reshape(-1).copy()
    return raw, _raw_recording_stats(raw)


def _sanitize_for_analysis(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy to float64 and replace non-finite samples only (no amplitude clipping)."""
    clean = np.asarray(signal, dtype=np.float64).reshape(-1).copy()
    bad = ~np.isfinite(clean)
    count = int(np.count_nonzero(bad))
    if count:
        clean[bad] = 0.0
    return clean, count


def _safe_rms(signal: np.ndarray) -> float:
    vals = np.asarray(signal, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


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


def _analyze_tone(
    raw_recording: np.ndarray,
    raw_stats: dict,
    sample_rate: int,
    target_hz: float,
    target_band: tuple[float, float],
) -> dict:
    signal, non_finite_count = _sanitize_for_analysis(raw_recording)
    duration_sec = len(signal) / sample_rate
    peak_amplitude = float(np.max(np.abs(signal))) if signal.size else 0.0
    pipeline_ok = raw_stats["raw_sane"]
    saturated = peak_amplitude > NORMALIZED_PEAK_LIMIT

    pre_samples = int(PRE_ROLL_SEC * sample_rate)
    preroll = signal[:pre_samples] if pre_samples > 0 else signal[:1]
    preroll_rms = _safe_rms(preroll)
    rms = _safe_rms(signal)

    nperseg = min(WELCH_NPERSEG, len(signal))
    f, psd = welch(signal, fs=sample_rate, nperseg=nperseg)
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

    target_lo, target_hi = target_band
    target_psd = _psd_at_target(f, psd, target_hz)
    target_band_max, peak_hz = _band_peak(f, psd, target_lo, target_hi)
    band_18_19, _ = _band_peak(f, psd, 18000.0, 19000.0)
    band_20_21, _ = _band_peak(f, psd, 20000.0, 21000.0)

    if not pipeline_ok:
        detected = False
    elif target_hz == REF_FREQ_HZ:
        other_ultra = max(band_18_19, band_20_21)
        band_ratio = target_band_max / (other_ultra + 1e-30)
        detected = (
            not saturated
            and target_band_max > other_ultra * MIN_BAND_RATIO
            and rms > preroll_rms * MIN_RMS_RATIO
            and np.isfinite(rms)
            and rms > 1e-6
        )
    elif target_hz == FREQ_18500_HZ:
        other_band = band_20_21
        band_ratio = target_band_max / (other_band + 1e-30)
        detected = (
            not saturated
            and target_band_max > other_band * MIN_BAND_RATIO
            and target_psd > other_band * MIN_BAND_RATIO
        )
    else:
        other_band = band_18_19
        band_ratio = target_band_max / (other_band + 1e-30)
        detected = (
            not saturated
            and target_band_max > other_band * MIN_BAND_RATIO
            and target_psd > other_band * MIN_BAND_RATIO
        )

    return {
        "target_hz": target_hz,
        "peak_hz": peak_hz,
        "target_psd": target_psd,
        "target_band_max": target_band_max,
        "band_18_19": band_18_19,
        "band_20_21": band_20_21,
        "band_ratio": band_ratio,
        "rms": rms,
        "preroll_rms": preroll_rms,
        "duration_sec": duration_sec,
        "non_finite_count": non_finite_count,
        "peak_amplitude": peak_amplitude,
        "saturated": saturated,
        "pipeline_ok": pipeline_ok,
        "raw_stats": raw_stats,
        "detected": detected,
    }


def _apply_ultrasonic_baseline(metrics: dict, baseline_ultra_psd: float) -> dict:
    if metrics["target_hz"] == REF_FREQ_HZ:
        return metrics
    above_ref = metrics["target_band_max"] > baseline_ultra_psd * MIN_BAND_RATIO
    metrics = dict(metrics)
    metrics["detected"] = metrics["detected"] and above_ref
    metrics["baseline_ultra_psd"] = baseline_ultra_psd
    return metrics


def _fmt_rms(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.2e}"


def _result_line(metrics: dict) -> str:
    if not metrics.get("pipeline_ok", True):
        return "NOT DETECTED (raw recording pipeline invalid)"
    if metrics.get("saturated"):
        return "NOT DETECTED (out-of-range samples)"
    return "DETECTED" if metrics["detected"] else "NOT DETECTED"


def _run_test(
    label: str,
    target_hz: float,
    target_band: tuple[float, float],
    input_device: int,
    output_device: int,
    sample_rate: int,
    duration_sec: float,
    volume: float,
) -> dict:
    print(f"Running {label} ...", flush=True)
    tone = _generate_tone(target_hz, duration_sec, volume, sample_rate)
    recording, raw_stats = _record_while_playing(
        tone, input_device, output_device, sample_rate, duration_sec
    )
    _print_raw_stats(raw_stats)
    metrics = _analyze_tone(recording, raw_stats, sample_rate, target_hz, target_band)
    metrics["raw_stats"] = raw_stats
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker -> microphone acoustic path diagnostic")
    parser.add_argument("--input-device", type=int, default=DEFAULT_INPUT_DEVICE)
    parser.add_argument("--output-device", type=int, default=DEFAULT_OUTPUT_DEVICE)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--volume", type=float, default=DEFAULT_VOLUME)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    nyquist = args.sample_rate / 2.0
    if FREQ_20500_HZ >= nyquist:
        print(
            f"WARNING: sample rate {args.sample_rate} Hz cannot faithfully represent "
            f"{FREQ_20500_HZ} Hz (Nyquist={nyquist:.0f} Hz). Try --sample-rate 48000."
        )

    try:
        ref = _run_test(
            "1 kHz reference",
            REF_FREQ_HZ,
            (900.0, 1100.0),
            args.input_device,
            args.output_device,
            args.sample_rate,
            args.duration,
            args.volume,
        )
        print()
        baseline_ultra = max(ref["band_18_19"], ref["band_20_21"])

        ultra_185 = _apply_ultrasonic_baseline(
            _run_test(
                "18.5 kHz",
                FREQ_18500_HZ,
                (18000.0, 19000.0),
                args.input_device,
                args.output_device,
                args.sample_rate,
                args.duration,
                args.volume,
            ),
            baseline_ultra,
        )
        print()
        ultra_205 = _apply_ultrasonic_baseline(
            _run_test(
                "20.5 kHz",
                FREQ_20500_HZ,
                (20000.0, 21000.0),
                args.input_device,
                args.output_device,
                args.sample_rate,
                args.duration,
                args.volume,
            ),
            baseline_ultra,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    print("SPEAKER -> MICROPHONE TEST")
    print("=========================")
    print(f"Input device : {args.input_device} ({sd.query_devices(args.input_device)['name']})")
    print(f"Output device: {args.output_device} ({sd.query_devices(args.output_device)['name']})")
    print(f"Sample rate  : {args.sample_rate} Hz")
    print(f"Duration     : {args.duration:.1f} s per tone (volume {args.volume})")
    print(f"Pre-roll     : {PRE_ROLL_SEC:.1f} s silence before each tone (via playrec padding)")
    print(f"Capture mode : sd.playrec (single synchronized stream, immediate copy)")
    print()

    print("1 kHz reference:")
    print(f"   RMS          : {_fmt_rms(ref['rms'])}")
    print(f"   PSD          : {ref['target_psd']:.2e} (at {REF_FREQ_HZ} Hz)")
    print(f"   Peak         : {ref['peak_hz']:.0f} Hz")
    print(f"   18-19 kHz max: {ref['band_18_19']:.2e}")
    print(f"   20-21 kHz max: {ref['band_20_21']:.2e}")
    print(f"   Duration     : {ref['duration_sec']:.2f} s")
    print(f"   Non-finite   : {ref['non_finite_count']}")
    print(f"   Peak sample  : {ref['peak_amplitude']:.2e}")
    print(f"   RESULT       : {_result_line(ref)}")
    print()

    print("18.5 kHz:")
    print(f"   Peak         : {ultra_185['peak_hz']:.0f} Hz")
    print(f"   Target PSD   : {ultra_185['target_psd']:.2e}")
    print(f"   18-19 kHz max: {ultra_185['band_18_19']:.2e}")
    print(f"   20-21 kHz max: {ultra_185['band_20_21']:.2e}")
    print(f"   Band ratio   : {ultra_185['band_ratio']:.2e} (target vs 20-21 kHz)")
    print(f"   RMS          : {_fmt_rms(ultra_185['rms'])}")
    print(f"   Duration     : {ultra_185['duration_sec']:.2f} s")
    print(f"   Non-finite   : {ultra_185['non_finite_count']}")
    print(f"   Peak sample  : {ultra_185['peak_amplitude']:.2e}")
    print(f"   RESULT       : {_result_line(ultra_185)}")
    print()

    print("20.5 kHz:")
    print(f"   Peak         : {ultra_205['peak_hz']:.0f} Hz")
    print(f"   Target PSD   : {ultra_205['target_psd']:.2e}")
    print(f"   18-19 kHz max: {ultra_205['band_18_19']:.2e}")
    print(f"   20-21 kHz max: {ultra_205['band_20_21']:.2e}")
    print(f"   Band ratio   : {ultra_205['band_ratio']:.2e} (target vs 18-19 kHz)")
    print(f"   RMS          : {_fmt_rms(ultra_205['rms'])}")
    print(f"   Duration     : {ultra_205['duration_sec']:.2f} s")
    print(f"   Non-finite   : {ultra_205['non_finite_count']}")
    print(f"   Peak sample  : {ultra_205['peak_amplitude']:.2e}")
    print(f"   RESULT       : {_result_line(ultra_205)}")
    print()

    print("SUMMARY")
    print("-------")
    print(f"Raw pipeline : {'OK' if all(m['pipeline_ok'] for m in (ref, ultra_185, ultra_205)) else 'INVALID'}")
    print(f"1 kHz path   : {'OK' if ref['detected'] else 'FAIL'}")
    print(f"18.5 kHz path: {'OK' if ultra_185['detected'] else 'FAIL'}")
    print(f"20.5 kHz path: {'OK' if ultra_205['detected'] else 'FAIL'}")
    if not all(m["pipeline_ok"] for m in (ref, ultra_185, ultra_205)):
        print("Raw samples were outside normalized float32 range - fix capture before judging hardware.")
    elif not ref["detected"]:
        print("Audible reference not detected - check devices, volume, and mic placement first.")
    elif not ultra_185["detected"] and not ultra_205["detected"]:
        print("Audible path works but ultrasonics do not - likely hardware rolloff or audio processing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
