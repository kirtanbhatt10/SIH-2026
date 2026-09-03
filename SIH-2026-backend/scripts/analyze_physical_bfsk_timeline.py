#!/usr/bin/env python3
"""
Physical BFSK timeline forensic analyzer (read-only diagnostic).

Finds newest forensic_captures/pc2_capture_*.wav and generated_payloads/*.wav,
prints chronological STFT carrier timeline, collapsed segments, timing stats,
and optional plots. Does NOT modify production code or source audio.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORENSIC_DIR = os.path.join(_REPO_ROOT, "forensic_captures")
PAYLOAD_DIR = os.path.join(_REPO_ROOT, "generated_payloads")

FREQ_0 = 18500.0
FREQ_1 = 20500.0
BIT_DURATION_MS = 50.0
SAMPLE_RATE = 48000
NPERSEG = 2048
NOVERLAP = 1536
HOP = NPERSEG - NOVERLAP  # 512 samples
FRAME_STEP_SEC = HOP / SAMPLE_RATE  # ~10.67 ms
RATIO_THRESHOLD = 1.5
STRONG_PERCENTILE = 75.0


@dataclass
class WavMeta:
    path: str
    sample_rate: int
    channels: int
    samples: int
    duration_sec: float
    rms: float
    peak: float


@dataclass
class TimelineResult:
    meta: WavMeta
    bin_18500_hz: float
    bin_20500_hz: float
    times: np.ndarray
    p185: np.ndarray
    p205: np.ndarray
    states: list[str]
    segments: list[dict]
    timing: dict
    alternation: dict
    activity: dict
    verdict: dict


def _newest(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    return max(paths, key=os.path.getmtime) if paths else None


def load_mono(path: str) -> tuple[np.ndarray, int, int]:
    sr, data = wavfile.read(path)
    channels = 1 if data.ndim == 1 else int(data.shape[1])
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    elif data.dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr, channels


def wav_meta(path: str) -> WavMeta:
    audio, sr, ch = load_mono(path)
    return WavMeta(
        path=os.path.abspath(path),
        sample_rate=sr,
        channels=ch,
        samples=len(audio),
        duration_sec=len(audio) / sr,
        rms=float(np.sqrt(np.mean(audio**2))),
        peak=float(np.max(np.abs(audio))),
    )


def _nearest_bin_freq(target_hz: float, freqs: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(freqs[idx])


def _band_power(power: np.ndarray, freqs: np.ndarray, center_hz: float, half_width: float = 46.875) -> np.ndarray:
    """Sum power in +/- half_width Hz around center (one FFT bin width at 48k/2048)."""
    mask = (freqs >= center_hz - half_width) & (freqs <= center_hz + half_width)
    if not np.any(mask):
        return np.zeros(power.shape[1])
    return power[mask, :].sum(axis=0)


def _classify_frame(p185: float, p205: float, ratio_threshold: float = RATIO_THRESHOLD) -> str:
    if p185 <= 0 and p205 <= 0:
        return "NONE"
    if p185 > ratio_threshold * p205:
        return "18500"
    if p205 > ratio_threshold * p185:
        return "20500"
    return "AMBIGUOUS"


def _power_label(p: float, ref: float) -> str:
    if ref <= 0:
        return "LOW"
    return "HIGH" if p >= 0.25 * ref else "LOW"


def analyze_timeline(path: str) -> TimelineResult:
    audio, sr, ch = load_mono(path)
    meta = wav_meta(path)

    freqs, times, zxx = stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2

    bin_18500 = _nearest_bin_freq(FREQ_0, freqs)
    bin_20500 = _nearest_bin_freq(FREQ_1, freqs)

    p185 = _band_power(power, freqs, bin_18500)
    p205 = _band_power(power, freqs, bin_20500)

    p185_max = float(np.max(p185)) if len(p185) else 0.0
    p205_max = float(np.max(p205)) if len(p205) else 0.0

    states: list[str] = []
    for a, b in zip(p185, p205):
        states.append(_classify_frame(float(a), float(b)))

    # Collapse consecutive carrier states (18500 / 20500 only).
    segments: list[dict] = []
    i = 0
    while i < len(states):
        st = states[i]
        if st not in ("18500", "20500"):
            i += 1
            continue
        j = i + 1
        while j < len(states) and states[j] == st:
            j += 1
        start_t = float(times[i])
        end_t = float(times[j - 1]) + NPERSEG / sr
        dur_ms = (end_t - start_t) * 1000.0
        segments.append({
            "start": start_t,
            "end": end_t,
            "carrier": st,
            "duration_ms": dur_ms,
        })
        i = j

    durations_ms = [s["duration_ms"] for s in segments]
    if durations_ms:
        dur_arr = np.array(durations_ms)
        timing = {
            "n_18500": sum(1 for s in segments if s["carrier"] == "18500"),
            "n_20500": sum(1 for s in segments if s["carrier"] == "20500"),
            "transitions": max(0, len(segments) - 1),
            "median_ms": float(np.median(dur_arr)),
            "mean_ms": float(np.mean(dur_arr)),
            "min_ms": float(np.min(dur_arr)),
            "max_ms": float(np.max(dur_arr)),
            "std_ms": float(np.std(dur_arr)),
        }
    else:
        timing = {
            "n_18500": 0, "n_20500": 0, "transitions": 0,
            "median_ms": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0,
        }

    # Alternation analysis on collapsed carrier segments.
    alt_transitions = 0
    alt_possible = 0
    for k in range(1, len(segments)):
        if segments[k]["carrier"] != segments[k - 1]["carrier"]:
            alt_transitions += 1
        alt_possible += 1
    alt_ratio = alt_transitions / alt_possible if alt_possible > 0 else 0.0

    n_frames = len(states)
    n_185 = states.count("18500")
    n_205 = states.count("20500")
    n_amb = states.count("AMBIGUOUS")
    n_none = states.count("NONE")
    carrier_frames = n_185 + n_205

    alternation = {
        "alt_transitions": alt_transitions,
        "alt_ratio": alt_ratio,
        "pct_carrier": 100.0 * carrier_frames / max(n_frames, 1),
        "pct_ambiguous": 100.0 * n_amb / max(n_frames, 1),
        "pct_none": 100.0 * n_none / max(n_frames, 1),
        "deviation_ms": abs(timing["median_ms"] - BIT_DURATION_MS) if timing["median_ms"] > 0 else float("inf"),
    }

    # Activity window: frames where either band exceeds percentile threshold.
    combined = p185 + p205
    if len(combined) > 0:
        thresh = np.percentile(combined, STRONG_PERCENTILE)
        strong = combined >= thresh
        if np.any(strong):
            idx = np.where(strong)[0]
            act_start = float(times[idx[0]])
            act_end = float(times[idx[-1]]) + NPERSEG / sr
        else:
            act_start = act_end = 0.0
    else:
        act_start = act_end = 0.0

    activity = {
        "start": act_start,
        "end": act_end,
        "duration": act_end - act_start,
    }

    verdict = _build_verdict(timing, alternation, activity, meta)

    return TimelineResult(
        meta=meta,
        bin_18500_hz=bin_18500,
        bin_20500_hz=bin_20500,
        times=times,
        p185=p185,
        p205=p205,
        states=states,
        segments=segments,
        timing=timing,
        alternation=alternation,
        activity=activity,
        verdict=verdict,
    )


def _build_verdict(timing: dict, alternation: dict, activity: dict, meta: WavMeta) -> dict:
    has_185 = timing["n_18500"] >= 2
    has_205 = timing["n_20500"] >= 2
    ev_185 = "PASS" if has_185 else ("UNCERTAIN" if timing["n_18500"] >= 1 else "FAIL")
    ev_205 = "PASS" if has_205 else ("UNCERTAIN" if timing["n_20500"] >= 1 else "FAIL")

    seq_switch = (
        timing["transitions"] >= 3
        and alternation["alt_ratio"] >= 0.5
        and alternation["pct_carrier"] >= 10.0
    )
    ev_switch = "PASS" if seq_switch else ("UNCERTAIN" if timing["transitions"] >= 1 else "FAIL")

    timing_ok = (
        timing["median_ms"] > 0
        and abs(timing["median_ms"] - BIT_DURATION_MS) <= 30.0
        and timing["std_ms"] < 40.0
    )
    ev_timing = "PASS" if timing_ok else ("UNCERTAIN" if timing["median_ms"] > 0 else "FAIL")

  # duration correlation: activity within reasonable range of expected BFSK (~7.8s)
    dur_ok = 2.0 <= activity["duration"] <= 12.0
    ev_dur = "PASS" if dur_ok and activity["duration"] > 0 else "FAIL"

    passes = sum(1 for v in [ev_185, ev_205, ev_switch, ev_timing] if v == "PASS")

    if passes >= 3 and ev_switch == "PASS" and ev_timing in ("PASS", "UNCERTAIN"):
        overall = "CONFIRMED PHYSICAL BFSK"
    elif passes >= 2 and timing["transitions"] >= 2:
        overall = "LIKELY BFSK BUT LOW SNR"
    elif activity["duration"] > 0 and alternation["pct_carrier"] < 15.0:
        overall = "ULTRASONIC ACTIVITY PRESENT BUT NOT BFSK"
    elif meta.rms < 1e-5:
        overall = "HARDWARE PATH NOT CONFIRMED"
    else:
        overall = "INSUFFICIENT DATA"

    return {
        "18.5 kHz carrier": ev_185,
        "20.5 kHz carrier": ev_205,
        "Sequential carrier switching": ev_switch,
        "~50 ms symbol timing": ev_timing,
        "Signal-duration correlation": ev_dur,
        "overall": overall,
    }


def _print_file_header(label: str, meta: WavMeta) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"FILE            : {meta.path}")
    print(f"SAMPLE RATE     : {meta.sample_rate} Hz")
    print(f"CHANNELS        : {meta.channels}")
    print(f"SAMPLES         : {meta.samples}")
    print(f"DURATION        : {meta.duration_sec:.3f} s")
    print(f"RMS             : {meta.rms:.6e}")
    print(f"PEAK AMPLITUDE  : {meta.peak:.6e}")


def _print_bins(result: TimelineResult) -> None:
    print(f"\n{'=' * 60}")
    print("CARRIER BINS")
    print(f"{'=' * 60}")
    print(f"18.5 kHz target : {FREQ_0:.1f} Hz")
    print(f"18.5 kHz actual : {result.bin_18500_hz:.4f} Hz")
    print(f"20.5 kHz target : {FREQ_1:.1f} Hz")
    print(f"20.5 kHz actual : {result.bin_20500_hz:.4f} Hz")
    print(f"Frame step      : {FRAME_STEP_SEC * 1000:.2f} ms ({HOP} samples)")


def _print_timeline(result: TimelineResult) -> None:
    print(f"\n{'=' * 60}")
    print("CHRONOLOGICAL CARRIER TIMELINE (ALL FRAMES)")
    print(f"{'=' * 60}")
    print(f"{'TIME':<10} {'18500':<12} {'20500':<12} {'STATE':<12}")
    print("-" * 60)
    p185_max = max(float(np.max(result.p185)), 1e-30)
    p205_max = max(float(np.max(result.p205)), 1e-30)
    for t, a, b, st in zip(result.times, result.p185, result.p205, result.states):
        l185 = _power_label(float(a), p185_max)
        l205 = _power_label(float(b), p205_max)
        print(f"{t:>8.3f}s  {l185:<12} {l205:<12} {st:<12}  P185={a:.3e} P205={b:.3e}")


def _print_segments(result: TimelineResult) -> None:
    print(f"\n{'=' * 60}")
    print("COLLAPSED CARRIER SEGMENTS")
    print(f"{'=' * 60}")
    print(f"{'START':<12} {'END':<12} {'CARRIER':<14} {'DURATION':<12}")
    print("-" * 60)
    for seg in result.segments:
        print(
            f"{seg['start']:>8.3f}s   {seg['end']:>8.3f}s   "
            f"{seg['carrier'] + ' Hz':<14} {seg['duration_ms']:>8.2f} ms"
        )
    if not result.segments:
        print("(no 18500/20500 segments detected)")


def _print_timing(result: TimelineResult) -> None:
    t = result.timing
    a = result.alternation
    print(f"\n{'=' * 60}")
    print("BFSK TIMING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Expected symbol duration : {BIT_DURATION_MS:.0f} ms")
    print(f"Measured median          : {t['median_ms']:.2f} ms")
    print(f"Measured mean            : {t['mean_ms']:.2f} ms")
    print(f"Measured std             : {t['std_ms']:.2f} ms")
    print(f"Measured min             : {t['min_ms']:.2f} ms")
    print(f"Measured max             : {t['max_ms']:.2f} ms")
    print(f"18500 segments           : {t['n_18500']}")
    print(f"20500 segments           : {t['n_20500']}")
    print(f"Carrier transitions      : {t['transitions']}")
    print(f"\nAlternating transition ratio : {a['alt_ratio']:.3f}")
    print(f"Carrier frame %              : {a['pct_carrier']:.1f}%")
    print(f"Ambiguous frame %            : {a['pct_ambiguous']:.1f}%")
    print(f"None frame %                 : {a['pct_none']:.1f}%")
    print(f"Median deviation from 50 ms  : {a['deviation_ms']:.2f} ms")


def _print_activity(result: TimelineResult, pc1_duration: float) -> None:
    act = result.activity
    print(f"\n{'=' * 60}")
    print("SIGNAL TIME WINDOW")
    print(f"{'=' * 60}")
    print(f"Signal start     : {act['start']:.3f} s")
    print(f"Signal end       : {act['end']:.3f} s")
    print(f"Signal duration  : {act['duration']:.3f} s")
    print(f"PC1 WAV duration : {pc1_duration:.3f} s (expected ~7.8 s)")


def _print_comparison(pc1: TimelineResult, pc2: TimelineResult) -> None:
    print(f"\n{'=' * 60}")
    print("PC1 WAV vs PC2 CAPTURE COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'METRIC':<28} {'PC1 WAV':>16} {'PC2 CAPTURE':>16}")
    print("-" * 62)
    rows = [
        ("Sample rate (Hz)", pc1.meta.sample_rate, pc2.meta.sample_rate),
        ("Duration (s)", pc1.meta.duration_sec, pc2.meta.duration_sec),
        ("18.5 kHz frame hits", pc1.states.count("18500"), pc2.states.count("18500")),
        ("20.5 kHz frame hits", pc1.states.count("20500"), pc2.states.count("20500")),
        ("Transitions", pc1.timing["transitions"], pc2.timing["transitions"]),
        ("Median symbol (ms)", pc1.timing["median_ms"], pc2.timing["median_ms"]),
        ("18500 segments", pc1.timing["n_18500"], pc2.timing["n_18500"]),
        ("20500 segments", pc1.timing["n_20500"], pc2.timing["n_20500"]),
        ("Carrier frame %", pc1.alternation["pct_carrier"], pc2.alternation["pct_carrier"]),
        ("Alt. transition ratio", pc1.alternation["alt_ratio"], pc2.alternation["alt_ratio"]),
    ]
    for name, v1, v2 in rows:
        print(f"{name:<28} {v1:>16.4g} {v2:>16.4g}")


def _save_plots(pc2: TimelineResult, capture_base: str) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tl_path = os.path.join(FORENSIC_DIR, "physical_bfsk_timeline.png")
        pw_path = os.path.join(FORENSIC_DIR, "physical_bfsk_power.png")

        fig, ax = plt.subplots(figsize=(14, 4))
        y_map = {"18500": FREQ_0, "20500": FREQ_1, "AMBIGUOUS": (FREQ_0 + FREQ_1) / 2, "NONE": 0}
        colors = {"18500": "cyan", "20500": "lime", "AMBIGUOUS": "orange", "NONE": "gray"}
        for t, st in zip(pc2.times, pc2.states):
            if st == "NONE":
                continue
            ax.scatter(t, y_map[st], c=colors[st], s=8, alpha=0.7)
        ax.axhline(FREQ_0, color="cyan", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(FREQ_1, color="lime", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Carrier (Hz)")
        ax.set_title(f"Physical BFSK carrier timeline: {capture_base}")
        ax.set_ylim(17000, 22000)
        fig.tight_layout()
        fig.savefig(tl_path, dpi=120)
        plt.close(fig)
        saved.append(tl_path)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(pc2.times, pc2.p185, label="18.5 kHz band power", color="cyan")
        ax.plot(pc2.times, pc2.p205, label="20.5 kHz band power", color="lime")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band power")
        ax.set_title("Carrier band power vs time (PC2 capture)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pw_path, dpi=120)
        plt.close(fig)
        saved.append(pw_path)
    except Exception as exc:
        print(f"(plot generation skipped: {exc})")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical BFSK timeline forensic analyzer")
    parser.add_argument("--capture", help="PC2 capture WAV (default: newest pc2_capture_*.wav)")
    parser.add_argument("--pc1-wav", help="PC1 BFSK WAV (default: newest generated_payloads/*.wav)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-timeline-rows", type=int, default=0, help="Limit timeline rows (0=all)")
    args = parser.parse_args()

    capture = args.capture or _newest(os.path.join(FORENSIC_DIR, "pc2_capture_*.wav"))
    pc1_wav = args.pc1_wav or _newest(os.path.join(PAYLOAD_DIR, "*.wav"))

    if not capture or not os.path.isfile(capture):
        print("ERROR: No PC2 capture found in forensic_captures/pc2_capture_*.wav")
        return 1
    if not pc1_wav or not os.path.isfile(pc1_wav):
        print("ERROR: No PC1 BFSK WAV found in generated_payloads/*.wav")
        return 1

    print("=" * 60)
    print("PHYSICAL BFSK TIMELINE FORENSIC ANALYZER")
    print("=" * 60)

    pc2 = analyze_timeline(capture)
    pc1 = analyze_timeline(pc1_wav)

    _print_file_header("STEP 1 - PC2 CAPTURE", pc2.meta)
    _print_bins(pc2)

    if args.max_timeline_rows > 0:
        # Print representative subset for very long files if requested.
        subset = TimelineResult(
            meta=pc2.meta, bin_18500_hz=pc2.bin_18500_hz, bin_20500_hz=pc2.bin_20500_hz,
            times=pc2.times[:args.max_timeline_rows],
            p185=pc2.p185[:args.max_timeline_rows], p205=pc2.p205[:args.max_timeline_rows],
            states=pc2.states[:args.max_timeline_rows],
            segments=pc2.segments, timing=pc2.timing, alternation=pc2.alternation,
            activity=pc2.activity, verdict=pc2.verdict,
        )
        _print_timeline(subset)
        print(f"\n(showing first {args.max_timeline_rows} of {len(pc2.times)} frames)")
    else:
        _print_timeline(pc2)

    _print_segments(pc2)
    _print_timing(pc2)
    _print_activity(pc2, pc1.meta.duration_sec)
    _print_comparison(pc1, pc2)

    if not args.no_plots:
        saved = _save_plots(pc2, os.path.basename(capture))
        if saved:
            print(f"\nPlots saved:")
            for p in saved:
                print(f"  {p}")

    v = pc2.verdict
    print(f"\n{'=' * 60}")
    print("PHYSICAL BFSK FORENSIC DIAGNOSIS")
    print(f"{'=' * 60}")
    print(f"18.5 kHz carrier              : {v['18.5 kHz carrier']}")
    print(f"20.5 kHz carrier              : {v['20.5 kHz carrier']}")
    print(f"Sequential carrier switching  : {v['Sequential carrier switching']}")
    print(f"~50 ms symbol timing          : {v['~50 ms symbol timing']}")
    print(f"Signal-duration correlation   : {v['Signal-duration correlation']}")
    print(f"\nOverall                       : {v['overall']}")
    print(
        f"\nMeasured: transitions={pc2.timing['transitions']}, "
        f"median={pc2.timing['median_ms']:.2f} ms, "
        f"carrier_frames={pc2.alternation['pct_carrier']:.1f}%, "
        f"alt_ratio={pc2.alternation['alt_ratio']:.3f}"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
