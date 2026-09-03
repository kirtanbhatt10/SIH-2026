#!/usr/bin/env python3
"""
PC2 microphone BFSK forensic timeline (diagnostic only).

Analyzes a PC2 microphone capture WAV for time-resolved 18500/20500 Hz
carrier dominance. Does NOT modify production code.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import stft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import FORENSIC_CAPTURE_DIR

FORENSIC_DIR = os.path.join(_REPO_ROOT, FORENSIC_CAPTURE_DIR)

FREQ_0 = 18500.0
FREQ_1 = 20500.0
EXPECTED_PC1_PLAYBACK_S = 7.8
NPERSEG = 2048
NOVERLAP = 1536
BAND_HALF_WIDTH = 100.0
EPS = 1e-30

DEFAULT_RATIO_THRESHOLD = 1.5
DEFAULT_MIN_POWER_DB = 14.0
MIN_ABS_CARRIER_POWER = 1e-12


@dataclass
class FrameRow:
    time: float
    p185: float
    p205: float
    ratio: float
    score: float
    state: str
    lvl185: str
    lvl205: str


@dataclass
class Segment:
    start: float
    end: float
    carrier: str
    duration_ms: float


def find_newest_pc2_capture() -> str | None:
    pattern = os.path.join(FORENSIC_DIR, "pc2_capture_*.wav")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_audio(path: str) -> tuple[np.ndarray, int, int]:
    """Return mono audio, sample rate, and source channel index."""
    data, sr = sf.read(path, always_2d=True)
    data = data.astype(np.float64)
    n_ch = data.shape[1]
    if n_ch == 1:
        return data[:, 0], int(sr), 0

    best_ch = 0
    best_score = -1.0
    for ch in range(n_ch):
        _, _, zxx = stft(
            data[:, ch],
            fs=sr,
            window="hann",
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            boundary=None,
            padded=False,
        )
        freqs = np.fft.rfftfreq(NPERSEG, d=1.0 / sr)
        power = np.abs(zxx) ** 2
        b0 = _nearest_bin(FREQ_0, freqs)
        b1 = _nearest_bin(FREQ_1, freqs)
        p0 = _band_power(power, freqs, b0 - BAND_HALF_WIDTH, b0 + BAND_HALF_WIDTH).max()
        p1 = _band_power(power, freqs, b1 - BAND_HALF_WIDTH, b1 + BAND_HALF_WIDTH).max()
        score = float(p0 + p1)
        if score > best_score:
            best_score = score
            best_ch = ch
    return data[:, best_ch], int(sr), best_ch


def _nearest_bin(target: float, freqs: np.ndarray) -> float:
    return float(freqs[int(np.argmin(np.abs(freqs - target)))])


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)


def _band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = _band_mask(freqs, lo, hi)
    if not np.any(mask):
        return np.zeros(power.shape[1])
    return power[mask, :].sum(axis=0)


def _local_noise_floor(power: np.ndarray, freqs: np.ndarray, center: float) -> np.ndarray:
    side_masks = [
        _band_mask(freqs, center - 900, center - 500),
        _band_mask(freqs, center - 400, center - 100),
        _band_mask(freqs, center + 100, center + 400),
        _band_mask(freqs, center + 500, center + 900),
    ]
    floors = []
    for m in side_masks:
        if np.any(m):
            floors.append(np.median(power[m, :], axis=0))
    if not floors:
        return np.full(power.shape[1], EPS)
    return np.maximum(np.median(np.stack(floors, axis=0), axis=0), EPS)


def analyze_timeline(
    audio: np.ndarray,
    sr: int,
    ratio_threshold: float,
    min_power_db: float,
) -> tuple[list[FrameRow], dict]:
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

    bin185 = _nearest_bin(FREQ_0, freqs)
    bin205 = _nearest_bin(FREQ_1, freqs)

    p185 = _band_power(power, freqs, bin185 - BAND_HALF_WIDTH, bin185 + BAND_HALF_WIDTH)
    p205 = _band_power(power, freqs, bin205 - BAND_HALF_WIDTH, bin205 + BAND_HALF_WIDTH)

    nf185 = _local_noise_floor(power, freqs, bin185)
    nf205 = _local_noise_floor(power, freqs, bin205)

    snr185 = 10.0 * np.log10((p185 + EPS) / nf185)
    snr205 = 10.0 * np.log10((p205 + EPS) / nf205)

    p185_max = float(np.max(p185))
    p205_max = float(np.max(p205))

    frames: list[FrameRow] = []
    for i, t in enumerate(times):
        a, b = float(p185[i]), float(p205[i])
        s185, s205 = float(snr185[i]), float(snr205[i])
        ratio = a / (b + EPS)
        score = a - b

        above185 = s185 >= min_power_db and a >= MIN_ABS_CARRIER_POWER
        above205 = s205 >= min_power_db and b >= MIN_ABS_CARRIER_POWER
        peak = max(a, b)

        if peak < EPS or (not above185 and not above205):
            state = "NONE"
            lvl185, lvl205 = "LOW", "LOW"
        elif a >= ratio_threshold * b and above185:
            state = "18500"
            lvl185, lvl205 = "HIGH", "LOW"
        elif b >= ratio_threshold * a and above205:
            state = "20500"
            lvl185, lvl205 = "LOW", "HIGH"
        else:
            state = "AMBIGUOUS"
            lvl185 = "HIGH" if p185_max > 0 and a >= 0.25 * p185_max else "LOW"
            lvl205 = "HIGH" if p205_max > 0 and b >= 0.25 * p205_max else "LOW"

        frames.append(FrameRow(t, a, b, ratio, score, state, lvl185, lvl205))

    meta = {
        "bin185": bin185,
        "bin205": bin205,
        "p185_max": p185_max,
        "p205_max": p205_max,
        "p185_mean": float(np.mean(p185)),
        "p205_mean": float(np.mean(p205)),
        "frame_step_s": (NPERSEG - NOVERLAP) / sr,
        "frame_dur_s": NPERSEG / sr,
    }
    return frames, meta


def collapse_carrier_segments(frames: list[FrameRow], sr: int) -> list[Segment]:
    frame_dur = NPERSEG / sr
    segments: list[Segment] = []
    cur: str | None = None
    start = 0.0
    for f in frames:
        if f.state not in ("18500", "20500"):
            if cur is not None:
                end = f.time
                segments.append(Segment(start, end, cur, (end - start) * 1000))
                cur = None
            continue
        if cur != f.state:
            if cur is not None:
                end = f.time
                segments.append(Segment(start, end, cur, (end - start) * 1000))
            cur = f.state
            start = f.time
    if cur is not None:
        end = frames[-1].time + frame_dur
        segments.append(Segment(start, end, cur, (end - start) * 1000))
    return segments


def compute_summary(frames: list[FrameRow], segments: list[Segment]) -> dict:
    n = len(frames)
    states = [f.state for f in frames]
    n185 = states.count("18500")
    n205 = states.count("20500")
    n_amb = states.count("AMBIGUOUS")
    n_none = states.count("NONE")
    active = n185 + n205

    transitions = 0
    alt_trans = 0
    prev: str | None = None
    for s in segments:
        if prev and s.carrier != prev:
            transitions += 1
            alt_trans += 1
        prev = s.carrier

    alt_ratio = alt_trans / transitions if transitions > 0 else 0.0
    durs = [s.duration_ms for s in segments]

    active_times = [f.time for f in frames if f.state in ("18500", "20500", "AMBIGUOUS")]
    carrier_times = [f.time for f in frames if f.state in ("18500", "20500")]
    first_active = min(active_times) if active_times else None
    last_active = max(carrier_times) if carrier_times else None
    active_duration = (last_active - first_active) if (first_active is not None and last_active is not None) else 0.0

    longest_seg = max((s.duration_ms for s in segments), default=0.0)
    longest_region = 0.0
    if segments:
        run_start = segments[0].start
        run_end = segments[0].end
        for s in segments[1:]:
            if s.start <= run_end + 0.02:
                run_end = max(run_end, s.end)
            else:
                longest_region = max(longest_region, run_end - run_start)
                run_start, run_end = s.start, s.end
        longest_region = max(longest_region, run_end - run_start)

    return {
        "total_frames": n,
        "n185": n185,
        "n205": n205,
        "n_amb": n_amb,
        "n_none": n_none,
        "active_pct": 100.0 * active / max(n, 1),
        "transitions": transitions,
        "alt_ratio": alt_ratio,
        "median_ms": float(np.median(durs)) if durs else 0.0,
        "mean_ms": float(np.mean(durs)) if durs else 0.0,
        "std_ms": float(np.std(durs)) if durs else 0.0,
        "longest_seg_ms": longest_seg,
        "longest_region_s": longest_region,
        "first_active": first_active,
        "last_active": last_active,
        "active_duration": active_duration,
        "n_segments": len(segments),
    }


def classify_bfsk_evidence(summary: dict, meta: dict) -> str:
    active_pct = summary["active_pct"]
    pct185 = 100.0 * summary["n185"] / max(summary["total_frames"], 1)
    pct205 = 100.0 * summary["n205"] / max(summary["total_frames"], 1)
    amb_pct = 100.0 * summary["n_amb"] / max(summary["total_frames"], 1)

    if meta["p185_max"] < 1e-14 and meta["p205_max"] < 1e-14:
        return "NO_ULTRASONIC_ACTIVITY"

    strong = max(meta["p185_max"], meta["p205_max"]) >= 1e-6

    confirmed = (
        strong
        and pct185 >= 15
        and pct205 >= 15
        and amb_pct < 15
        and summary["transitions"] >= 10
        and summary["alt_ratio"] >= 0.55
        and 25 <= summary["median_ms"] <= 130
        and active_pct >= 40
        and summary["n_segments"] >= 10
    )
    if confirmed:
        return "CONFIRMED_BFSK"

    likely = (
        strong
        and pct185 >= 10
        and pct205 >= 10
        and amb_pct < 25
        and summary["transitions"] >= 5
        and summary["alt_ratio"] >= 0.4
        and 25 <= summary["median_ms"] <= 130
        and active_pct >= 15
    )
    if likely:
        return "LIKELY_BFSK"

    if summary["active_pct"] < 3.0:
        return "NO_ULTRASONIC_ACTIVITY"

    if meta["p185_max"] > 1e-15 or meta["p205_max"] > 1e-15 or summary["active_pct"] >= 3:
        return "ULTRASONIC_ACTIVITY_NO_BFSK_PATTERN"

    return "NO_ULTRASONIC_ACTIVITY"


def _fmt_power(p: float) -> str:
    if p == 0:
        return "0.00e+00"
    return f"{p:.2e}"


def _decisive_frames(frames: list[FrameRow], limit: int = 50) -> list[FrameRow]:
    decisive = [f for f in frames if f.state in ("18500", "20500")]
    if len(decisive) >= 30:
        return decisive[:limit]
    out: list[FrameRow] = []
    for f in frames:
        if f.state != "NONE" or (f.p185 > 0 or f.p205 > 0):
            out.append(f)
        if len(out) >= limit:
            break
    return out


def save_csv(path: str, frames: list[FrameRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "p18500", "p20500", "ratio", "score", "state"])
        for f in frames:
            w.writerow([f"{f.time:.6f}", f.p185, f.p205, f.ratio, f.score, f.state])


def save_plot(path: str, frames: list[FrameRow], wav_name: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [f.time for f in frames]
        p185 = [f.p185 for f in frames]
        p205 = [f.p205 for f in frames]
        state_map = {"NONE": 0, "AMBIGUOUS": 0.5, "18500": 1, "20500": 2}
        states = [state_map.get(f.state, 0) for f in frames]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax1.plot(times, p185, label="18500 Hz", color="cyan")
        ax1.plot(times, p205, label="20500 Hz", color="lime")
        ax1.set_ylabel("Band power")
        ax1.legend(loc="upper right")
        ax1.set_title(f"PC2 mic BFSK timeline: {wav_name}")

        ax2.plot(times, states, drawstyle="steps-post", color="orange")
        ax2.set_yticks([0, 0.5, 1, 2])
        ax2.set_yticklabels(["NONE", "AMB", "18500", "20500"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Carrier state")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return True
    except Exception:
        return False


def build_report(
    path: str,
    channel: int,
    duration: float,
    frames: list[FrameRow],
    segments: list[Segment],
    summary: dict,
    meta: dict,
    evidence: str,
) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PC2 MICROPHONE BFSK TIMELINE ANALYSIS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"FILE: {os.path.abspath(path)}")
    lines.append(f"SAMPLE RATE: {meta['sample_rate']} Hz")
    lines.append(f"CHANNEL: {channel}")
    lines.append(f"DURATION: {duration:.3f} s")
    lines.append(f"TOTAL FRAMES: {summary['total_frames']}")
    lines.append(f"FRAME STEP: {meta['frame_step_s'] * 1000:.3f} ms")
    lines.append("")
    lines.append("TARGET CARRIERS:")
    lines.append(f"  18500 Hz -> {meta['bin185']:.4f} Hz")
    lines.append(f"  20500 Hz -> {meta['bin205']:.4f} Hz")
    lines.append("")
    lines.append("=" * 60)
    lines.append("FRAME TIMELINE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'TIME':<8} {'18500 Hz':<14} {'20500 Hz':<14} {'STATE':<12}")
    lines.append("-" * 60)
    for f in frames:
        lines.append(
            f"{f.time:6.3f}s {_fmt_power(f.p185):<14} {_fmt_power(f.p205):<14} {f.state:<12}"
        )
    lines.append("")
    lines.append("=" * 60)
    lines.append("COLLAPSED CARRIER SEGMENTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'START':<10} {'END':<10} {'CARRIER':<10} {'DURATION':<10}")
    lines.append("-" * 50)
    if segments:
        for s in segments:
            lines.append(
                f"{s.start:6.3f}s   {s.end:6.3f}s   {s.carrier:<10} {s.duration_ms:6.0f} ms"
            )
    else:
        lines.append("(no carrier segments)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"18500 frames: {summary['n185']}")
    lines.append(f"20500 frames: {summary['n205']}")
    lines.append(f"AMBIGUOUS:    {summary['n_amb']}")
    lines.append(f"NONE:         {summary['n_none']}")
    lines.append("")
    lines.append(f"ACTIVE FRAME %: {summary['active_pct']:.1f}%")
    lines.append(f"CARRIER TRANSITIONS: {summary['transitions']}")
    lines.append(f"ALTERNATING TRANSITION RATIO: {summary['alt_ratio']:.3f}")
    lines.append("")
    lines.append(f"MEDIAN CARRIER SEGMENT: {summary['median_ms']:.1f} ms")
    lines.append(f"MEAN CARRIER SEGMENT:   {summary['mean_ms']:.1f} ms")
    lines.append(f"STD CARRIER SEGMENT:    {summary['std_ms']:.1f} ms")
    lines.append(f"LONGEST SEGMENT:        {summary['longest_seg_ms']:.1f} ms")
    lines.append(f"LONGEST ACTIVE REGION:  {summary['longest_region_s']:.3f} s")
    lines.append("")
    fa = summary["first_active"]
    la = summary["last_active"]
    lines.append(f"FIRST ACTIVE: {fa:.3f}s" if fa is not None else "FIRST ACTIVE: (none)")
    lines.append(f"LAST ACTIVE:  {la:.3f}s" if la is not None else "LAST ACTIVE:  (none)")
    lines.append(f"ACTIVE DURATION: {summary['active_duration']:.3f} s")
    lines.append("")
    lines.append(f"EXPECTED PC1 PLAYBACK: ~{EXPECTED_PC1_PLAYBACK_S:.1f} seconds")
    lines.append("")
    lines.append("=" * 60)
    lines.append("BFSK EVIDENCE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(evidence)
    lines.append("")
    lines.append("NOTE: Ultrasonic activity is NOT equivalent to sequential BFSK.")
    lines.append("=" * 60)
    return "\n".join(lines)


def print_compact(frames: list[FrameRow], segments: list[Segment]) -> None:
    print(f"{'TIME':<8} {'18500':<10} {'20500':<10} {'STATE':<12}")
    print("-" * 44)
    for f in frames:
        print(f"{f.time:6.3f}s {f.lvl185:<10} {f.lvl205:<10} {f.state:<12}")
    print()
    print("COLLAPSED CARRIER SEGMENTS")
    print(f"{'START':<10} {'END':<10} {'CARRIER':<10} {'DURATION':<10}")
    print("-" * 44)
    if segments:
        for s in segments:
            print(f"{s.start:6.3f}s   {s.end:6.3f}s   {s.carrier:<10} {s.duration_ms:6.0f} ms")
    else:
        print("(no carrier segments)")


def run_analysis(
    path: str,
    output_txt: str,
    output_csv: str,
    output_png: str,
    ratio_threshold: float,
    min_power_db: float,
    compact: bool,
) -> dict:
    audio, sr, channel = load_audio(path)
    duration = len(audio) / sr
    frames, meta = analyze_timeline(audio, sr, ratio_threshold, min_power_db)
    meta["sample_rate"] = sr
    segments = collapse_carrier_segments(frames, sr)
    summary = compute_summary(frames, segments)
    evidence = classify_bfsk_evidence(summary, meta)

    report = build_report(path, channel, duration, frames, segments, summary, meta, evidence)
    os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as fh:
        fh.write(report)
    save_csv(output_csv, frames)
    if save_plot(output_png, frames, os.path.basename(path)):
        plot_msg = output_png
    else:
        plot_msg = "skipped (matplotlib unavailable)"

    if compact:
        print_compact(frames, segments)
        print(f"\nBFSK EVIDENCE: {evidence}")
        print(f"Saved: {output_txt}")
        return {
            "path": path,
            "channel": channel,
            "duration": duration,
            "frames": frames,
            "segments": segments,
            "summary": summary,
            "meta": meta,
            "evidence": evidence,
            "plot": plot_msg,
        }

    print(report)
    print(f"\nSaved: {output_txt}")
    print(f"Saved: {output_csv}")
    print(f"Plot:  {plot_msg}")

    return {
        "path": path,
        "channel": channel,
        "duration": duration,
        "frames": frames,
        "segments": segments,
        "summary": summary,
        "meta": meta,
        "evidence": evidence,
        "plot": plot_msg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PC2 microphone BFSK forensic timeline (diagnostic only)"
    )
    parser.add_argument("--wav", help="PC2 capture WAV (default: newest forensic_captures/pc2_capture_*.wav)")
    parser.add_argument(
        "--output",
        default=os.path.join(FORENSIC_DIR, "mic_bfsk_timeline.txt"),
        help="Text report output path",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help=f"Dominance ratio threshold (default {DEFAULT_RATIO_THRESHOLD})",
    )
    parser.add_argument(
        "--min-power-db",
        type=float,
        default=DEFAULT_MIN_POWER_DB,
        help=f"Min carrier SNR above local noise in dB (default {DEFAULT_MIN_POWER_DB})",
    )
    parser.add_argument("--compact", action="store_true", help="Compact demo output")
    args = parser.parse_args()

    wav_path = args.wav or find_newest_pc2_capture()
    if not wav_path or not os.path.isfile(wav_path):
        print("ERROR: No PC2 capture WAV found.")
        print(f"  Looked in: {os.path.join(FORENSIC_DIR, 'pc2_capture_*.wav')}")
        print("  Use --wav path/to/capture.wav")
        return 1

    out_dir = os.path.dirname(args.output) or FORENSIC_DIR
    csv_path = os.path.join(out_dir, "mic_bfsk_timeline.csv")
    png_path = os.path.join(out_dir, "mic_bfsk_timeline.png")

    run_analysis(
        wav_path,
        args.output,
        csv_path,
        png_path,
        args.ratio_threshold,
        args.min_power_db,
        args.compact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
