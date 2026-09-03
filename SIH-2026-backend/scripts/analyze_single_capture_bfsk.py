#!/usr/bin/env python3
"""
Single-capture BFSK forensic analyzer (diagnostic only).

Analyzes one microphone WAV for sequential BFSK carrier behavior.
Does NOT modify production code.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.stats import pearsonr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

FORENSIC_DIR = os.path.join(_REPO_ROOT, "forensic_captures")

FREQ_0 = 18500.0
FREQ_1 = 20500.0
BIT_DURATION_MS = 50.0
NPERSEG = 2048
NOVERLAP = 1536
DOMINANCE_RATIO = 1.5
SNR_MIN_DB = 12.0
MIN_CARRIER_POWER = 1e-12
EPS = 1e-30

# Reference correlation (optional) reuses existing diagnostic helpers.
try:
    from scripts.analyze_physical_bfsk_correlation import (
        build_pc2_carrier_trace,
        derive_pc1_symbol_sequence,
        evaluate_offset,
        shuffle_control_test,
        load_mono as _load_mono_corr,
    )
    _HAS_CORR = True
except Exception:
    _HAS_CORR = False


@dataclass
class FrameRow:
    time: float
    p185: float
    p205: float
    snr185: float
    snr205: float
    ratio: float
    score: float
    state: str


@dataclass
class Segment:
    start: float
    end: float
    state: str
    duration_ms: float


@dataclass
class ChannelResult:
    channel: int
    rms: float
    peak: float
    p185_max: float
    p205_max: float
    p185_mean: float
    p205_mean: float
    score: float  # combined activity metric


@dataclass
class AnalysisResult:
    path: str
    sample_rate: int
    n_channels: int
    n_samples: int
    duration: float
    selected_channel: int
    channel_results: list[ChannelResult]
    frames: list[FrameRow] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    classification: str = "UNCERTAIN"
    confidence: str = "LOW"
    correlation: dict | None = None


def load_wav_all_channels(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True)
    return data.astype(np.float64), int(sr)


def _nearest_bin_freq(target: float, freqs: np.ndarray) -> float:
    return float(freqs[int(np.argmin(np.abs(freqs - target)))])


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)


def _band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = _band_mask(freqs, lo, hi)
    if not np.any(mask):
        return np.zeros(power.shape[1])
    return power[mask, :].sum(axis=0)


def _local_noise_floor(power: np.ndarray, freqs: np.ndarray, center: float) -> np.ndarray:
    """Per-frame noise from sidebands around carrier, excluding carrier band."""
    side_lo_a = _band_mask(freqs, center - 900, center - 500)
    side_lo_b = _band_mask(freqs, center - 400, center - 100)
    side_hi_a = _band_mask(freqs, center + 100, center + 400)
    side_hi_b = _band_mask(freqs, center + 500, center + 900)
    masks = [side_lo_a, side_lo_b, side_hi_a, side_hi_b]
    floors = []
    for m in masks:
        if np.any(m):
            floors.append(np.median(power[m, :], axis=0))
    if not floors:
        return np.full(power.shape[1], EPS)
    return np.maximum(np.median(np.stack(floors, axis=0), axis=0), EPS)


def analyze_channel(audio: np.ndarray, sr: int) -> tuple[list[FrameRow], dict]:
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

    bin185 = _nearest_bin_freq(FREQ_0, freqs)
    bin205 = _nearest_bin_freq(FREQ_1, freqs)

    p185 = _band_power(power, freqs, bin185 - 46.875, bin185 + 46.875)
    p205 = _band_power(power, freqs, bin205 - 46.875, bin205 + 46.875)

    nf185 = _local_noise_floor(power, freqs, bin185)
    nf205 = _local_noise_floor(power, freqs, bin205)

    snr185 = 10.0 * np.log10((p185 + EPS) / nf185)
    snr205 = 10.0 * np.log10((p205 + EPS) / nf205)

    frames: list[FrameRow] = []
    for i, t in enumerate(times):
        a, b = float(p185[i]), float(p205[i])
        s185, s205 = float(snr185[i]), float(snr205[i])
        ratio = a / (b + EPS)
        score = (a - b) / (a + b + EPS)

        peak_pwr = max(a, b)
        above185 = s185 >= SNR_MIN_DB and a >= MIN_CARRIER_POWER
        above205 = s205 >= SNR_MIN_DB and b >= MIN_CARRIER_POWER

        if peak_pwr < MIN_CARRIER_POWER or (not above185 and not above205):
            state = "NONE"
        elif a >= DOMINANCE_RATIO * b and above185:
            state = "18500 HIGH"
        elif b >= DOMINANCE_RATIO * a and above205:
            state = "20500 HIGH"
        else:
            state = "AMBIGUOUS"

        frames.append(FrameRow(t, a, b, s185, s205, ratio, score, state))

    meta = {
        "bin185": bin185,
        "bin205": bin205,
        "p185_max": float(np.max(p185)),
        "p205_max": float(np.max(p205)),
        "p185_mean": float(np.mean(p185)),
        "p205_mean": float(np.mean(p205)),
        "snr185_max": float(np.max(snr185)),
        "snr205_max": float(np.max(snr205)),
    }
    return frames, meta


def _state_base(state: str) -> str:
    if state.startswith("18500"):
        return "18500"
    if state.startswith("20500"):
        return "20500"
    if state == "AMBIGUOUS":
        return "AMBIGUOUS"
    return "NONE"


def collapse_segments(frames: list[FrameRow], sr: int) -> list[Segment]:
    if not frames:
        return []
    frame_dur = NPERSEG / sr
    segments: list[Segment] = []
    cur = _state_base(frames[0].state)
    start = frames[0].time
    for i in range(1, len(frames)):
        st = _state_base(frames[i].state)
        if st != cur:
            end = frames[i - 1].time + frame_dur
            segments.append(Segment(start, end, cur, (end - start) * 1000))
            cur = st
            start = frames[i].time
    end = frames[-1].time + frame_dur
    segments.append(Segment(start, end, cur, (end - start) * 1000))
    return segments


def compute_stats(frames: list[FrameRow], segments: list[Segment], sr: int) -> dict:
    bases = [_state_base(f.state) for f in frames]
    n = len(bases)
    n185 = bases.count("18500")
    n205 = bases.count("20500")
    n_amb = bases.count("AMBIGUOUS")
    n_none = bases.count("NONE")

    carrier_segs = [s for s in segments if s.state in ("18500", "20500") and s.duration_ms >= 42.67]
    short_filtered = [s for s in segments if s.state in ("18500", "20500")]

    durs = [s.duration_ms for s in carrier_segs]
    transitions = 0
    alt_trans = 0
    prev = None
    for s in short_filtered:
        if s.state in ("18500", "20500"):
            if prev and s.state != prev:
                transitions += 1
                alt_trans += 1
            elif prev:
                pass
            prev = s.state

    alt_ratio = alt_trans / transitions if transitions > 0 else 0.0

    carrier_snrs185 = [f.snr185 for f in frames if _state_base(f.state) == "18500"]
    carrier_snrs205 = [f.snr205 for f in frames if _state_base(f.state) == "20500"]

    return {
        "n185": n185,
        "n205": n205,
        "n_amb": n_amb,
        "n_none": n_none,
        "pct185": 100.0 * n185 / max(n, 1),
        "pct205": 100.0 * n205 / max(n, 1),
        "pct_amb": 100.0 * n_amb / max(n, 1),
        "pct_none": 100.0 * n_none / max(n, 1),
        "carrier_active_pct": 100.0 * (n185 + n205) / max(n, 1),
        "transitions": transitions,
        "n185_segs": sum(1 for s in carrier_segs if s.state == "18500"),
        "n205_segs": sum(1 for s in carrier_segs if s.state == "20500"),
        "median_ms": float(np.median(durs)) if durs else 0.0,
        "mean_ms": float(np.mean(durs)) if durs else 0.0,
        "std_ms": float(np.std(durs)) if durs else 0.0,
        "min_ms": float(np.min(durs)) if durs else 0.0,
        "max_ms": float(np.max(durs)) if durs else 0.0,
        "alt_ratio": alt_ratio,
        "median_snr185": float(np.median(carrier_snrs185)) if carrier_snrs185 else 0.0,
        "median_snr205": float(np.median(carrier_snrs205)) if carrier_snrs205 else 0.0,
    }


def classify_signal(stats: dict, meta: dict) -> tuple[str, str]:
    pct185 = stats["pct185"]
    pct205 = stats["pct205"]
    active = stats["carrier_active_pct"]
    alt = stats["alt_ratio"]
    med = stats["median_ms"]

    if active < 5.0 and meta["p185_max"] < 1e-20 and meta["p205_max"] < 1e-20:
        return "NO_ULTRASONIC_ACTIVITY", "HIGH"

    if pct185 > 70 and pct205 < 15 and alt < 0.2:
        return "SINGLE_TONE_18500", "MEDIUM" if active > 20 else "LOW"

    if pct205 > 70 and pct185 < 15 and alt < 0.2:
        return "SINGLE_TONE_20500", "MEDIUM" if active > 20 else "LOW"

    strong_signal = max(meta["p185_max"], meta["p205_max"]) >= 1e-6
    both_carriers_snr = min(meta["snr185_max"], meta["snr205_max"]) >= 15.0

    bfsk_ok = (
        pct185 >= 10
        and pct205 >= 10
        and stats["transitions"] >= 5
        and alt >= 0.4
        and 25 <= med <= 130
        and active >= 15
        and stats["pct_amb"] < 25
        and strong_signal
        and both_carriers_snr
        and stats["median_snr185"] >= 10
        and stats["median_snr205"] >= 10
    )
    if bfsk_ok:
        conf = "HIGH" if alt >= 0.55 and stats["transitions"] >= 10 else "MEDIUM"
        return "LIKELY_BFSK", conf

    if active >= 5 or meta["p185_max"] > 1e-15 or meta["p205_max"] > 1e-15:
        return "ULTRASONIC_ACTIVITY_NO_BFSK_PATTERN", "MEDIUM"

    return "UNCERTAIN", "LOW"


def _lvl(p: float, pmax: float) -> str:
    return "HIGH" if pmax > 0 and p >= 0.25 * pmax else "LOW"


def write_timelines(result: AnalysisResult, base_name: str = "single_capture_bfsk") -> tuple[str, str]:
    full_path = os.path.join(FORENSIC_DIR, f"{base_name}_timeline_full.txt")
    compact_path = os.path.join(FORENSIC_DIR, f"{base_name}_timeline.txt")

    p185_max = max((f.p185 for f in result.frames), default=1.0)
    p205_max = max((f.p205 for f in result.frames), default=1.0)

    with open(full_path, "w", encoding="utf-8") as ff:
        ff.write(f"FILE: {result.path}\n")
        ff.write(f"CHANNEL: {result.selected_channel}\n")
        ff.write(f"{'TIME':>8} {'P185':>12} {'P205':>12} {'S185':>8} {'S205':>8} {'RATIO':>8} {'SCORE':>8} {'STATE':>12}\n")
        ff.write("-" * 80 + "\n")
        for f in result.frames:
            ff.write(
                f"{f.time:8.3f} {f.p185:12.6e} {f.p205:12.6e} "
                f"{f.snr185:8.2f} {f.snr205:8.2f} {f.ratio:8.3f} {f.score:8.3f} {f.state:>12}\n"
            )

    with open(compact_path, "w", encoding="utf-8") as fc:
        fc.write("=" * 60 + "\n")
        fc.write("PHYSICAL BFSK CARRIER TIMELINE\n")
        fc.write("=" * 60 + "\n\n")
        fc.write(f"FILE: {result.path}\n")
        fc.write(f"CHANNEL: {result.selected_channel}\n\n")
        fc.write(f"{'TIME':<10} {'18500':<12} {'20500':<12} {'STATE':<12}\n")
        fc.write("-" * 60 + "\n")
        for f in result.frames:
            fc.write(
                f"{f.time:6.3f}s   {_lvl(f.p185, p185_max):<12} {_lvl(f.p205, p205_max):<12} "
                f"{_state_base(f.state):<12}\n"
            )
        fc.write("\n" + "=" * 60 + "\n")
        fc.write("COLLAPSED CARRIER SEGMENTS\n")
        fc.write("=" * 60 + "\n\n")
        fc.write(f"{'START':<10} {'END':<10} {'STATE':<10} {'DURATION':<10}\n")
        fc.write("-" * 60 + "\n")
        for s in result.segments:
            fc.write(f"{s.start:6.3f}s   {s.end:6.3f}s   {s.state:<10} {s.duration_ms:6.0f} ms\n")

    return full_path, compact_path


def save_plot(result: AnalysisResult, out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [f.time for f in result.frames]
        p185 = [f.p185 for f in result.frames]
        p205 = [f.p205 for f in result.frames]
        state_map = {"18500": 1, "20500": 2, "AMBIGUOUS": 0.5, "NONE": 0}
        states = [state_map.get(_state_base(f.state), 0) for f in result.frames]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax1.plot(times, p185, label="P18500", color="cyan")
        ax1.plot(times, p205, label="P20500", color="lime")
        ax1.set_ylabel("Band power")
        ax1.legend()
        ax1.set_title(os.path.basename(result.path))

        ax2.plot(times, states, drawstyle="steps-post", color="orange")
        ax2.set_yticks([0, 0.5, 1, 2])
        ax2.set_yticklabels(["NONE", "AMB", "18500", "20500"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Carrier state")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception:
        return False


def run_correlation(capture_path: str, reference_path: str) -> dict:
    if not _HAS_CORR:
        return {"error": "correlation module unavailable"}
    symbols = derive_pc1_symbol_sequence(*_load_mono_corr(reference_path))
    audio, sr = _load_mono_corr(capture_path)
    trace = build_pc2_carrier_trace(audio, sr)

    best = None
    for off in np.arange(-1.0, 12.01, 0.02):
        r = evaluate_offset(symbols, trace, float(off))
        if best is None or (r["agreement"], r["pearson"], r["usable"]) > (
            best["agreement"], best["pearson"], best["usable"]
        ):
            best = r
    control = shuffle_control_test(symbols, trace, best["offset"])
    significant = (
        best["pearson"] > control["control_max"] + 0.05
        and best["agreement"] >= 0.55
        and best["usable"] >= 20
    )
    return {**best, **control, "significant": significant}


def analyze_file(path: str, reference: str | None = None) -> AnalysisResult:
    data, sr = load_wav_all_channels(path)
    n_ch = data.shape[1]
    n_samples = data.shape[0]
    duration = n_samples / sr

    ch_results: list[ChannelResult] = []
    best_ch = 0
    best_score = -1.0
    best_frames: list[FrameRow] = []
    best_meta: dict = {}

    print("=" * 60)
    print("SINGLE CAPTURE BFSK ANALYZER")
    print("=" * 60)
    print(f"FILE         : {os.path.abspath(path)}")
    print(f"SAMPLE RATE  : {sr} Hz")
    print(f"CHANNELS     : {n_ch}")
    print(f"SAMPLES      : {n_samples}")
    print(f"DURATION     : {duration:.3f} s")

    for ch in range(n_ch):
        audio = data[:, ch]
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        frames, meta = analyze_channel(audio, sr)
        score = meta["p185_max"] + meta["p205_max"]
        cr = ChannelResult(ch, rms, peak, meta["p185_max"], meta["p205_max"],
                           meta["p185_mean"], meta["p205_mean"], score)
        ch_results.append(cr)
        print(f"\nChannel {ch}: RMS={rms:.6e} PEAK={peak:.6e}")
        print(f"  18.5 kHz bin={meta['bin185']:.4f} MAX={meta['p185_max']:.6e} MEAN={meta['p185_mean']:.6e}")
        print(f"  20.5 kHz bin={meta['bin205']:.4f} MAX={meta['p205_max']:.6e} MEAN={meta['p205_mean']:.6e}")
        print(f"  SNR185 max={meta['snr185_max']:.2f} dB  SNR205 max={meta['snr205_max']:.2f} dB")
        if score > best_score:
            best_score = score
            best_ch = ch
            best_frames = frames
            best_meta = meta

    print(f"\nSELECTED CHANNEL: {best_ch} (strongest 18.5/20.5 kHz activity)")

    segments = collapse_segments(best_frames, sr)
    stats = compute_stats(best_frames, segments, sr)
    classification, confidence = classify_signal(stats, best_meta)

    result = AnalysisResult(
        path=path,
        sample_rate=sr,
        n_channels=n_ch,
        n_samples=n_samples,
        duration=duration,
        selected_channel=best_ch,
        channel_results=ch_results,
        frames=best_frames,
        segments=segments,
        stats=stats,
        classification=classification,
        confidence=confidence,
    )

    base = os.path.splitext(os.path.basename(path))[0]
    full_p, compact_p = write_timelines(result, "single_capture_bfsk")
    per_base = f"single_capture_{base}"
    per_full = os.path.join(FORENSIC_DIR, f"{per_base}_timeline_full.txt")
    per_compact = os.path.join(FORENSIC_DIR, f"{per_base}_timeline.txt")
    shutil.copy2(full_p, per_full)
    shutil.copy2(compact_p, per_compact)
    print(f"\nFull timeline  : {full_p}")
    print(f"Compact timeline: {compact_p}")
    print(f"Per-file copies : {per_full}")

    plot_path = os.path.join(FORENSIC_DIR, "single_capture_bfsk_timeline.png")
    per_plot = os.path.join(FORENSIC_DIR, f"{per_base}_timeline.png")
    if save_plot(result, plot_path):
        shutil.copy2(plot_path, per_plot)
        print(f"Plot saved     : {plot_path}")
    else:
        print("Plot skipped: matplotlib unavailable")

    if reference:
        print(f"\nREFERENCE CORRELATION: {reference}")
        corr = run_correlation(path, reference)
        result.correlation = corr
        if "error" in corr:
            print(f"  {corr['error']}")
        else:
            sig = "SIGNIFICANT" if corr.get("significant") else "NOT SIGNIFICANT"
            print(f"  Best offset     : {corr['offset']:.3f} s")
            print(f"  Pearson         : {corr['pearson']:.4f}")
            print(f"  Agreement       : {corr['agreement']:.4f} ({corr['matches']}/{corr['usable']})")
            print(f"  Control max     : {corr['control_max']:.4f}")
            print(f"  CORRELATION     : {sig}")

    print("\n" + "=" * 60)
    print("SINGLE CAPTURE BFSK DIAGNOSIS")
    print("=" * 60)
    print(f"File: {os.path.basename(path)}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.3f} s")
    print(f"Selected channel: {best_ch}")
    sel = ch_results[best_ch]
    print(f"RMS: {sel.rms:.6e}")
    print(f"PEAK: {sel.peak:.6e}")
    print(f"\n18500 Hz: max={best_meta['p185_max']:.6e} mean={best_meta['p185_mean']:.6e}")
    print(f"20500 Hz: max={best_meta['p205_max']:.6e} mean={best_meta['p205_mean']:.6e}")
    print(f"\n18500 frame %: {stats['pct185']:.1f}%")
    print(f"20500 frame %: {stats['pct205']:.1f}%")
    print(f"Ambiguous %:   {stats['pct_amb']:.1f}%")
    print(f"None %:        {stats['pct_none']:.1f}%")
    print(f"\nTransitions:     {stats['transitions']}")
    print(f"Alternation ratio: {stats['alt_ratio']:.3f}")
    print(f"\nMedian segment: {stats['median_ms']:.1f} ms")
    print(f"Mean segment:   {stats['mean_ms']:.1f} ms")
    print(f"Std segment:    {stats['std_ms']:.1f} ms")
    print(f"\nSignal classification: {classification}")
    print(f"Confidence: {confidence}")
    print("\nNOTE: Ultrasonic activity detected is NOT equivalent to BFSK detected.")
    print("=" * 60)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-capture BFSK forensic analyzer")
    parser.add_argument("--wav", required=True, help="Captured WAV path")
    parser.add_argument("--reference", help="Optional PC1 BFSK reference WAV")
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        print(f"ERROR: file not found: {args.wav}")
        return 1
    if args.reference and not os.path.isfile(args.reference):
        print(f"ERROR: reference not found: {args.reference}")
        return 1

    analyze_file(args.wav, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
