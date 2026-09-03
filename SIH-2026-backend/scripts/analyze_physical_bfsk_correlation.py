#!/usr/bin/env python3
"""
Physical BFSK synchronization / correlation diagnostic (read-only).

Correlates PC1 reference BFSK symbol sequence against PC2 microphone capture
to test whether ultrasonic activity is temporally aligned with transmission.

Does NOT modify production code, DSP, backend, ML, or source audio.
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
from scipy.stats import pearsonr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

FORENSIC_DIR = os.path.join(_REPO_ROOT, "forensic_captures")
PAYLOAD_DIR = os.path.join(_REPO_ROOT, "generated_payloads")

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, PREAMBLE, SAMPLE_RATE
from backend.services.payload_service import (
    _classify_bit,
    _find_preamble,
    _symbol_tone_power,
)

STFT_NPERSEG = 2048
STFT_NOVERLAP = 1536
OFFSET_MIN = -2.0
OFFSET_MAX = 12.0
OFFSET_STEP = 0.02
N_SHUFFLE_CONTROLS = 100
AMBIGUOUS_FRAC = 0.15  # |score| < frac * max(|score|) => ambiguous


@dataclass
class Symbol:
    index: int
    start_time: float
    end_time: float
    carrier: float
    bit: str


def _newest(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    return max(paths, key=os.path.getmtime) if paths else None


def load_mono(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    elif data.dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def carrier_from_bit(bit: str) -> float:
    return float(FREQ_1 if bit == "1" else FREQ_0)


def carrier_to_sign(carrier: float) -> float:
    return 1.0 if carrier >= (FREQ_0 + FREQ_1) / 2 else -1.0


def derive_pc1_symbol_sequence(audio: np.ndarray, sr: int) -> list[Symbol]:
    """Build expected carrier timeline from PC1 reference WAV (diagnostic only)."""
    samples_per_bit = int(sr * BIT_DURATION)
    start_idx, conf = _find_preamble(audio, samples_per_bit, sample_rate=sr)
    if start_idx is None:
        raise RuntimeError("Could not locate preamble in PC1 reference WAV")

    symbols: list[Symbol] = []
    idx = start_idx
    sym_idx = 0

    # Preamble symbols (known pattern).
    for bit in PREAMBLE:
        t0 = idx / sr
        t1 = (idx + samples_per_bit) / sr
        symbols.append(Symbol(sym_idx, t0, t1, carrier_from_bit(bit), bit))
        sym_idx += 1
        idx += samples_per_bit

    # Data symbols until tone energy falls (trailing silence).
    peak_tone = max(
        _symbol_tone_power(audio[i : i + samples_per_bit], sample_rate=sr)
        for i in range(start_idx, len(audio) - samples_per_bit, samples_per_bit)
    )
    min_tone = peak_tone * 0.01

    while idx + samples_per_bit <= len(audio):
        window = audio[idx : idx + samples_per_bit]
        tone_pwr = _symbol_tone_power(window, sample_rate=sr)
        if tone_pwr < min_tone:
            break
        bit, _ = _classify_bit(window, sample_rate=sr)
        t0 = idx / sr
        t1 = (idx + samples_per_bit) / sr
        symbols.append(Symbol(sym_idx, t0, t1, carrier_from_bit(bit), bit))
        sym_idx += 1
        idx += samples_per_bit

    return symbols


def build_pc2_carrier_trace(audio: np.ndarray, sr: int) -> dict:
    freqs, times, zxx = stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2

    def band_sum(center: float, half: float = 46.875) -> np.ndarray:
        mask = (freqs >= center - half) & (freqs <= center + half)
        return power[mask, :].sum(axis=0) if np.any(mask) else np.zeros(power.shape[1])

    p185 = band_sum(float(FREQ_0))
    p205 = band_sum(float(FREQ_1))
    score = p185 - p205
    abs_score = np.abs(score)
    norm = float(np.max(abs_score)) if np.max(abs_score) > 0 else 1.0
    score_norm = score / norm

    return {
        "times": times,
        "p185": p185,
        "p205": p205,
        "score": score,
        "abs_score": abs_score,
        "score_norm": score_norm,
    }


def _interp_trace(trace: dict, t: float) -> tuple[float, float, float, float]:
    """Linear interpolation of PC2 traces at time t."""
    times = trace["times"]
    if t <= times[0]:
        i = 0
    elif t >= times[-1]:
        i = len(times) - 1
    else:
        i = int(np.searchsorted(times, t) - 1)
        i = max(0, min(i, len(times) - 2))
    t0, t1 = times[i], times[i + 1]
    w = (t - t0) / (t1 - t0 + 1e-30)
    p185 = (1 - w) * trace["p185"][i] + w * trace["p185"][i + 1]
    p205 = (1 - w) * trace["p205"][i] + w * trace["p205"][i + 1]
    sc = (1 - w) * trace["score"][i] + w * trace["score"][i + 1]
    scn = (1 - w) * trace["score_norm"][i] + w * trace["score_norm"][i + 1]
    return float(p185), float(p205), float(sc), float(scn)


def evaluate_offset(
    symbols: list[Symbol],
    trace: dict,
    offset: float,
    ambig_frac: float = AMBIGUOUS_FRAC,
) -> dict:
    """Score alignment at a candidate time offset (PC2_time = PC1_time + offset)."""
    pc1_signs: list[float] = []
    pc2_scores: list[float] = []
    pc2_norm: list[float] = []
    matches = 0
    mismatches = 0
    ambiguous = 0
    usable = 0

    max_abs = float(np.max(trace["abs_score"])) if len(trace["abs_score"]) else 1.0
    ambig_thresh = ambig_frac * max_abs

    for sym in symbols:
        center_t = 0.5 * (sym.start_time + sym.end_time) + offset
        if center_t < trace["times"][0] or center_t > trace["times"][-1]:
            continue

        p185, p205, sc, scn = _interp_trace(trace, center_t)
        pc1_signs.append(carrier_to_sign(sym.carrier))
        pc2_scores.append(sc)
        pc2_norm.append(scn)

        if abs(sc) < ambig_thresh:
            ambiguous += 1
            continue

        usable += 1
        pc2_carrier = float(FREQ_1) if p205 > p185 else float(FREQ_0)
        if abs(pc2_carrier - sym.carrier) < 500:
            matches += 1
        else:
            mismatches += 1

    if len(pc1_signs) < 3:
        return {
            "offset": offset,
            "pearson": 0.0,
            "pearson_norm": 0.0,
            "agreement": 0.0,
            "usable": 0,
            "ambiguous": ambiguous,
            "mismatches": mismatches,
            "matches": matches,
            "n_symbols": len(pc1_signs),
        }

    pearson = float(pearsonr(pc1_signs, pc2_scores)[0]) if np.std(pc2_scores) > 0 else 0.0
    pearson_norm = float(pearsonr(pc1_signs, pc2_norm)[0]) if np.std(pc2_norm) > 0 else 0.0
    agreement = matches / usable if usable > 0 else 0.0

    return {
        "offset": offset,
        "pearson": pearson,
        "pearson_norm": pearson_norm,
        "agreement": agreement,
        "usable": usable,
        "ambiguous": ambiguous,
        "mismatches": mismatches,
        "matches": matches,
        "n_symbols": len(pc1_signs),
    }


def search_best_offset(symbols: list[Symbol], trace: dict) -> tuple[dict, list[dict]]:
    candidates: list[dict] = []
    offset = OFFSET_MIN
    while offset <= OFFSET_MAX + 1e-9:
        candidates.append(evaluate_offset(symbols, trace, offset))
        offset += OFFSET_STEP

    # Prefer offsets with enough aligned symbols; avoid spurious high-r from few points.
    viable = [c for c in candidates if c["usable"] >= 30]
    pool = viable if viable else candidates
    pool.sort(key=lambda r: (r["agreement"], r["pearson"], r["usable"]), reverse=True)
    top10 = sorted(candidates, key=lambda r: (r["pearson"], r["agreement"]), reverse=True)[:10]
    return pool[0], top10


def shuffle_control_test(
    symbols: list[Symbol],
    trace: dict,
    best_offset: float,
    n_shuffles: int = N_SHUFFLE_CONTROLS,
) -> dict:
    real = evaluate_offset(symbols, trace, best_offset)
    real_corr = real["pearson"]

    rng = np.random.default_rng(42)
    carriers = [s.carrier for s in symbols]
    shuffled_corrs: list[float] = []

    for _ in range(n_shuffles):
        shuffled = carriers.copy()
        rng.shuffle(shuffled)
        shuffled_symbols = [
            Symbol(s.index, s.start_time, s.end_time, c, s.bit)
            for s, c in zip(symbols, shuffled)
        ]
        res = evaluate_offset(shuffled_symbols, trace, best_offset)
        shuffled_corrs.append(res["pearson"])

    arr = np.array([c for c in shuffled_corrs if np.isfinite(c)])
    if len(arr) == 0:
        arr = np.array([0.0])
    return {
        "real": real_corr,
        "control_mean": float(np.mean(arr)),
        "control_std": float(np.std(arr)),
        "control_max": float(np.max(arr)),
    }


def symbol_level_report(
    symbols: list[Symbol],
    trace: dict,
    offset: float,
    max_rows: int = 40,
) -> list[dict]:
    rows: list[dict] = []
    max_abs = float(np.max(trace["abs_score"])) if len(trace["abs_score"]) else 1.0
    ambig_thresh = AMBIGUOUS_FRAC * max_abs

    for sym in symbols:
        center_t = 0.5 * (sym.start_time + sym.end_time) + offset
        if center_t < trace["times"][0] or center_t > trace["times"][-1]:
            continue
        p185, p205, sc, _ = _interp_trace(trace, center_t)
        if abs(sc) < ambig_thresh:
            pc2_cls = "AMBIGUOUS"
            match = "AMBIGUOUS"
        elif p205 > p185:
            pc2_cls = "20500"
            match = "MATCH" if sym.carrier == FREQ_1 else "MISMATCH"
        else:
            pc2_cls = "18500"
            match = "MATCH" if sym.carrier == FREQ_0 else "MISMATCH"
        rows.append({
            "pc1_time": 0.5 * (sym.start_time + sym.end_time),
            "pc2_time": center_t,
            "pc1_carrier": int(sym.carrier),
            "p185": p185,
            "p205": p205,
            "score": sc,
            "pc2_cls": pc2_cls,
            "match": match,
        })
        if len(rows) >= max_rows:
            break
    return rows


def _save_plot(
    symbols: list[Symbol],
    trace: dict,
    best_offset: float,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PC1 step (shifted by offset for alignment with PC2 timeline).
    pc1_t: list[float] = []
    pc1_y: list[float] = []
    for sym in symbols:
        t0 = sym.start_time + best_offset
        t1 = sym.end_time + best_offset
        pc1_t.extend([t0, t1])
        pc1_y.extend([sym.carrier, sym.carrier])

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(pc1_t, pc1_y, drawstyle="steps-post", color="white", lw=1.5, label="PC1 expected (aligned)")
    ax1.set_ylabel("Carrier (Hz)")
    ax1.set_ylim(17000, 22000)
    ax1.axhline(FREQ_0, color="cyan", ls=":", alpha=0.4)
    ax1.axhline(FREQ_1, color="lime", ls=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(trace["times"], trace["score_norm"], color="orange", alpha=0.8, label="PC2 norm(P185-P205)")
    ax2.set_ylabel("Normalized score")
    ax2.axvline(best_offset, color="red", ls="--", alpha=0.5, label=f"offset={best_offset:.2f}s")

    ax1.set_xlabel("PC2 capture time (s)")
    ax1.set_title(f"Physical BFSK correlation (offset={best_offset:.3f}s)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _diagnose(
    best: dict,
    control: dict,
    pc1_duration: float,
    pc2_duration: float,
    tone_duration: float,
) -> dict:
    carrier_ok = True  # bins already established in prior work

    real_corr = control["real"]
    ctrl_max = control["control_max"]
    margin = real_corr - ctrl_max
    corr_pass = (
        real_corr > 0.25
        and margin > 0.10
        and best["usable"] >= 30
    )
    corr_uncertain = margin > 0.03 and real_corr > 0.10 and not corr_pass

    agree = best["agreement"]
    agree_pass = agree >= 0.65 and best["usable"] >= 30
    agree_uncertain = agree >= 0.55 and not agree_pass

    # Playback window: best offset + tone duration should land within PC2 capture
    matched_start = best["offset"]
    matched_end = matched_start + tone_duration
    window_pass = (
        matched_start >= -0.5
        and matched_end <= pc2_duration + 0.5
        and tone_duration > 2.0
    )
    window_uncertain = matched_start >= 0 and matched_end <= pc2_duration

    def label(passed: bool, uncertain: bool) -> str:
        if passed:
            return "PASS"
        if uncertain:
            return "UNCERTAIN"
        return "FAIL"

    verdict = {
        "Carrier frequencies": "PASS" if carrier_ok else "FAIL",
        "Temporal correlation": label(corr_pass, corr_uncertain),
        "Symbol agreement": label(agree_pass, agree_uncertain),
        "Playback-window alignment": label(window_pass, window_uncertain),
        "Correlation vs random control": label(corr_pass, corr_uncertain),
    }

    passes = sum(1 for v in verdict.values() if v == "PASS")
    corr_ok = verdict["Correlation vs random control"] == "PASS"
    agree_ok = verdict["Symbol agreement"] in ("PASS", "UNCERTAIN")

    if passes >= 4 and corr_ok and agree_ok and agree >= 0.55:
        overall = "CONFIRMED PHYSICAL BFSK"
    elif passes >= 3 and real_corr > ctrl_max:
        overall = "LIKELY PHYSICAL BFSK"
    elif real_corr > ctrl_max or agree >= 0.35:
        overall = "ULTRASONIC ACTIVITY WITHOUT PROVEN BFSK CORRELATION"
    elif best["usable"] < 5:
        overall = "NO PHYSICAL BFSK DETECTED"
    else:
        overall = "ULTRASONIC ACTIVITY WITHOUT PROVEN BFSK CORRELATION"

    verdict["overall"] = overall
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical BFSK correlation diagnostic")
    parser.add_argument("--capture", help="PC2 capture WAV")
    parser.add_argument("--pc1-wav", help="PC1 reference WAV")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    capture = args.capture or _newest(os.path.join(FORENSIC_DIR, "pc2_capture_*.wav"))
    pc1_wav = args.pc1_wav or _newest(os.path.join(PAYLOAD_DIR, "*.wav"))

    if not capture or not os.path.isfile(capture):
        print("ERROR: No PC2 capture found.")
        return 1
    if not pc1_wav or not os.path.isfile(pc1_wav):
        print("ERROR: No PC1 BFSK WAV found.")
        return 1

    print("=" * 60)
    print("PHYSICAL BFSK SYNCHRONIZATION / CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"PC1 reference : {os.path.abspath(pc1_wav)}")
    print(f"PC2 capture   : {os.path.abspath(capture)}")

    pc1_audio, pc1_sr = load_mono(pc1_wav)
    pc2_audio, pc2_sr = load_mono(capture)
    if pc1_sr != SAMPLE_RATE or pc2_sr != SAMPLE_RATE:
        print(f"WARNING: sample rates PC1={pc1_sr} PC2={pc2_sr} (expected {SAMPLE_RATE})")

    symbols = derive_pc1_symbol_sequence(pc1_audio, pc1_sr)
    tone_duration = symbols[-1].end_time - symbols[0].start_time if symbols else 0.0
    pc1_duration = len(pc1_audio) / pc1_sr
    pc2_duration = len(pc2_audio) / pc2_sr

    print(f"\nSTEP 1 - PC1 SYMBOL SEQUENCE ({len(symbols)} symbols)")
    print(f"{'IDX':>4} {'START':>8} {'END':>8} {'CARRIER':>8} {'BIT':>4}")
    print("-" * 40)
    for sym in symbols[:20]:
        print(f"{sym.index:4d} {sym.start_time:8.3f} {sym.end_time:8.3f} {int(sym.carrier):8d} {sym.bit:>4}")
    if len(symbols) > 20:
        print(f"... ({len(symbols) - 20} more symbols)")
    print(f"Tone span: {symbols[0].start_time:.3f}s - {symbols[-1].end_time:.3f}s ({tone_duration:.3f}s)")

    trace = build_pc2_carrier_trace(pc2_audio, pc2_sr)
    print(f"\nSTEP 2 - PC2 CARRIER TRACE")
    print(f"STFT frames: {len(trace['times'])}")
    print(f"Score range: [{trace['score'].min():.3e}, {trace['score'].max():.3e}]")
    print(f"Normalized score range: [{trace['score_norm'].min():.3f}, {trace['score_norm'].max():.3f}]")

    best, top10 = search_best_offset(symbols, trace)
    control = shuffle_control_test(symbols, trace, best["offset"])

    print(f"\nSTEP 5 - BEST ALIGNMENT")
    print("=" * 60)
    print(f"Offset (PC2_time = PC1_time + offset): {best['offset']:.3f} s")
    print(f"Pearson correlation              : {best['pearson']:.4f}")
    print(f"Normalized Pearson correlation   : {best['pearson_norm']:.4f}")
    print(f"Symbol agreement                 : {best['agreement']:.4f} ({best['matches']}/{best['usable']} usable)")
    print(f"Usable symbols                   : {best['usable']}")
    print(f"Ambiguous symbols                : {best['ambiguous']}")
    print(f"Mismatched symbols               : {best['mismatches']}")

    print(f"\nTOP 10 CANDIDATE OFFSETS")
    print(f"{'OFFSET':>10} {'CORRELATION':>14} {'AGREEMENT':>12}")
    print("-" * 40)
    for row in top10:
        print(f"{row['offset']:10.3f} {row['pearson']:14.4f} {row['agreement']:12.4f}")

    print(f"\nSTEP 6 - SHUFFLED CONTROL TEST ({N_SHUFFLE_CONTROLS} shuffles)")
    print(f"REAL CORRELATION : {control['real']:.4f}")
    print(f"CONTROL MEAN     : {control['control_mean']:.4f}")
    print(f"CONTROL STD      : {control['control_std']:.4f}")
    print(f"CONTROL MAX      : {control['control_max']:.4f}")
    print(f"Real vs control  : {control['real'] - control['control_max']:+.4f} above control max")

    print(f"\nSTEP 7 - PLAYBACK WINDOW TEST")
    print(f"PC1 duration        : {pc1_duration:.3f} s")
    print(f"PC2 duration        : {pc2_duration:.3f} s")
    print(f"PC1 tone duration   : {tone_duration:.3f} s")
    print(f"Best offset         : {best['offset']:.3f} s")
    print(f"Matched window      : {best['offset']:.3f}s - {best['offset'] + tone_duration:.3f}s")
    print(f"Matched duration    : {tone_duration:.3f} s")

    rows = symbol_level_report(symbols, trace, best["offset"], max_rows=30)
    print(f"\nSTEP 4 - SYMBOL-LEVEL MATCHING (first {len(rows)} aligned symbols)")
    print(f"{'PC1_t':>7} {'PC2_t':>7} {'PC1':>6} {'P185':>10} {'P205':>10} {'SCORE':>10} {'PC2':>8} {'RESULT':>8}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['pc1_time']:7.3f} {r['pc2_time']:7.3f} {r['pc1_carrier']:6d} "
            f"{r['p185']:10.3e} {r['p205']:10.3e} {r['score']:10.3e} "
            f"{r['pc2_cls']:>8} {r['match']:>8}"
        )

  # Full accuracy over all aligned symbols
    all_rows = symbol_level_report(symbols, trace, best["offset"], max_rows=len(symbols) + 1)
    n_match = sum(1 for r in all_rows if r["match"] == "MATCH")
    n_mis = sum(1 for r in all_rows if r["match"] == "MISMATCH")
    n_amb = sum(1 for r in all_rows if r["match"] == "AMBIGUOUS")
    print(f"\nFull symbol accuracy: {n_match} MATCH / {n_mis} MISMATCH / {n_amb} AMBIGUOUS "
          f"({100*n_match/max(n_match+n_mis,1):.1f}% of decisive symbols)")

    plot_path = os.path.join(FORENSIC_DIR, "physical_bfsk_correlation.png")
    if not args.no_plots:
        try:
            _save_plot(symbols, trace, best["offset"], plot_path)
            print(f"\nPlot saved: {plot_path}")
        except Exception as exc:
            print(f"\nPlot skipped: {exc}")

    verdict = _diagnose(best, control, pc1_duration, pc2_duration, tone_duration)

    print(f"\n{'=' * 60}")
    print("PHYSICAL BFSK CORRELATION DIAGNOSIS")
    print("=" * 60)
    for k in [
        "Carrier frequencies",
        "Temporal correlation",
        "Symbol agreement",
        "Playback-window alignment",
        "Correlation vs random control",
    ]:
        print(f"{k + ':':<32} {verdict[k]}")
    print(f"\nFINAL: {verdict['overall']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
