#!/usr/bin/env python3
"""
Controlled physical BFSK hardware test (diagnostic only).

Experiments:
  A - speaker OFF noise floor
  B1/B2 - single carrier 18500/20500 Hz
  C - physical BFSK with synchronized capture

Does NOT modify production code.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import stft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

FORENSIC_DIR = os.path.join(_REPO_ROOT, "forensic_captures")
PAYLOAD_DIR = os.path.join(_REPO_ROOT, "generated_payloads")

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.payload_service import encode_text_to_signal, save_signal_to_wav

# Reuse correlation analysis from existing diagnostic module.
from scripts.analyze_physical_bfsk_correlation import (
    build_pc2_carrier_trace,
    derive_pc1_symbol_sequence,
    evaluate_offset,
    shuffle_control_test,
    load_mono,
)

MIC_DEVICE = 1
SPK_DEVICE = 3
SR = SAMPLE_RATE
PRE_SEC = 1.0
POST_SEC = 1.0
TONE_SEC = 5.0
SILENCE_SEC = 12.0
STFT_NPERSEG = 2048
STFT_NOVERLAP = 1536
RATIO = 1.5


@dataclass
class Analysis:
    rms: float
    peak: float
    p185: float
    p205: float
    noise_floor: float
    snr_185: float
    snr_205: float
    peak_freq: float
    signal_start: float
    signal_end: float
    median_symbol_ms: float
    transitions: int
    ambiguous_pct: float
    correlation: float | None = None
    agreement: float | None = None
    best_offset: float | None = None


def save_wav(path: str, audio: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, SR, (pcm * 32767).astype(np.int16))


def generate_tone(freq: float, duration_sec: float, amplitude: float = 0.5) -> np.ndarray:
    n = int(SR * duration_sec)
    t = np.arange(n, dtype=np.float64) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def record_only(duration_sec: float) -> np.ndarray:
    frames = int(duration_sec * SR)
    print(f"  [REC] device={MIC_DEVICE} duration={duration_sec:.1f}s (speaker OFF)")
    audio = sd.rec(frames, samplerate=SR, channels=1, dtype="float32", device=MIC_DEVICE)
    sd.wait()
    return audio[:, 0].copy()


def synchronized_capture(play_audio: np.ndarray, label: str) -> np.ndarray:
    play_dur = len(play_audio) / SR
    total = PRE_SEC + play_dur + POST_SEC
    frames = int(total * SR)
    print(f"  [SYNC] {label}: pre={PRE_SEC}s play={play_dur:.2f}s post={POST_SEC}s total={total:.2f}s")
    print(f"  [REC] mic device={MIC_DEVICE}")
    recording = sd.rec(frames, samplerate=SR, channels=1, dtype="float32", device=MIC_DEVICE)
    time.sleep(PRE_SEC)
    print(f"  [PLAY] speaker device={SPK_DEVICE}")
    sd.play(play_audio, samplerate=SR, device=SPK_DEVICE)
    sd.wait()
    time.sleep(POST_SEC)
    sd.wait()
    return recording[:, 0].copy()


def _whole_file_metrics(audio: np.ndarray, noise_floor: float | None = None) -> dict:
    n = len(audio)
    window = np.hanning(n)
    spec = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    power = (np.abs(spec) ** 2) / max(n, 1)

    def band(center: float, hw: float = 100.0) -> float:
        m = (freqs >= center - hw) & (freqs <= center + hw)
        return float(np.max(power[m])) if np.any(m) else 0.0

    us = (freqs >= 18000) & (freqs <= 21000)
    if noise_floor is None:
        excl = ((freqs >= FREQ_0 - 100) & (freqs <= FREQ_0 + 100)) | (
            (freqs >= FREQ_1 - 100) & (freqs <= FREQ_1 + 100)
        )
        nf_mask = us & ~excl
        noise_floor = float(np.median(power[nf_mask])) if np.any(nf_mask) else 1e-30

    p185 = band(FREQ_0)
    p205 = band(FREQ_1)
    us_peak_idx = int(np.argmax(power[us])) if np.any(us) else 0
    peak_freq = float(freqs[us][us_peak_idx]) if np.any(us) else 0.0

    return {
        "rms": float(np.sqrt(np.mean(audio**2))),
        "peak": float(np.max(np.abs(audio))),
        "p185": p185,
        "p205": p205,
        "noise_floor": noise_floor,
        "snr_185": 10 * np.log10((p185 + 1e-30) / (noise_floor + 1e-30)),
        "snr_205": 10 * np.log10((p205 + 1e-30) / (noise_floor + 1e-30)),
        "peak_freq": peak_freq,
    }


def _activity_window(trace: dict) -> tuple[float, float]:
    combined = trace["p185"] + trace["p205"]
    if len(combined) == 0:
        return 0.0, 0.0
    thresh = np.percentile(combined, 75)
    active = combined >= thresh
    if not np.any(active):
        return 0.0, 0.0
    idx = np.where(active)[0]
    start = float(trace["times"][idx[0]])
    end = float(trace["times"][idx[-1]]) + STFT_NPERSEG / SR
    return start, end


def _timing_from_trace(trace: dict) -> tuple[float, int, float]:
    p185, p205 = trace["p185"], trace["p205"]
    states = []
    for a, b in zip(p185, p205):
        if a <= 0 and b <= 0:
            states.append("NONE")
        elif a > RATIO * b:
            states.append("18500")
        elif b > RATIO * a:
            states.append("20500")
        else:
            states.append("AMBIGUOUS")

    segments = []
    i = 0
    while i < len(states):
        st = states[i]
        if st not in ("18500", "20500"):
            i += 1
            continue
        j = i + 1
        while j < len(states) and states[j] == st:
            j += 1
        t0 = trace["times"][i]
        t1 = trace["times"][j - 1] + STFT_NPERSEG / SR
        segments.append((t1 - t0) * 1000.0)
        i = j

    transitions = 0
    prev = None
    for st in states:
        if st in ("18500", "20500"):
            if prev and st != prev:
                transitions += 1
            prev = st

    amb_pct = 100.0 * states.count("AMBIGUOUS") / max(len(states), 1)
    median_ms = float(np.median(segments)) if segments else 0.0
    return median_ms, transitions, amb_pct


def analyze_capture(path: str, noise_floor: float | None = None) -> Analysis:
    audio, _ = load_mono(path)
    m = _whole_file_metrics(audio, noise_floor)
    trace = build_pc2_carrier_trace(audio, SR)
    sig_start, sig_end = _activity_window(trace)
    med_ms, trans, amb = _timing_from_trace(trace)
    return Analysis(
        rms=m["rms"],
        peak=m["peak"],
        p185=m["p185"],
        p205=m["p205"],
        noise_floor=m["noise_floor"],
        snr_185=m["snr_185"],
        snr_205=m["snr_205"],
        peak_freq=m["peak_freq"],
        signal_start=sig_start,
        signal_end=sig_end,
        median_symbol_ms=med_ms,
        transitions=trans,
        ambiguous_pct=amb,
    )


def correlation_vs_reference(capture_path: str, reference_path: str) -> dict:
    pc1_audio, _ = load_mono(reference_path)
    pc2_audio, _ = load_mono(capture_path)
    symbols = derive_pc1_symbol_sequence(pc1_audio, SR)
    trace = build_pc2_carrier_trace(pc2_audio, SR)

    # Search offset constrained to plausible playback window: 0 to 3 seconds.
    best = None
    for off in np.arange(0.0, 3.01, 0.02):
        r = evaluate_offset(symbols, trace, float(off))
        if best is None or (r["agreement"], r["pearson"], r["usable"]) > (
            best["agreement"],
            best["pearson"],
            best["usable"],
        ):
            best = r

    control = shuffle_control_test(symbols, trace, best["offset"])
    return {**best, **control}


def print_bfsk_timeline(path: str, max_rows: int = 0) -> None:
    audio, _ = load_mono(path)
    trace = build_pc2_carrier_trace(audio, SR)
    p185_max = max(float(np.max(trace["p185"])), 1e-30)
    p205_max = max(float(np.max(trace["p205"])), 1e-30)
    print(f"\n{'TIME':>8} {'P18500':>12} {'P20500':>12} {'N185':>8} {'N205':>8} {'STATE':>10}")
    print("-" * 70)
    n = len(trace["times"])
    indices = range(n) if max_rows == 0 else range(0, min(n, max_rows))
    for i in indices:
        t = trace["times"][i]
        a, b = trace["p185"][i], trace["p205"][i]
        na = "HIGH" if a >= 0.25 * p185_max else "LOW"
        nb = "HIGH" if b >= 0.25 * p205_max else "LOW"
        if a > RATIO * b:
            st = "18500"
        elif b > RATIO * a:
            st = "20500"
        elif a > 0 or b > 0:
            st = "AMBIGUOUS"
        else:
            st = "NONE"
        print(f"{t:8.3f} {a:12.3e} {b:12.3e} {na:>8} {nb:>8} {st:>10}")


def save_plot(
    paths: dict[str, str],
    corr: dict,
    ref_path: str,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=False)

    labels = [
        ("A OFF", paths["silence"]),
        ("B1 18500", paths["18500"]),
        ("B2 20500", paths["20500"]),
        ("C BFSK", paths["bfsk"]),
    ]
    for ax, (title, p) in zip(axes[:4], labels):
        audio, _ = load_mono(p)
        tr = build_pc2_carrier_trace(audio, SR)
        ax.plot(tr["times"], tr["p185"], label="P18500", color="cyan", alpha=0.8)
        ax.plot(tr["times"], tr["p205"], label="P20500", color="lime", alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Band power")
        ax.legend(loc="upper right", fontsize=7)

    # Panel 5: BFSK correlation alignment
    ax = axes[4]
    bfsk_audio, _ = load_mono(paths["bfsk"])
    tr = build_pc2_carrier_trace(bfsk_audio, SR)
    ax.plot(tr["times"], tr["score_norm"], color="orange", label="norm(P185-P205)")
    offset = corr["offset"]
    pc1_audio, _ = load_mono(ref_path)
    syms = derive_pc1_symbol_sequence(pc1_audio, SR)
    for sym in syms:
        t = 0.5 * (sym.start_time + sym.end_time) + offset
        if 0 <= t <= tr["times"][-1]:
            ax.axvline(t, color="white", alpha=0.05, lw=0.5)
    ax.axvspan(PRE_SEC, PRE_SEC + 7.8, color="green", alpha=0.15, label="expected window")
    ax.axvline(offset, color="red", ls="--", label=f"best offset={offset:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_title("Physical BFSK score + expected playback window")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def diagnose(
    a: Analysis,
    b1: Analysis,
    b2: Analysis,
    c: Analysis,
    corr: dict,
    noise_floor: float,
) -> str:
    b1_ok = b1.snr_185 > a.snr_185 + 6
    b2_ok = b2.snr_205 > a.snr_205 + 6
    if not b1_ok and not b2_ok:
        return "4. SPEAKER/MICROPHONE CHANNEL FAILURE"
    if b1.snr_185 < 3 and b2.snr_205 < 3:
        return "3. PHYSICAL ULTRASONIC SIGNAL TOO WEAK"

    window_ok = PRE_SEC <= c.signal_start <= PRE_SEC + 0.5 and 7.0 <= c.signal_end <= PRE_SEC + 8.8
    corr_ok = (
        corr["pearson"] > 0.25
        and corr["pearson"] > corr["control_max"] + 0.05
        and corr["agreement"] >= 0.65
        and corr["usable"] >= 30
        and window_ok
    )
    if corr_ok:
        return "1. PHYSICAL BFSK CONFIRMED"

    carriers_ok = c.snr_185 > noise_floor and c.snr_205 > noise_floor
    if carriers_ok:
        return "2. PHYSICAL CARRIERS DETECTED, BFSK NOT PROVEN"
    return "3. PHYSICAL ULTRASONIC SIGNAL TOO WEAK"


def main() -> int:
    os.makedirs(FORENSIC_DIR, exist_ok=True)
    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    py = sys.executable
    mic_info = sd.query_devices(MIC_DEVICE)
    spk_info = sd.query_devices(SPK_DEVICE)

    log("=" * 60)
    log("CONTROLLED PHYSICAL BFSK HARDWARE TEST")
    log("=" * 60)
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Python: {py}")
    log(f"Microphone [{MIC_DEVICE}]: {mic_info['name']}")
    log(f"Speaker [{SPK_DEVICE}]: {spk_info['name']}")

    # ---- Experiment A ----
    log("\nEXPERIMENT A — SPEAKER OFF")
    path_a = os.path.join(FORENSIC_DIR, "control_silence.wav")
    audio_a = record_only(SILENCE_SEC)
    save_wav(path_a, audio_a)
    a = analyze_capture(path_a)
    noise_floor = a.noise_floor
    log("\n" + "=" * 48)
    log("CONTROL A — SPEAKER OFF")
    log("=" * 48)
    log(f"RMS: {a.rms:.6e}")
    log(f"PEAK: {a.peak:.6e}")
    log(f"18500 POWER: {a.p185:.6e}")
    log(f"20500 POWER: {a.p205:.6e}")
    log(f"ULTRASONIC NOISE FLOOR: {a.noise_floor:.6e}")
    log(f"DETECTED: {a.p185 > noise_floor * 3 or a.p205 > noise_floor * 3}")
    log("=" * 48)

    # ---- Experiment B1 ----
    log("\nEXPERIMENT B1 — 18500 Hz CONTINUOUS TONE")
    tone185 = generate_tone(FREQ_0, TONE_SEC)
    path_b1 = os.path.join(FORENSIC_DIR, "control_18500.wav")
    audio_b1 = synchronized_capture(tone185, "B1-18500")
    save_wav(path_b1, audio_b1)
    b1 = analyze_capture(path_b1, noise_floor=noise_floor)
    log(f"  Signal start/end: {b1.signal_start:.3f}s - {b1.signal_end:.3f}s")
    log(f"  18500 SNR: {b1.snr_185:.2f} dB (baseline {a.snr_185:.2f} dB)")

    # ---- Experiment B2 ----
    log("\nEXPERIMENT B2 — 20500 Hz CONTINUOUS TONE")
    tone205 = generate_tone(FREQ_1, TONE_SEC)
    path_b2 = os.path.join(FORENSIC_DIR, "control_20500.wav")
    audio_b2 = synchronized_capture(tone205, "B2-20500")
    save_wav(path_b2, audio_b2)
    b2 = analyze_capture(path_b2, noise_floor=noise_floor)
    log(f"  Signal start/end: {b2.signal_start:.3f}s - {b2.signal_end:.3f}s")
    log(f"  20500 SNR: {b2.snr_205:.2f} dB (baseline {a.snr_205:.2f} dB)")

    # ---- Experiment C ----
    log("\nEXPERIMENT C — PHYSICAL BFSK")
    payload = "SIH_PC1_PC2_TEST"
    bfsk_signal = encode_text_to_signal(payload)
    payload_id = str(uuid.uuid4())[:8]
    ref_wav = os.path.join(PAYLOAD_DIR, f"{payload_id}.wav")
    save_signal_to_wav(ref_wav, bfsk_signal, SR)
    log(f"  Generated reference WAV: {ref_wav}")

    path_c = os.path.join(FORENSIC_DIR, "control_physical_bfsk.wav")
    audio_c = synchronized_capture(bfsk_signal.astype(np.float32), "C-BFSK")
    save_wav(path_c, audio_c)
    c = analyze_capture(path_c, noise_floor=noise_floor)

    log("\nBFSK TIMELINE (physical capture, first 80 frames):")
    print_bfsk_timeline(path_c, max_rows=80)

    corr = correlation_vs_reference(path_c, ref_wav)
    c.correlation = corr["pearson"]
    c.agreement = corr["agreement"]
    c.best_offset = corr["offset"]

    log("\nCORRELATION vs PC1 reference:")
    log(f"  Best offset: {corr['offset']:.3f} s")
    log(f"  Real correlation: {corr['real']:.4f}")
    log(f"  Control mean: {corr['control_mean']:.4f}")
    log(f"  Control std: {corr['control_std']:.4f}")
    log(f"  Control max: {corr['control_max']:.4f}")
    log(f"  Symbol agreement: {corr['agreement']:.4f} ({corr['matches']}/{corr['usable']})")

    paths = {"silence": path_a, "18500": path_b1, "20500": path_b2, "bfsk": path_c}
    plot_path = os.path.join(FORENSIC_DIR, "hardware_bfsk_control.png")
    try:
        save_plot(paths, corr, ref_wav, plot_path)
        log(f"\nPlot saved: {plot_path}")
    except Exception as exc:
        log(f"\nPlot skipped: {exc}")

    log("\nCONTROL COMPARISON")
    log(f"{'TEST':<12} {'18.5kHz':>12} {'20.5kHz':>12} {'SNR185':>8} {'SNR205':>8} {'MEDms':>8} {'CORR':>8}")
    log("-" * 72)
    for name, an, corr_val in [
        ("OFF", a, None),
        ("18500", b1, None),
        ("20500", b2, None),
        ("BFSK", c, corr["pearson"]),
    ]:
        cv = f"{corr_val:.3f}" if corr_val is not None else "N/A"
        log(
            f"{name:<12} {an.p185:12.3e} {an.p205:12.3e} {an.snr_185:8.2f} "
            f"{an.snr_205:8.2f} {an.median_symbol_ms:8.1f} {cv:>8}"
        )

    final = diagnose(a, b1, b2, c, corr, noise_floor)

    log("\n" + "=" * 48)
    log("HARDWARE BFSK TEST RESULT")
    log("=" * 48)
    log(f"Environment:")
    log(f"  Python: {py}")
    log(f"  Microphone: [{MIC_DEVICE}] {mic_info['name']}")
    log(f"  Speaker: [{SPK_DEVICE}] {spk_info['name']}")
    log(f"\nEXPERIMENT A — SPEAKER OFF:")
    log(f"  Result: captured {path_a}")
    log(f"  18.5 kHz: {a.p185:.3e} SNR={a.snr_185:.2f} dB")
    log(f"  20.5 kHz: {a.p205:.3e} SNR={a.snr_205:.2f} dB")
    log(f"  Noise floor: {noise_floor:.3e}")
    log(f"\nEXPERIMENT B1 — 18500 Hz:")
    log(f"  Result: {'PASS' if b1.snr_185 > a.snr_185 + 6 else 'FAIL'}")
    log(f"  Signal level: {b1.p185:.3e}")
    log(f"  SNR: {b1.snr_185:.2f} dB")
    log(f"  Window: {b1.signal_start:.3f}s - {b1.signal_end:.3f}s")
    log(f"\nEXPERIMENT B2 — 20500 Hz:")
    log(f"  Result: {'PASS' if b2.snr_205 > a.snr_205 + 6 else 'FAIL'}")
    log(f"  Signal level: {b2.p205:.3e}")
    log(f"  SNR: {b2.snr_205:.2f} dB")
    log(f"  Window: {b2.signal_start:.3f}s - {b2.signal_end:.3f}s")
    log(f"\nEXPERIMENT C — PHYSICAL BFSK:")
    log(f"  Generated WAV: {ref_wav}")
    log(f"  Capture WAV: {path_c}")
    log(f"  Signal start: {c.signal_start:.3f} s")
    log(f"  Signal end: {c.signal_end:.3f} s")
    log(f"  18500 detected: SNR={c.snr_185:.2f} dB")
    log(f"  20500 detected: SNR={c.snr_205:.2f} dB")
    log(f"  Median symbol duration: {c.median_symbol_ms:.2f} ms")
    log(f"  Transition count: {c.transitions}")
    log(f"  Ambiguous percentage: {c.ambiguous_pct:.1f}%")
    log(f"\nCORRELATION:")
    log(f"  Best offset: {corr['offset']:.3f} s")
    log(f"  Real correlation: {corr['real']:.4f}")
    log(f"  Control mean: {corr['control_mean']:.4f}")
    log(f"  Control std: {corr['control_std']:.4f}")
    log(f"  Control max: {corr['control_max']:.4f}")
    log(f"  Symbol agreement: {corr['agreement']:.4f}")
    log(f"\nFINAL DIAGNOSIS:")
    log(f"  {final}")
    log("=" * 48)

    report_path = os.path.join(FORENSIC_DIR, "hardware_bfsk_control_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
