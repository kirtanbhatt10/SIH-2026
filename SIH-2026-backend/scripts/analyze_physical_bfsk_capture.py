#!/usr/bin/env python3
"""
Physical PC1 -> PC2 BFSK capture forensic analysis (read-only diagnostic).

Finds newest forensic_captures/pc2_capture_*.wav and generated_payloads/*.wav,
performs ultrasonic/STFT/transition analysis, optional plots, prints report.

Does NOT modify DSP, backend, ML, or source audio.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORENSIC_DIR = os.path.join(_REPO_ROOT, "forensic_captures")
PAYLOAD_DIR = os.path.join(_REPO_ROOT, "generated_payloads")

# Expected BFSK parameters (read-only reference; do not modify config files).
FREQ_0 = 18500.0
FREQ_1 = 20500.0
BIT_DURATION_MS = 50.0
SAMPLE_RATE_EXPECTED = 48000
ULTRASONIC_LOW = 18000.0
ULTRASONIC_HIGH = 21000.0
CARRIER_BAND_HZ = 100.0

STFT_NPERSEG = 2048
STFT_NOVERLAP = 1536
STFT_HOP = STFT_NPERSEG - STFT_NOVERLAP  # 512 samples â†’ ~10.67 ms hop


@dataclass
class WavInfo:
    path: str
    sample_rate: int
    samples: int
    duration_sec: float
    channels: int
    rms: float
    peak: float


@dataclass
class BandMetrics:
    center_hz: float
    peak_freq_hz: float
    peak_power: float
    mean_power: float
    snr_db: float


@dataclass
class FileAnalysis:
    info: WavInfo
    total_energy: float
    ultrasonic_energy: float
    ultrasonic_ratio: float
    global_peak_freq: float
    global_peak_power: float
    noise_floor: float
    band_18500: BandMetrics
    band_20500: BandMetrics
    activity_start_sec: float
    activity_end_sec: float
    activity_duration_sec: float
    symbol_duration_ms: float
    transition_count: int
    transition_intervals_ms: list[float]
    timing_confidence: float
    frame_times: np.ndarray
    power_18500: np.ndarray
    power_20500: np.ndarray
    snr_18500: np.ndarray
    snr_20500: np.ndarray
    classifications: list[str]
    evidence: dict[str, str]
    bfsk_score: int


def _newest_file(pattern: str) -> Optional[str]:
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def load_mono_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    elif data.dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr, channels


def wav_info(path: str) -> WavInfo:
    audio, sr, channels = load_mono_wav(path)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    return WavInfo(
        path=os.path.abspath(path),
        sample_rate=sr,
        samples=len(audio),
        duration_sec=len(audio) / sr,
        channels=channels,
        rms=rms,
        peak=peak,
    )


def _band_mask(freqs: np.ndarray, center: float, half_width: float = CARRIER_BAND_HZ) -> np.ndarray:
    return (freqs >= center - half_width) & (freqs <= center + half_width)


def _ultrasonic_mask(freqs: np.ndarray) -> np.ndarray:
    return (freqs >= ULTRASONIC_LOW) & (freqs <= ULTRASONIC_HIGH)


def _whole_file_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(audio)
    window = np.hanning(n)
    spec = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power = (np.abs(spec) ** 2) / max(n, 1)
    return freqs, power


def _band_metrics_from_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    center: float,
    noise_floor: float,
) -> BandMetrics:
    mask = _band_mask(freqs, center)
    if not np.any(mask):
        return BandMetrics(center, center, 0.0, 0.0, -np.inf)
    band_power = power[mask]
    band_freqs = freqs[mask]
    peak_idx = int(np.argmax(band_power))
    peak_power = float(band_power[peak_idx])
    mean_power = float(np.mean(band_power))
    snr = 10.0 * np.log10((peak_power + 1e-30) / (noise_floor + 1e-30))
    return BandMetrics(
        center_hz=center,
        peak_freq_hz=float(band_freqs[peak_idx]),
        peak_power=peak_power,
        mean_power=mean_power,
        snr_db=snr,
    )


def _estimate_noise_floor(freqs: np.ndarray, power: np.ndarray) -> float:
    us_mask = _ultrasonic_mask(freqs)
    us_power = power[us_mask]
    if len(us_power) == 0:
        return 1e-30
    # Exclude carrier bands from noise estimate.
    carrier_excl = _band_mask(freqs, FREQ_0) | _band_mask(freqs, FREQ_1)
    noise_mask = us_mask & ~carrier_excl
    if np.any(noise_mask):
        return float(np.median(power[noise_mask]))
    return float(np.median(us_power))


def _stft_analysis(audio: np.ndarray, sr: int) -> dict:
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
    us_mask = _ultrasonic_mask(freqs)
    m185 = _band_mask(freqs, FREQ_0)
    m205 = _band_mask(freqs, FREQ_1)
    carrier_excl = m185 | m205
    noise_mask = us_mask & ~carrier_excl

    p185 = power[m185, :].sum(axis=0)
    p205 = power[m205, :].sum(axis=0)
    pus = power[us_mask, :].sum(axis=0)
    pnoise = power[noise_mask, :].mean(axis=0) if np.any(noise_mask) else np.median(power, axis=0)

    # Per-frame noise floor from non-carrier ultrasonic bins.
    noise_floor_frames = np.median(power[noise_mask, :], axis=0) if np.any(noise_mask) else pnoise
    noise_floor_frames = np.maximum(noise_floor_frames, 1e-30)

    snr185 = 10.0 * np.log10((p185 + 1e-30) / noise_floor_frames)
    snr205 = 10.0 * np.log10((p205 + 1e-30) / noise_floor_frames)

    # Dominant ultrasonic frequency per frame.
    us_power = power[us_mask, :]
    us_freqs = freqs[us_mask]
    dom_idx = np.argmax(us_power, axis=0)
    dom_freq = us_freqs[dom_idx]

    return {
        "times": times,
        "freqs": freqs,
        "power": power,
        "p185": p185,
        "p205": p205,
        "pus": pus,
        "noise_floor_frames": noise_floor_frames,
        "snr185": snr185,
        "snr205": snr205,
        "dom_freq": dom_freq,
    }


def _derive_snr_threshold(snr185: np.ndarray, snr205: np.ndarray, pus: np.ndarray) -> tuple[float, str]:
    """Adaptive threshold from quiet-frame SNR distribution."""
    quiet = (pus < np.percentile(pus, 25)) | ((snr185 < 3.0) & (snr205 < 3.0))
    if np.sum(quiet) < 5:
        quiet = np.ones_like(pus, dtype=bool)
    pooled = np.concatenate([snr185[quiet], snr205[quiet]])
    noise_snr_median = float(np.median(pooled))
    noise_snr_p90 = float(np.percentile(pooled, 90))
    threshold = max(6.0, noise_snr_p90 + 3.0)
    explanation = (
        f"quiet-frame SNR median={noise_snr_median:.2f} dB, "
        f"p90={noise_snr_p90:.2f} dB -> threshold={threshold:.2f} dB "
        f"(max(6 dB, p90+3 dB))"
    )
    return threshold, explanation


def _classify_frames(
    snr185: np.ndarray,
    snr205: np.ndarray,
    threshold: float,
) -> list[str]:
    out: list[str] = []
    for s0, s1 in zip(snr185, snr205):
        if s0 > threshold and s0 > s1:
            out.append("18500")
        elif s1 > threshold and s1 > s0:
            out.append("20500")
        else:
            out.append("NONE")
    return out


def _transition_stats(classifications: list[str], frame_times: np.ndarray) -> tuple[int, list[float], float, float]:
    """Count carrier transitions and estimate symbol duration."""
    transitions = 0
    intervals_ms: list[float] = []
    run_start_idx = 0
    current = classifications[0] if classifications else "NONE"

    for i in range(1, len(classifications)):
        if classifications[i] != current and classifications[i] in ("18500", "20500") and current in ("18500", "20500"):
            transitions += 1
            dt = (frame_times[i] - frame_times[run_start_idx]) * 1000.0
            if dt > 5.0:
                intervals_ms.append(dt)
            run_start_idx = i
            current = classifications[i]
        elif classifications[i] in ("18500", "20500") and current == "NONE":
            run_start_idx = i
            current = classifications[i]
        elif classifications[i] in ("18500", "20500") and current in ("18500", "20500") and classifications[i] != current:
            pass  # handled above

    # Re-scan for alternation edges more simply.
    transitions = 0
    intervals_ms = []
    prev = None
    seg_start_t = frame_times[0] if len(frame_times) else 0.0
    for i, cls in enumerate(classifications):
        if cls not in ("18500", "20500"):
            continue
        if prev is None:
            prev = cls
            seg_start_t = frame_times[i]
            continue
        if cls != prev:
            transitions += 1
            intervals_ms.append((frame_times[i] - seg_start_t) * 1000.0)
            seg_start_t = frame_times[i]
            prev = cls

    if intervals_ms:
        median_ms = float(np.median(intervals_ms))
        mad = float(np.median(np.abs(np.array(intervals_ms) - median_ms)))
        timing_conf = float(max(0.0, min(1.0, 1.0 - mad / max(median_ms, 1.0))))
    else:
        median_ms = 0.0
        timing_conf = 0.0

    return transitions, intervals_ms, median_ms, timing_conf


def _activity_window(pus: np.ndarray, times: np.ndarray) -> tuple[float, float, float]:
    if len(pus) == 0:
        return 0.0, 0.0, 0.0
    thresh = np.percentile(pus, 75)
    active = pus >= thresh
    if not np.any(active):
        return 0.0, 0.0, 0.0
    idx = np.where(active)[0]
    start = float(times[idx[0]])
    end = float(times[idx[-1]] + STFT_NPERSEG / SAMPLE_RATE_EXPECTED)
    return start, end, end - start


def _evidence_label(ok: bool, uncertain: bool = False) -> str:
    if uncertain:
        return "UNCERTAIN"
    return "PASS" if ok else "FAIL"


def analyze_file(path: str, label: str) -> FileAnalysis:
    audio, sr, channels = load_mono_wav(path)
    info = wav_info(path)

    freqs, power = _whole_file_spectrum(audio, sr)
    us_mask = _ultrasonic_mask(freqs)
    total_energy = float(np.sum(power))
    ultrasonic_energy = float(np.sum(power[us_mask]))
    ultrasonic_ratio = ultrasonic_energy / (total_energy + 1e-30)
    noise_floor = _estimate_noise_floor(freqs, power)

    if np.any(us_mask):
        us_power = power[us_mask]
        us_freqs = freqs[us_mask]
        gpk = int(np.argmax(us_power))
        global_peak_freq = float(us_freqs[gpk])
        global_peak_power = float(us_power[gpk])
    else:
        global_peak_freq = 0.0
        global_peak_power = 0.0

    b185 = _band_metrics_from_spectrum(freqs, power, FREQ_0, noise_floor)
    b205 = _band_metrics_from_spectrum(freqs, power, FREQ_1, noise_floor)

    st = _stft_analysis(audio, sr)
    threshold, _ = _derive_snr_threshold(st["snr185"], st["snr205"], st["pus"])
    classifications = _classify_frames(st["snr185"], st["snr205"], threshold)
    transitions, intervals_ms, symbol_ms, timing_conf = _transition_stats(classifications, st["times"])
    act_start, act_end, act_dur = _activity_window(st["pus"], st["times"])

    n185 = sum(1 for c in classifications if c == "18500")
    n205 = sum(1 for c in classifications if c == "20500")
    n_frames = max(len(classifications), 1)

    # Evidence scoring (7 categories).
    ev_185 = b185.snr_db >= 10.0 and n185 >= 3
    ev_205 = b205.snr_db >= 10.0 and n205 >= 3
    ev_alt = n185 >= 5 and n205 >= 5 and transitions >= 3
    ev_timing = symbol_ms > 0 and abs(symbol_ms - BIT_DURATION_MS) <= 25.0 and timing_conf >= 0.3
    ev_duration = act_dur > 2.0
    ev_snr = (b185.snr_db >= 6.0 or b205.snr_db >= 6.0) and (
        np.median(st["snr185"]) > 3.0 or np.median(st["snr205"]) > 3.0
    )
    ev_consistency = (
        abs(b185.peak_freq_hz - FREQ_0) <= 250
        and abs(b205.peak_freq_hz - FREQ_1) <= 250
    )

    evidence = {
        "18.5 kHz carrier": _evidence_label(ev_185, uncertain=b185.snr_db >= 3 and not ev_185),
        "20.5 kHz carrier": _evidence_label(ev_205, uncertain=b205.snr_db >= 3 and not ev_205),
        "Alternation": _evidence_label(ev_alt, uncertain=transitions >= 1 and not ev_alt),
        "50 ms timing": _evidence_label(ev_timing, uncertain=symbol_ms > 0 and not ev_timing),
        "Activity duration": _evidence_label(ev_duration),
        "SNR": _evidence_label(ev_snr, uncertain=not ev_snr and (b185.snr_db > 0 or b205.snr_db > 0)),
        "Carrier consistency": _evidence_label(ev_consistency, uncertain=not ev_consistency),
    }
    score = sum(1 for v in evidence.values() if v == "PASS")

    return FileAnalysis(
        info=info,
        total_energy=total_energy,
        ultrasonic_energy=ultrasonic_energy,
        ultrasonic_ratio=ultrasonic_ratio,
        global_peak_freq=global_peak_freq,
        global_peak_power=global_peak_power,
        noise_floor=noise_floor,
        band_18500=b185,
        band_20500=b205,
        activity_start_sec=act_start,
        activity_end_sec=act_end,
        activity_duration_sec=act_dur,
        symbol_duration_ms=symbol_ms,
        transition_count=transitions,
        transition_intervals_ms=intervals_ms,
        timing_confidence=timing_conf,
        frame_times=st["times"],
        power_18500=st["p185"],
        power_20500=st["p205"],
        snr_18500=st["snr185"],
        snr_20500=st["snr205"],
        classifications=classifications,
        evidence=evidence,
        bfsk_score=score,
    )


def _print_wav_info(info: WavInfo) -> None:
    print(f"  path      : {info.path}")
    print(f"  sample_rate: {info.sample_rate} Hz")
    print(f"  samples   : {info.samples}")
    print(f"  duration  : {info.duration_sec:.3f} s")
    print(f"  channels  : {info.channels}")
    print(f"  RMS       : {info.rms:.6e}")
    print(f"  peak      : {info.peak:.6e}")


def _print_band(label: str, b: BandMetrics) -> None:
    print(f"{label}:")
    print(f"    peak frequency : {b.peak_freq_hz:.2f} Hz")
    print(f"    peak power     : {b.peak_power:.6e}")
    print(f"    mean power     : {b.mean_power:.6e}")
    print(f"    estimated SNR  : {b.snr_db:.2f} dB")


def _print_timeline(times: np.ndarray, p185: np.ndarray, p205: np.ndarray, classifications: list[str], max_rows: int = 80) -> None:
    print(f"{'TIME':>8}  {'18.5kHz':>12}  {'20.5kHz':>12}  {'WINNER':>8}")
    n = len(times)
    if n <= max_rows:
        indices = range(n)
    else:
        step = max(1, n // max_rows)
        indices = list(range(0, n, step))
    p185_max = max(float(np.max(p185)), 1e-30)
    p205_max = max(float(np.max(p205)), 1e-30)
    for i in indices:
        lvl185 = "HIGH" if p185[i] > 0.25 * p185_max else "LOW"
        lvl205 = "HIGH" if p205[i] > 0.25 * p205_max else "LOW"
        print(f"{times[i]:8.3f}  {lvl185:>12}  {lvl205:>12}  {classifications[i]:>8}")


def _save_plots(capture_path: str, pc1: FileAnalysis, pc2: FileAnalysis) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audio, sr, _ = load_mono_wav(capture_path)
        _, _, zxx = stft(
            audio,
            fs=sr,
            window="hann",
            nperseg=STFT_NPERSEG,
            noverlap=STFT_NOVERLAP,
            boundary=None,
            padded=False,
        )
        freqs = np.fft.rfftfreq(STFT_NPERSEG, d=1.0 / sr)
        # stft returns its own freq axis; recompute from returned shape
        from scipy.signal import stft as _stft

        f_stft, t_stft, z = _stft(
            audio, fs=sr, window="hann", nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP
        )
        power_db = 10.0 * np.log10(np.abs(z) ** 2 + 1e-30)
        band = (f_stft >= ULTRASONIC_LOW) & (f_stft <= ULTRASONIC_HIGH)

        base = os.path.splitext(os.path.basename(capture_path))[0]
        spec_path = os.path.join(FORENSIC_DIR, f"{base}_spectrogram_18_21k.png")
        tl_path = os.path.join(FORENSIC_DIR, f"{base}_carrier_timeline.png")

        fig, ax = plt.subplots(figsize=(12, 5))
        pcm = ax.pcolormesh(
            t_stft,
            f_stft[band],
            power_db[band, :],
            shading="gouraud",
            cmap="magma",
        )
        ax.axhline(FREQ_0, color="cyan", ls="--", lw=0.8, label="18.5 kHz ref")
        ax.axhline(FREQ_1, color="lime", ls="--", lw=0.8, label="20.5 kHz ref")
        ax.set_ylim(ULTRASONIC_LOW, ULTRASONIC_HIGH)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Ultrasonic spectrogram (18-21 kHz): {os.path.basename(capture_path)}")
        fig.colorbar(pcm, ax=ax, label="Power (dB)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(spec_path, dpi=120)
        plt.close(fig)
        saved.append(spec_path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(pc2.frame_times, pc2.power_18500, label="18.5 kHz band power", color="cyan")
        ax.plot(pc2.frame_times, pc2.power_20500, label="20.5 kHz band power", color="lime")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band power")
        ax.set_title("Carrier band power vs time (PC2 capture)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(tl_path, dpi=120)
        plt.close(fig)
        saved.append(tl_path)
    except Exception as exc:
        print(f"  (plot generation skipped: {exc})")
    return saved


def _final_diagnosis(pc1: FileAnalysis, pc2: FileAnalysis) -> tuple[str, str]:
    score = pc2.bfsk_score
    if score >= 6 and pc2.evidence["Alternation"] == "PASS":
        return "CONFIRMED PHYSICAL BFSK", "Multiple independent evidence categories pass including alternation."
    if score >= 4 and pc2.transition_count >= 2:
        return "LIKELY BFSK BUT LOW SNR", "Some BFSK structure present but SNR or timing confidence is marginal."
    if pc2.ultrasonic_ratio > 0.01 and pc2.bfsk_score < 4:
        return "ULTRASONIC ACTIVITY PRESENT BUT NOT BFSK", (
            "Energy exists in 18-21 kHz but lacks reliable alternation at ~50 ms between 18.5/20.5 kHz."
        )
    if pc2.ultrasonic_ratio < 0.001:
        return "HARDWARE PATH NOT CONFIRMED", "Ultrasonic energy negligible; speaker->mic path may not have carried signal."
    return "INSUFFICIENT DATA", "Captured evidence is ambiguous; more synchronized capture may be needed."


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical BFSK capture forensic analysis")
    parser.add_argument("--capture", help="PC2 capture WAV (default: newest forensic_captures/pc2_capture_*.wav)")
    parser.add_argument("--pc1-wav", help="PC1 BFSK WAV (default: newest generated_payloads/*.wav)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    capture = args.capture or _newest_file(os.path.join(FORENSIC_DIR, "pc2_capture_*.wav"))
    pc1_wav = args.pc1_wav or _newest_file(os.path.join(PAYLOAD_DIR, "*.wav"))

    if not capture or not os.path.isfile(capture):
        print("ERROR: No PC2 capture found in forensic_captures/pc2_capture_*.wav")
        return 1
    if not pc1_wav or not os.path.isfile(pc1_wav):
        print("ERROR: No PC1 BFSK WAV found in generated_payloads/*.wav")
        return 1

    print("=" * 60)
    print("TASK 1 â€” CAPTURE LOCATION")
    print("=" * 60)
    print("PC2 CAPTURE:")
    cap_info = wav_info(capture)
    _print_wav_info(cap_info)
    print()
    print("PC1 BFSK WAV:")
    pc1_info = wav_info(pc1_wav)
    _print_wav_info(pc1_info)
    print()

    pc1 = analyze_file(pc1_wav, "PC1")
    pc2 = analyze_file(capture, "PC2")

    print("=" * 60)
    print("TASK 2 â€” WHOLE-FILE ULTRASONIC ANALYSIS (PC2 CAPTURE)")
    print("=" * 60)
    print(f"Total energy              : {pc2.total_energy:.6e}")
    print(f"Ultrasonic energy (18-21k): {pc2.ultrasonic_energy:.6e}")
    print(f"Ultrasonic/total ratio    : {pc2.ultrasonic_ratio:.6e}")
    print(f"Global peak frequency     : {pc2.global_peak_freq:.2f} Hz")
    print(f"Global peak power         : {pc2.global_peak_power:.6e}")
    print(f"Noise floor estimate      : {pc2.noise_floor:.6e}")
    print()
    _print_band("18.5 kHz band (18500 +/- 100 Hz)", pc2.band_18500)
    print()
    _print_band("20.5 kHz band (20500 +/- 100 Hz)", pc2.band_20500)
    print()

    print("=" * 60)
    print("TASK 3 â€” TIME-RESOLVED ANALYSIS (PC2 CAPTURE)")
    print("=" * 60)
    threshold, thresh_expl = _derive_snr_threshold(pc2.snr_18500, pc2.snr_20500, pc2.power_18500 + pc2.power_20500)
    print(f"SNR classification threshold: {threshold:.2f} dB")
    print(f"Threshold rationale: {thresh_expl}")
    print()
    _print_timeline(pc2.frame_times, pc2.power_18500, pc2.power_20500, pc2.classifications)
    print()

    print("=" * 60)
    print("TASK 4 â€” BFSK TRANSITION ANALYSIS (PC2 CAPTURE)")
    print("=" * 60)
    print(f"Estimated BFSK symbol duration : {pc2.symbol_duration_ms:.2f} ms (expected {BIT_DURATION_MS:.0f} ms)")
    print(f"Estimated carrier transitions  : {pc2.transition_count}")
    print(f"Timing confidence              : {pc2.timing_confidence:.3f}")
    if pc2.transition_intervals_ms:
        arr = np.array(pc2.transition_intervals_ms)
        print(f"Transition intervals (ms)      : min={arr.min():.1f} median={np.median(arr):.1f} max={arr.max():.1f}")
    print()

    print("=" * 60)
    print("TASK 5 â€” CARRIER CLASSIFICATION SUMMARY (PC2)")
    print("=" * 60)
    n185 = pc2.classifications.count("18500")
    n205 = pc2.classifications.count("20500")
    nnone = pc2.classifications.count("NONE")
    print(f"Frames classified 18500 : {n185}")
    print(f"Frames classified 20500 : {n205}")
    print(f"Frames classified NONE  : {nnone}")
    print()

    print("=" * 60)
    print("TASK 6 â€” PLAYBACK WINDOW (PC2 CAPTURE)")
    print("=" * 60)
    print(f"Signal activity start : {pc2.activity_start_sec:.3f} s")
    print(f"Signal activity end   : {pc2.activity_end_sec:.3f} s")
    print(f"Signal duration       : {pc2.activity_duration_sec:.3f} s")
    print(f"PC1 WAV duration      : {pc1.info.duration_sec:.3f} s (expected ~7.8 s)")
  # flag if activity outside expected window
    if pc2.activity_duration_sec > 0 and abs(pc2.activity_duration_sec - pc1.info.duration_sec) > 5.0:
        print("NOTE: Activity duration differs substantially from PC1 WAV â€” may include background.")
    print()

    print("=" * 60)
    print("TASK 7 â€” BFSK EVIDENCE SCORE (PC2 CAPTURE)")
    print("=" * 60)
    for k, v in pc2.evidence.items():
        print(f"  {k:<24}: {v}")
    print(f"\nBFSK EVIDENCE SCORE: {pc2.bfsk_score} / 7")
    print()

    print("=" * 60)
    print("TASK 8 â€” PC1 WAV vs PC2 CAPTURE COMPARISON")
    print("=" * 60)
    print(f"{'METRIC':<28} {'PC1 WAV':>18} {'PC2 CAPTURE':>18}")
    print("-" * 66)
    rows = [
        ("18500 Hz peak power", pc1.band_18500.peak_power, pc2.band_18500.peak_power),
        ("20500 Hz peak power", pc1.band_20500.peak_power, pc2.band_20500.peak_power),
        ("18500 Hz SNR (dB)", pc1.band_18500.snr_db, pc2.band_18500.snr_db),
        ("20500 Hz SNR (dB)", pc1.band_20500.snr_db, pc2.band_20500.snr_db),
        ("Symbol duration (ms)", pc1.symbol_duration_ms, pc2.symbol_duration_ms),
        ("Signal duration (s)", pc1.activity_duration_sec, pc2.activity_duration_sec),
        ("Transitions", pc1.transition_count, pc2.transition_count),
        ("Ultrasonic ratio", pc1.ultrasonic_ratio, pc2.ultrasonic_ratio),
        ("Observed carrier 0 (Hz)", pc1.band_18500.peak_freq_hz, pc2.band_18500.peak_freq_hz),
        ("Observed carrier 1 (Hz)", pc1.band_20500.peak_freq_hz, pc2.band_20500.peak_freq_hz),
    ]
    for name, v1, v2 in rows:
        print(f"{name:<28} {v1:>18.6g} {v2:>18.6g}")
    print()

    if not args.no_plots:
        print("=" * 60)
        print("TASK 11 â€” VISUALIZATION")
        print("=" * 60)
        saved = _save_plots(capture, pc1, pc2)
        for p in saved:
            print(f"  saved: {p}")
        print()

    diag, why = _final_diagnosis(pc1, pc2)
    strength = "STRONG" if pc2.bfsk_score >= 6 else "MODERATE" if pc2.bfsk_score >= 4 else "WEAK" if pc2.bfsk_score >= 2 else "NONE"

    print("=" * 60)
    print("PHYSICAL BFSK FORENSIC ANALYSIS")
    print("=" * 60)
    print(f"PC1 WAV     : {pc1.info.path}")
    print(f"PC2 CAPTURE : {pc2.info.path}")
    print(f"Sample rate : {pc2.info.sample_rate} Hz")
    print()
    print("Expected carriers:")
    print(f"  {FREQ_0:.0f} Hz")
    print(f"  {FREQ_1:.0f} Hz")
    print()
    print(f"Observed carrier 0 : {pc2.band_18500.peak_freq_hz:.2f} Hz")
    print(f"Observed carrier 1 : {pc2.band_20500.peak_freq_hz:.2f} Hz")
    print()
    print(f"18.5 kHz evidence          : {pc2.evidence['18.5 kHz carrier']}")
    print(f"20.5 kHz evidence          : {pc2.evidence['20.5 kHz carrier']}")
    print(f"Alternating carrier evidence: {pc2.evidence['Alternation']}")
    print(f"50 ms timing evidence      : {pc2.evidence['50 ms timing']}")
    print(f"Playback-window correlation: {pc2.evidence['Activity duration']}")
    print(f"SNR                        : {pc2.evidence['SNR']}")
    print()
    print(f"Estimated symbol duration  : {pc2.symbol_duration_ms:.2f} ms")
    print(f"Estimated transitions      : {pc2.transition_count}")
    print()
    print(f"BFSK EVIDENCE              : {strength}")
    print()
    print(f"FINAL DIAGNOSIS            : {diag}")
    print(f"Reason                     : {why}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

