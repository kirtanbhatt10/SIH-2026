"""
recording_quality.py — Reject dead captures before they become "evidence"
==========================================================================
AI 1 (DSP / Signal Processing Lead)

WHY THIS EXISTS
---------------
On 15 Aug 2026 we discovered that nearly every `rx_*.wav` in the physical test
set contained no audio at all:

    rx_18000.wav  peak = 6.10e-05  =  2 LSB, only 3 unique sample values / 12 s
    rx_17000.wav  peak = 6.10e-05  =  2 LSB
    baseline_quiet.wav peak = 3.05e-05 = 1 LSB
    clap.wav      peak = 2.46e-01  = 8064 LSB   <-- mic itself is fine

The transmitted tone was absent, not merely quiet: in rx_4000_v2.wav the 4 kHz
bin sat at 4.5e-09 while the file's own maximum was at 375 Hz.

Analysing such a file produces garbage that *looks* impressive — a near-zero
noise floor makes 20*log10(peak/floor) explode, so we saw "SNR = 108 dB" from a
file containing three distinct sample values. Charter §13 forbids presenting
unmeasured values as results; this module makes that failure mode loud.

USAGE
-----
    from dsp.recording_quality import check_recording, LSB

    report = check_recording(audio, sample_rate=48000, expected_hz=18000)
    if not report["ok"]:
        print(report["summary"])        # explains exactly what failed

CLI (check files already on disk):
    python -m dsp.recording_quality *.wav
    python -m dsp.recording_quality rx_18000.wav --expected 18000
"""

from __future__ import annotations

import sys

import numpy as np

# Smallest non-zero step an int16 converter can emit. Even when a device hands
# us float32, the underlying ADC is usually 16-bit, so real audio lands on
# multiples of this value.
LSB = 1.0 / 32768.0                 # 3.0518e-05

# ── Thresholds ───────────────────────────────────────────────────────────
MIN_PEAK_LSB = 50.0        # below this the capture is effectively dead
WARN_PEAK_LSB = 300.0      # usable but very quiet — raise input gain
MIN_UNIQUE_VALUES = 64     # 3-5 unique values over 12 s means no real signal
CLIP_FRACTION = 1e-4       # >0.01% of samples at full scale = clipping
# A transmitted tone must clear its local floor by this margin.
#
# Calibrated empirically, not guessed: the peak/median ratio of PURE WHITE NOISE
# in a +/-200 Hz band measured over 40 trials was mean 9.4 dB, max 11.0 dB,
# p99 10.9 dB. A 10 dB threshold therefore fires on pure noise ~20% of the time.
# 15 dB gave 0% false positives across the same trials.
MIN_TONE_SNR_DB = 15.0


def _tone_snr_db(audio: np.ndarray, sample_rate: int, expected_hz: float) -> tuple:
    """
    Measure how far the expected tone rises above its own neighbourhood.

    Local floor is the MEDIAN over +/-2 kHz excluding +/-300 Hz around the tone,
    so the tone cannot inflate the floor it is compared against.
    """
    x = np.asarray(audio, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) / np.sum(w)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    eps = 1e-20

    band = (freqs >= expected_hz - 200) & (freqs <= expected_hz + 200)
    if not np.any(band):
        return None, None

    k = int(np.argmax(spec[band]))
    peak_hz = float(freqs[band][k])
    peak_mag = float(spec[band][k])

    near = (freqs >= expected_hz - 2000) & (freqs <= expected_hz + 2000)
    excl = (freqs >= expected_hz - 300) & (freqs <= expected_hz + 300)
    local = spec[near & ~excl]
    floor = float(np.median(local)) if local.size else float(np.median(spec))

    snr_db = float(20 * np.log10(peak_mag / (floor + eps) + eps))
    return peak_hz, snr_db


def check_recording(audio, sample_rate: int = 48000, expected_hz: float = None) -> dict:
    """
    Validate a recording. Returns a report dict; `report["ok"]` is the verdict.

    Parameters
    ----------
    audio : array-like
        Samples, any dtype. Integer input is scaled to [-1, 1].
    sample_rate : int
    expected_hz : float, optional
        If given, also verify the transmitted tone is actually present.
    """
    a = np.asarray(audio)

    # Normalize integer PCM to float [-1, 1]
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        a = a.astype(np.float64) / float(max(abs(info.min), info.max))
    else:
        a = a.astype(np.float64)

    if a.ndim > 1:
        a = a.mean(axis=1)

    n = len(a)
    problems, warnings = [], []

    peak = float(np.max(np.abs(a))) if n else 0.0
    rms = float(np.sqrt(np.mean(a**2))) if n else 0.0
    peak_lsb = peak / LSB
    unique = int(len(np.unique(a)))
    clipped = int(np.count_nonzero(np.abs(a) >= 0.999))

    # ── 1. Dead capture ──────────────────────────────────────────────────
    if peak_lsb < MIN_PEAK_LSB:
        problems.append(
            f"DEAD CAPTURE: peak {peak:.3e} = {peak_lsb:.1f} LSB "
            f"(need >= {MIN_PEAK_LSB:.0f}). The microphone delivered essentially "
            f"nothing. Check the input device, OS mic permissions, and gain."
        )
    elif peak_lsb < WARN_PEAK_LSB:
        warnings.append(
            f"Very quiet: peak {peak_lsb:.0f} LSB. Usable but raise input gain."
        )

    # ── 2. Quantisation-only content ─────────────────────────────────────
    if unique < MIN_UNIQUE_VALUES:
        problems.append(
            f"ONLY {unique} UNIQUE SAMPLE VALUES in {n} samples. Real audio has "
            f"thousands. This file is quantisation noise, not a recording."
        )

    # ── 3. Digital silence ───────────────────────────────────────────────
    nonzero = np.count_nonzero(a)
    if nonzero == 0:
        problems.append("File is entirely zeros.")
    elif nonzero / max(1, n) < 0.01:
        problems.append(f"{100*nonzero/n:.2f}% of samples are non-zero — near-total silence.")

    # ── 4. Clipping ──────────────────────────────────────────────────────
    if clipped / max(1, n) > CLIP_FRACTION:
        problems.append(
            f"CLIPPING: {clipped} samples ({100*clipped/n:.2f}%) at full scale. "
            f"Lower the input gain; harmonics will contaminate the spectrum."
        )

    # ── 5. Is the transmitted tone actually there? ───────────────────────
    tone = {}
    if expected_hz:
        peak_hz, snr_db = _tone_snr_db(a, sample_rate, expected_hz)
        if peak_hz is not None:
            tone = {
                "expected_hz": float(expected_hz),
                "measured_peak_hz": round(peak_hz, 1),
                "frequency_error_hz": round(peak_hz - expected_hz, 1),
                "tone_snr_db": round(snr_db, 2),
                "tone_present": bool(snr_db >= MIN_TONE_SNR_DB),
            }
            if snr_db < MIN_TONE_SNR_DB:
                problems.append(
                    f"TONE ABSENT: nothing at {expected_hz} Hz "
                    f"(SNR {snr_db:.1f} dB < {MIN_TONE_SNR_DB} dB). "
                    f"The transmitter may not have played, or the mic cannot "
                    f"reach this frequency. Either is a finding — but do not "
                    f"report this file as a successful reception."
                )

    ok = not problems
    if ok:
        summary = f"OK — peak {peak_lsb:.0f} LSB, {unique} unique values"
        if tone.get("tone_present"):
            summary += f", tone at {tone['measured_peak_hz']} Hz @ {tone['tone_snr_db']} dB SNR"
    else:
        summary = "REJECTED — " + " | ".join(problems)

    return {
        "ok": ok,
        "summary": summary,
        "problems": problems,
        "warnings": warnings,
        "peak": peak,
        "peak_lsb": round(peak_lsb, 1),
        "rms": rms,
        "rms_dbfs": round(float(20 * np.log10(rms + 1e-20)), 2),
        "unique_values": unique,
        "num_samples": n,
        "duration_sec": round(n / sample_rate, 3) if sample_rate else None,
        "clipped_samples": clipped,
        "tone": tone,
    }


def _cli(argv):
    """Check WAV files already on disk."""
    from scipy.io import wavfile

    args = [a for a in argv if not a.startswith("--")]
    expected = None
    for i, a in enumerate(argv):
        if a == "--expected" and i + 1 < len(argv):
            expected = float(argv[i + 1])
            args = [x for x in args if x != argv[i + 1]]

    if not args:
        print(__doc__)
        return 1

    print(f"{'file':<28s} {'peak(LSB)':>10s} {'uniq':>7s}  verdict")
    print("-" * 78)

    bad = 0
    for path in args:
        try:
            sr, data = wavfile.read(path)
        except Exception as e:
            print(f"{path:<28s} {'-':>10s} {'-':>7s}  UNREADABLE: {e}")
            bad += 1
            continue

        exp = expected
        if exp is None:                       # infer from filename, e.g. rx_18000.wav
            import re
            m = re.search(r"(\d{3,5})", path)
            if m and "baseline" not in path.lower():
                v = int(m.group(1))
                if 100 <= v <= 24000:
                    exp = v

        r = check_recording(data, sr, exp)
        if not r["ok"]:
            bad += 1
        verdict = "OK" if r["ok"] else "REJECTED"
        print(f"{path:<28s} {r['peak_lsb']:>10.1f} {r['unique_values']:>7d}  {verdict}")
        for p in r["problems"]:
            print(f"{'':<28s} {'':>10s} {'':>7s}    - {p}")
        for w in r["warnings"]:
            print(f"{'':<28s} {'':>10s} {'':>7s}    ! {w}")

    print("-" * 78)
    print(f"{len(args) - bad} passed, {bad} rejected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
