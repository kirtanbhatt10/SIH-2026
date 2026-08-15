"""
make_evidence.py — Curate committable evidence from local recordings
=====================================================================
AI 1 (DSP / Signal Processing Lead)

PROBLEM
-------
`.gitignore` excludes *.wav and *.png, so every physical experiment we ran
(voiceband, 15-17.5 kHz, 18-19 kHz) left NO record in the repository.
The charter (§3, §10) requires us to *report experimentally measured limits*,
and §13 forbids presenting unmeasured values as results.

WHAT THIS DOES
--------------
Takes the raw recordings sitting on your local disk and produces a small,
curated, committable evidence set under docs/evidence/:

    docs/evidence/
    ├── audio/                       trimmed mono 16-bit WAV excerpts (<= 2 s)
    ├── spectra/                     one PNG per recording (spectrum + spectrogram)
    ├── measurements.json            machine-readable measured numbers
    ├── measurements.csv             same, for the PPT lead / spreadsheets
    └── MANIFEST.md                  human-readable table + SHA-256 provenance

Every number written here is MEASURED from a real recording. Nothing is
synthesized, assumed, or rounded up for presentation.

USAGE
-----
    # analyse every known recording found in the current directory
    python scripts/make_evidence.py

    # point at wherever your recordings actually live
    python scripts/make_evidence.py --input-dir C:/SIH-2026/recordings

    # only the decisive few (recommended for what you actually commit)
    python scripts/make_evidence.py --curated

    # see what would happen, write nothing
    python scripts/make_evidence.py --curated --dry-run

Then:
    git add -f docs/evidence
    git commit -m "docs: add measured hardware feasibility evidence"

(-f is not needed if you have the updated .gitignore with the
 docs/evidence/ allow-list, but it is harmless and safe.)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks


# ── Output budget ────────────────────────────────────────────────────────
# Keeps the repo small. A 2 s mono int16 clip at 48 kHz is ~188 KB.
MAX_CLIP_SEC = 2.0
PNG_DPI = 100
WARN_TOTAL_MB = 15.0

EVIDENCE_DIR = os.path.join("docs", "evidence")


# ── Which recordings matter ──────────────────────────────────────────────
# expected_hz = the tone we transmitted. None = silent baseline (control).
KNOWN_RECORDINGS = {
    # Voiceband control — proves the audio chain works at all (charter §3 fallback)
    "baseline_voiceband.wav": (None,  "voiceband", "Silent baseline (voiceband session)"),
    "rx_1000_v2.wav":         (1000,  "voiceband", "1 kHz tone through speaker->air->mic"),
    "rx_2000_v2.wav":         (2000,  "voiceband", "2 kHz tone through speaker->air->mic"),
    "rx_3000_v2.wav":         (3000,  "voiceband", "3 kHz tone through speaker->air->mic"),
    "rx_4000_v2.wav":         (4000,  "voiceband", "4 kHz tone through speaker->air->mic"),
    "rx_5000_v2.wav":         (5000,  "voiceband", "5 kHz tone through speaker->air->mic"),

    # Mid sweep — where does the hardware start to roll off?
    "baseline_control.wav":   (None,  "control",   "Silent baseline (control session)"),
    "rx_10000.wav":           (10000, "control",   "10 kHz tone"),
    "rx_12000.wav":           (12000, "control",   "12 kHz tone"),
    "rx_14000.wav":           (14000, "control",   "14 kHz tone"),

    # Upper edge — the actual feasibility question
    "baseline_lower.wav":     (None,  "upper",     "Silent baseline (15-17.5 kHz session)"),
    "rx_15000.wav":           (15000, "upper",     "15 kHz tone"),
    "rx_16000.wav":           (16000, "upper",     "16 kHz tone"),
    "rx_17000.wav":           (17000, "upper",     "17 kHz tone"),
    "rx_17500.wav":           (17500, "upper",     "17.5 kHz tone"),

    # Ultrasonic attempt — expected to fail or be marginal; that IS the finding
    "baseline_new.wav":       (None,  "ultrasonic", "Silent baseline (ultrasonic session)"),
    "baseline_quiet.wav":     (None,  "ultrasonic", "Silent baseline (quiet room)"),
    "rx_18000.wav":           (18000, "ultrasonic", "18 kHz tone"),
    "rx_18200.wav":           (18200, "ultrasonic", "18.2 kHz tone"),
    "rx_18400.wav":           (18400, "ultrasonic", "18.4 kHz tone"),
    "rx_18600.wav":           (18600, "ultrasonic", "18.6 kHz tone"),
    "rx_18800.wav":           (18800, "ultrasonic", "18.8 kHz tone"),
    "rx_19000.wav":           (19000, "ultrasonic", "19 kHz tone"),
}

# The minimum set that defends every claim we need to make.
CURATED = [
    "baseline_quiet.wav",   # noise floor
    "rx_4000_v2.wav",       # chain works (voiceband proof)
    "rx_15000.wav",         # still strong
    "rx_17000.wav",         # near the edge
    "rx_18000.wav",         # ultrasonic attempt
    "rx_19000.wav",         # ultrasonic limit
]

# A tone counts as "detected" only if it clears the local floor by this margin.
DETECTION_SNR_DB = 10.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_wav(path):
    """Load a WAV as float64 mono in [-1, 1], regardless of source dtype."""
    sr, raw = wavfile.read(path)

    if raw.dtype == np.int16:
        data = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        data = raw.astype(np.float64) / 2147483648.0
    elif raw.dtype == np.uint8:
        data = (raw.astype(np.float64) - 128.0) / 128.0
    else:
        data = raw.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data


def measure(sr, audio, expected_hz):
    """
    Measure the spectrum. Returns MEASURED values only.

    peak_snr_db is peak magnitude over the MEDIAN of a local band around it,
    i.e. how far the tone rises above its own neighbourhood noise floor.
    Median (not mean) so the peak itself doesn't inflate the floor.
    """
    x = audio - np.mean(audio)                      # remove DC
    n = len(x)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(x * window)) / np.sum(window)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    eps = 1e-12
    broadband_floor = float(np.median(spectrum))

    result = {
        "sample_rate_hz": int(sr),
        "duration_sec": round(n / sr, 3),
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x**2)) + eps)), 2),
        "broadband_noise_floor": float(broadband_floor),
        "expected_hz": expected_hz,
    }

    if expected_hz is None:
        # Baseline: report the loudest thing present, so we know the room.
        idx = int(np.argmax(spectrum))
        result.update({
            "measured_peak_hz": round(float(freqs[idx]), 1),
            "measured_peak_snr_db": round(
                float(20 * np.log10(spectrum[idx] / (broadband_floor + eps) + eps)), 2),
            "detected": None,      # not applicable to a baseline
            "note": "baseline / no tone transmitted",
        })
        return result, freqs, spectrum

    # Search a +/- 200 Hz window around the tone we actually transmitted.
    search = (freqs >= expected_hz - 200) & (freqs <= expected_hz + 200)
    if not np.any(search):
        result.update({"detected": False, "note": "expected frequency out of range"})
        return result, freqs, spectrum

    band_f, band_s = freqs[search], spectrum[search]
    k = int(np.argmax(band_s))
    peak_hz, peak_mag = float(band_f[k]), float(band_s[k])

    # Local floor: +/- 2 kHz around the tone, excluding +/- 300 Hz of the tone itself.
    near = (freqs >= expected_hz - 2000) & (freqs <= expected_hz + 2000)
    exclude = (freqs >= expected_hz - 300) & (freqs <= expected_hz + 300)
    local = spectrum[near & ~exclude]
    local_floor = float(np.median(local)) if local.size else broadband_floor

    snr_db = float(20 * np.log10(peak_mag / (local_floor + eps) + eps))

    result.update({
        "measured_peak_hz": round(peak_hz, 1),
        "frequency_error_hz": round(peak_hz - expected_hz, 1),
        "peak_magnitude": float(peak_mag),
        "local_noise_floor": float(local_floor),
        "measured_peak_snr_db": round(snr_db, 2),
        "detected": bool(snr_db >= DETECTION_SNR_DB),
        "detection_threshold_db": DETECTION_SNR_DB,
    })
    return result, freqs, spectrum


def save_clip(path, sr, audio, out_path):
    """Trim to MAX_CLIP_SEC and save as mono 16-bit to keep the repo small."""
    max_samples = int(MAX_CLIP_SEC * sr)
    clip = audio[:max_samples] if len(audio) > max_samples else audio

    peak = np.max(np.abs(clip))
    if peak > 0:
        clip = clip / peak * 0.95          # normalize for consistent playback

    wavfile.write(out_path, sr, (clip * 32767).astype(np.int16))
    return os.path.getsize(out_path)


def save_plot(name, sr, audio, freqs, spectrum, meas, out_path):
    """Spectrum (top) + spectrogram (bottom), annotated with measured values."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5))
    eps = 1e-12

    lo, hi = 0, min(24000, sr / 2)
    m = (freqs >= lo) & (freqs <= hi)
    ax1.plot(freqs[m] / 1000.0, 20 * np.log10(spectrum[m] + eps), linewidth=0.7)

    exp = meas.get("expected_hz")
    if exp:
        ax1.axvline(exp / 1000.0, color="tab:red", linestyle="--", linewidth=1,
                    label=f"transmitted {exp/1000:.1f} kHz")
    pk = meas.get("measured_peak_hz")
    if pk:
        ax1.axvline(pk / 1000.0, color="tab:green", linestyle=":", linewidth=1,
                    label=f"measured peak {pk/1000:.2f} kHz")

    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"{name} — measured SNR {meas.get('measured_peak_snr_db')} dB")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    nper = 1024
    if len(audio) >= nper:
        ax2.specgram(audio, NFFT=nper, Fs=sr, noverlap=nper // 2, cmap="viridis")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_ylim(0, min(24000, sr / 2))
        if exp:
            ax2.axhline(exp, color="red", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=PNG_DPI)
    plt.close(fig)
    return os.path.getsize(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=".", help="where your recordings live")
    ap.add_argument("--output-dir", default=EVIDENCE_DIR)
    ap.add_argument("--curated", action="store_true",
                    help="only the decisive recordings (recommended for committing)")
    ap.add_argument("--dry-run", action="store_true", help="analyse but write nothing")
    args = ap.parse_args()

    wanted = CURATED if args.curated else list(KNOWN_RECORDINGS)

    found = [(f, *KNOWN_RECORDINGS[f])
             for f in wanted
             if os.path.exists(os.path.join(args.input_dir, f))]
    missing = [f for f in wanted
               if not os.path.exists(os.path.join(args.input_dir, f))]

    print("=" * 74)
    print("  EVIDENCE BUILDER — AI 1 / DSP")
    print("=" * 74)
    print(f"  input      : {os.path.abspath(args.input_dir)}")
    print(f"  mode       : {'curated' if args.curated else 'all known'}")
    print(f"  found      : {len(found)} / {len(wanted)}")
    if missing:
        print(f"  missing    : {', '.join(missing)}")
    if args.dry_run:
        print("  DRY RUN — nothing will be written")
    print()

    if not found:
        print("  No recordings found.")
        print("  Point --input-dir at the folder holding your .wav files, e.g.")
        print("      python scripts/make_evidence.py --curated --input-dir C:/SIH-2026")
        return 1

    audio_dir = os.path.join(args.output_dir, "audio")
    spec_dir = os.path.join(args.output_dir, "spectra")
    if not args.dry_run:
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)
        os.makedirs("docs", exist_ok=True)

    rows, total_bytes = [], 0

    for fname, expected, session, desc in found:
        src = os.path.join(args.input_dir, fname)
        sr, audio = load_wav(src)
        meas, freqs, spectrum = measure(sr, audio, expected)

        stem = os.path.splitext(fname)[0]
        row = {
            "recording": fname,
            "session": session,
            "description": desc,
            "source_sha256": sha256(src),
            **meas,
        }

        if not args.dry_run:
            wav_out = os.path.join(audio_dir, f"{stem}.wav")
            png_out = os.path.join(spec_dir, f"{stem}.png")
            total_bytes += save_clip(src, sr, audio, wav_out)
            total_bytes += save_plot(stem, sr, audio, freqs, spectrum, meas, png_out)
            row["clip"] = os.path.relpath(wav_out, args.output_dir).replace("\\", "/")
            row["plot"] = os.path.relpath(png_out, args.output_dir).replace("\\", "/")

        rows.append(row)

        det = row.get("detected")
        mark = "----" if det is None else ("PASS" if det else "FAIL")
        exp_s = "baseline" if expected is None else f"{expected:>5d} Hz"
        print(f"  [{mark}] {fname:<24s} {exp_s}  "
              f"SNR {row.get('measured_peak_snr_db'):>7.2f} dB  "
              f"peak {row.get('measured_peak_hz')} Hz")

    if args.dry_run:
        print("\n  Dry run complete — no files written.")
        return 0

    # ── measurements.json ────────────────────────────────────────────────
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/make_evidence.py (AI 1 — DSP lead)",
        "detection_threshold_db": DETECTION_SNR_DB,
        "methodology": (
            "Mono, DC-removed, Hann-windowed rFFT normalized by window sum. "
            "Peak searched within +/-200 Hz of the transmitted tone. Local noise "
            "floor = median magnitude over +/-2 kHz around the tone, excluding "
            "+/-300 Hz around it. SNR = 20*log10(peak / local floor). A tone is "
            f"'detected' only at SNR >= {DETECTION_SNR_DB} dB."
        ),
        "caveat": (
            "Values are specific to the laptop mic/speaker, room, and distance used. "
            "They are not general hardware specifications."
        ),
        "measurements": rows,
    }
    with open(os.path.join(args.output_dir, "measurements.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ── measurements.csv ─────────────────────────────────────────────────
    cols = ["recording", "session", "expected_hz", "measured_peak_hz",
            "frequency_error_hz", "measured_peak_snr_db", "detected",
            "rms_dbfs", "duration_sec", "sample_rate_hz"]
    with open(os.path.join(args.output_dir, "measurements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── MANIFEST.md ──────────────────────────────────────────────────────
    tones = [r for r in rows if r.get("expected_hz")]
    ok = [r for r in tones if r.get("detected")]
    ceiling = max((r["expected_hz"] for r in ok), default=None)

    with open(os.path.join(args.output_dir, "MANIFEST.md"), "w") as f:
        f.write("# Measured Evidence — AI 1 (DSP / Signal Processing)\n\n")
        f.write(f"Generated: {payload['generated_utc']}  \n")
        f.write("Generator: `scripts/make_evidence.py`\n\n")
        f.write("> Every number here is measured from a real recording made through\n")
        f.write("> speaker -> air -> microphone. Nothing is synthetic or estimated.\n\n")

        f.write("## Headline finding\n\n")
        if ceiling:
            f.write(f"Highest transmitted tone recovered at >= {DETECTION_SNR_DB} dB SNR: "
                    f"**{ceiling} Hz**.\n\n")
        else:
            f.write("No transmitted tone cleared the detection threshold.\n\n")

        f.write("## Measurements\n\n")
        f.write("| Recording | Session | Sent | Measured peak | SNR (dB) | Detected |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            sent = "baseline" if r.get("expected_hz") is None else f"{r['expected_hz']} Hz"
            det = "—" if r.get("detected") is None else ("yes" if r["detected"] else "**no**")
            f.write(f"| `{r['recording']}` | {r['session']} | {sent} | "
                    f"{r.get('measured_peak_hz')} Hz | {r.get('measured_peak_snr_db')} | {det} |\n")

        f.write("\n## Method\n\n" + payload["methodology"] + "\n")
        f.write("\n## Caveat\n\n" + payload["caveat"] + "\n")

        f.write("\n## Provenance (SHA-256 of source recordings)\n\n")
        f.write("| Recording | SHA-256 |\n|---|---|\n")
        for r in rows:
            f.write(f"| `{r['recording']}` | `{r['source_sha256'][:32]}...` |\n")

    # ── docs/hardware-feasibility.md (charter §10 deliverable) ──────────
    tones_sorted = sorted(tones, key=lambda r: r["expected_hz"])
    ok_sorted = [r for r in tones_sorted if r.get("detected")]
    fail_sorted = [r for r in tones_sorted if not r.get("detected")]

    with open(os.path.join("docs", "hardware-feasibility.md"), "w") as f:
        f.write("# Hardware Feasibility Report\n\n")
        f.write("**Owner:** AI 1 (DSP / Signal Processing)  \n")
        f.write(f"**Generated:** {payload['generated_utc']}  \n")
        f.write("**Generator:** `scripts/make_evidence.py`\n\n")
        f.write("> Charter §3: *\"Report experimentally measured limits.\"*  \n")
        f.write("> Charter §13: do not present unmeasured values as results.\n\n")
        f.write("Every number below is measured from a real recording made through\n")
        f.write("speaker -> air -> microphone on the project laptop.\n\n---\n\n")

        f.write("## Headline\n\n")
        if ceiling:
            f.write(f"**Highest reliably received tone: {ceiling} Hz** ")
            f.write(f"(>= {DETECTION_SNR_DB} dB SNR).\n\n")
            if ceiling < 18000:
                f.write("This is **below the 18 kHz** lower edge of the intended\n")
                f.write("ultrasonic detection band. The MVP demonstration therefore\n")
                f.write("uses a supported test frequency, and the hardware limitation\n")
                f.write("is documented rather than worked around. Dedicated ultrasonic\n")
                f.write("hardware would be required to validate the 18-21 kHz band.\n\n")
            else:
                f.write("This reaches into the intended 18-21 kHz detection band.\n\n")
        else:
            f.write("**No transmitted tone cleared the detection threshold.**\n")
            f.write("No frequency-response claim is supported by current data.\n\n")

        f.write("## Frequency response\n\n")
        f.write("| Transmitted | Measured peak | Error | SNR (dB) | Received |\n")
        f.write("|---|---|---|---|---|\n")
        for r in tones_sorted:
            f.write(f"| {r['expected_hz']} Hz | {r.get('measured_peak_hz')} Hz | ")
            f.write(f"{r.get('frequency_error_hz')} Hz | {r.get('measured_peak_snr_db')} | ")
            f.write(("yes" if r.get("detected") else "**no**") + " |\n")

        base = [r for r in rows if r.get("expected_hz") is None]
        if base:
            f.write("\n## Noise floor (silent baselines)\n\n")
            f.write("| Recording | Loudest component | RMS (dBFS) |\n|---|---|---|\n")
            for r in base:
                f.write(f"| `{r['recording']}` | {r.get('measured_peak_hz')} Hz | ")
                f.write(f"{r.get('rms_dbfs')} |\n")

        f.write("\n## Method\n\n" + payload["methodology"] + "\n")

        f.write("\n## Limitations\n\n")
        f.write("- Specific to this laptop's microphone and speaker, this room,\n")
        f.write("  and the distance used. Not a general hardware specification.\n")
        f.write("- Single distance and single environment; no repeat trials.\n")
        f.write("- Speaker and microphone limits are not separated: a failure at\n")
        f.write("  a given frequency may be either transmit or receive.\n")
        if fail_sorted:
            lo = min(r["expected_hz"] for r in fail_sorted)
            f.write(f"- Frequencies from {lo} Hz upward were not reliably received.\n")

        f.write("\n## Provenance\n\n| Recording | SHA-256 |\n|---|---|\n")
        for r in rows:
            f.write(f"| `{r['recording']}` | `{r['source_sha256'][:32]}...` |\n")

    print(f"  Wrote docs/hardware-feasibility.md")

    mb = total_bytes / (1024 * 1024)
    print(f"\n  Wrote {len(rows)} recordings -> {args.output_dir}/  ({mb:.1f} MB)")
    if mb > WARN_TOTAL_MB:
        print(f"  WARNING: exceeds {WARN_TOTAL_MB} MB budget. Re-run with --curated.")
    print("\n  Commit with:")
    print("      git add -f docs/evidence")
    print('      git commit -m "docs: add measured hardware feasibility evidence"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
