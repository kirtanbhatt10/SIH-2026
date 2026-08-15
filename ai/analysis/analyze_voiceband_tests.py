"""
Voiceband (1-5 kHz) Physical Audio Chain Analysis
===================================================
Analyzes receiver recordings from the controlled 1-5 kHz experiment.
"""

import os
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile


def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_wav(filepath):
    sr, raw = wavfile.read(filepath)
    if raw.dtype == np.int16:
        data = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        data = raw.astype(np.float64) / 2147483648.0
    else:
        data = raw.astype(np.float64)
    if data.ndim > 1:
        data = data[:, 0]
    return sr, data


# ─────────────────────────────────────────────────────────────
# STEP 1: VERIFY SOURCE FILES
# ─────────────────────────────────────────────────────────────
print("=" * 75)
print("STEP 1: SOURCE WAV VERIFICATION")
print("=" * 75)

sources = [
    "1000hz_control.wav",
    "2000hz_control.wav",
    "3000hz_control.wav",
    "4000hz_control.wav",
    "5000hz_control.wav",
]

for f in sources:
    sr, data = load_wav(f)
    n = len(data)
    window = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    dom_idx = np.argmax(fft_mag)
    print(f"  {f}")
    print(f"    Sample Rate     : {sr} Hz")
    print(f"    Duration        : {len(data)/sr:.2f} s")
    print(f"    Dominant FFT Hz : {freqs[dom_idx]:.1f} Hz")
    print(f"    Peak Amplitude  : {np.max(np.abs(data)):.4f}")
    print()


# ─────────────────────────────────────────────────────────────
# STEP 2: VERIFY RECEIVER RECORDINGS
# ─────────────────────────────────────────────────────────────
print("=" * 75)
print("STEP 2: RECEIVER RECORDING METADATA & SHA256 VERIFICATION")
print("=" * 75)

rx_files = [
    "baseline_voiceband.wav",
    "rx_1000_v2.wav",
    "rx_2000_v2.wav",
    "rx_3000_v2.wav",
    "rx_4000_v2.wav",
    "rx_5000_v2.wav",
]

rec = {}
hashes = set()

for f in rx_files:
    sr, data = load_wav(f)
    h = sha256(f)
    hashes.add(h)
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    rec[f] = {"sr": sr, "data": data}
    print(f"  {f}")
    print(f"    SHA256   : {h}")
    print(f"    SR       : {sr} Hz")
    print(f"    Duration : {len(data)/sr:.2f} s")
    print(f"    Shape    : {data.shape}")
    print(f"    Peak     : {peak:.6f}")
    print(f"    RMS      : {rms:.6f}")
    print()

print(f"  Unique SHA256 Hashes: {len(hashes)} of {len(rx_files)}")
if len(hashes) == len(rx_files):
    print("  [OK] All recordings confirmed independent.\n")
else:
    print("  [FAIL] Some recordings share SHA256 hashes!\n")


# ─────────────────────────────────────────────────────────────
# STEPS 3-6: FFT, BASELINE INCREASE, PROMINENCE, STFT
# ─────────────────────────────────────────────────────────────
print("=" * 75)
print("STEPS 3-6: CARRIER ANALYSIS (FFT + BASELINE + PROMINENCE + STFT)")
print("=" * 75)

base_data = rec["baseline_voiceband.wav"]["data"]
base_sr = rec["baseline_voiceband.wav"]["sr"]
n_base = len(base_data)
w_base = np.hanning(n_base)
base_fft = np.abs(np.fft.rfft(base_data * w_base)) * (2.0 / n_base)
base_freqs = np.fft.rfftfreq(n_base, 1.0 / base_sr)

targets = [
    ("rx_1000_v2.wav", 1000.0),
    ("rx_2000_v2.wav", 2000.0),
    ("rx_3000_v2.wav", 3000.0),
    ("rx_4000_v2.wav", 4000.0),
    ("rx_5000_v2.wav", 5000.0),
]

results = {}

for filename, target_freq in targets:
    data = rec[filename]["data"]
    sr = rec[filename]["sr"]
    n = len(data)
    window = np.hanning(n)
    rx_fft = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
    rx_freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # --- FFT carrier measurement (±20 Hz) ---
    rx_mask = (rx_freqs >= target_freq - 20) & (rx_freqs <= target_freq + 20)
    base_mask = (base_freqs >= target_freq - 20) & (base_freqs <= target_freq + 20)

    rx_peak_mag = np.max(rx_fft[rx_mask]) if np.any(rx_mask) else 1e-12
    rx_peak_idx = np.argmax(rx_fft[rx_mask]) if np.any(rx_mask) else 0
    rx_peak_freq = rx_freqs[rx_mask][rx_peak_idx] if np.any(rx_mask) else target_freq

    base_peak_mag = np.max(base_fft[base_mask]) if np.any(base_mask) else 1e-12

    ratio = rx_peak_mag / max(base_peak_mag, 1e-12)
    rel_db = 20.0 * np.log10(max(ratio, 1e-12))

    # --- Local prominence (±500 Hz, excluding ±100 Hz) ---
    bg_mask = (
        (rx_freqs >= max(100, target_freq - 500))
        & (rx_freqs <= target_freq + 500)
        & (np.abs(rx_freqs - target_freq) > 100)
    )
    bg_median = np.median(rx_fft[bg_mask]) if np.any(bg_mask) else 1e-12
    prom_ratio = rx_peak_mag / max(bg_median, 1e-12)
    prom_db = 20.0 * np.log10(max(prom_ratio, 1e-12))

    # --- STFT sliding window ---
    n_fft = 8192
    hop = 2048
    num_frames = max(0, (len(data) - n_fft) // hop + 1)
    w_stft = np.hanning(n_fft)
    stft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Also compute baseline STFT for this carrier
    num_base_frames = max(0, (len(base_data) - n_fft) // hop + 1)
    base_frame_energies = []
    for i in range(num_base_frames):
        frame = base_data[i * hop : i * hop + n_fft]
        spec = np.abs(np.fft.rfft(frame * w_stft)) * (2.0 / n_fft)
        t_mask = np.abs(stft_freqs - target_freq) <= 30
        base_frame_energies.append(np.max(spec[t_mask]) if np.any(t_mask) else 0.0)
    base_frame_energies = np.array(base_frame_energies)
    base_stft_max = np.max(base_frame_energies) if len(base_frame_energies) > 0 else 0.0
    base_stft_med = np.median(base_frame_energies) if len(base_frame_energies) > 0 else 0.0

    rx_frame_energies = []
    rx_frame_freqs = []
    for i in range(num_frames):
        frame = data[i * hop : i * hop + n_fft]
        spec = np.abs(np.fft.rfft(frame * w_stft)) * (2.0 / n_fft)
        t_mask = np.abs(stft_freqs - target_freq) <= 30
        if np.any(t_mask):
            rx_frame_energies.append(np.max(spec[t_mask]))
            rx_frame_freqs.append(stft_freqs[t_mask][np.argmax(spec[t_mask])])
        else:
            rx_frame_energies.append(0.0)
            rx_frame_freqs.append(target_freq)

    rx_frame_energies = np.array(rx_frame_energies)
    rx_frame_freqs = np.array(rx_frame_freqs)

    max_stft = np.max(rx_frame_energies) if len(rx_frame_energies) > 0 else 0.0
    median_stft = np.median(rx_frame_energies) if len(rx_frame_energies) > 0 else 0.0

    # Elevated = frame energy > 3x baseline median AND > 3x own median
    threshold = max(3.0 * base_stft_med, 3.0 * median_stft, 1e-7)
    elevated_mask = rx_frame_energies > threshold
    elevated_frames = np.sum(elevated_mask)
    duration_sec = elevated_frames * (hop / sr)
    elevated_freq = np.median(rx_frame_freqs[elevated_mask]) if elevated_frames > 0 else target_freq

    # --- Classification ---
    if ratio >= 5.0 and prom_ratio >= 10.0 and duration_sec >= 1.0:
        classification = "CLEARLY CAPTURED"
    elif ratio >= 5.0 and prom_ratio >= 10.0:
        classification = "CLEARLY CAPTURED"
    elif ratio >= 2.0 or (prom_ratio >= 5.0 and duration_sec >= 0.5):
        classification = "WEAK / UNCERTAIN"
    else:
        classification = "NOT CAPTURED"

    results[filename] = {
        "target_freq": target_freq,
        "obs_freq": rx_peak_freq,
        "rec_mag": rx_peak_mag,
        "base_mag": base_peak_mag,
        "ratio": ratio,
        "rel_db": rel_db,
        "prom_ratio": prom_ratio,
        "prom_db": prom_db,
        "max_stft": max_stft,
        "base_stft_max": base_stft_max,
        "duration_sec": duration_sec,
        "elevated_freq": elevated_freq,
        "classification": classification,
    }

    print(f"\nTarget: {target_freq:.0f} Hz in '{filename}'")
    print(f"  Observed Peak Freq        : {rx_peak_freq:.1f} Hz")
    print(f"  Recorded Carrier Mag      : {rx_peak_mag:14.10f}")
    print(f"  Baseline Carrier Mag      : {base_peak_mag:14.10f}")
    print(f"  Relative FFT Increase     : {ratio:10.2f}x ({rel_db:+7.2f} dB)")
    print(f"  Local BG Median Mag       : {bg_median:14.10f}")
    print(f"  Local Prominence          : {prom_ratio:10.2f}x ({prom_db:+7.2f} dB)")
    print(f"  STFT Max (recording)      : {max_stft:14.10f}")
    print(f"  STFT Max (baseline)       : {base_stft_max:14.10f}")
    print(f"  STFT Elevated Duration    : {duration_sec:.2f} s")
    print(f"  STFT Elevated Obs Freq    : {elevated_freq:.1f} Hz")
    print(f"  Classification            : {classification}")


# ─────────────────────────────────────────────────────────────
# STEP 7: SPECTROGRAMS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 7: SPECTROGRAM IMAGE GENERATION (0.5 - 6 kHz)")
print("=" * 75)

n_fft_spec = 4096
hop_spec = 512

spec_map = {
    "baseline_voiceband.wav": "hardware_voiceband_baseline.png",
    "rx_1000_v2.wav": "hardware_voiceband_1000.png",
    "rx_2000_v2.wav": "hardware_voiceband_2000.png",
    "rx_3000_v2.wav": "hardware_voiceband_3000.png",
    "rx_4000_v2.wav": "hardware_voiceband_4000.png",
    "rx_5000_v2.wav": "hardware_voiceband_5000.png",
}

spec_cache = {}
global_max_db = -1000.0
global_min_db = 1000.0

for filename in spec_map:
    data = rec[filename]["data"]
    sr = rec[filename]["sr"]
    num_frames = max(0, (len(data) - n_fft_spec) // hop_spec + 1)
    window = np.hanning(n_fft_spec)
    freqs = np.fft.rfftfreq(n_fft_spec, 1.0 / sr)
    mask = (freqs >= 500) & (freqs <= 6000)
    sub_freqs = freqs[mask]

    matrix = []
    for i in range(num_frames):
        frame = data[i * hop_spec : i * hop_spec + n_fft_spec]
        mag = np.abs(np.fft.rfft(frame * window)) * (2.0 / n_fft_spec)
        matrix.append(mag[mask])
    matrix = np.array(matrix).T
    matrix_db = 20.0 * np.log10(np.maximum(matrix, 1e-10))

    spec_cache[filename] = {
        "matrix_db": matrix_db,
        "sub_freqs": sub_freqs,
        "duration": len(data) / sr,
    }
    global_max_db = max(global_max_db, np.max(matrix_db))
    global_min_db = min(global_min_db, np.min(matrix_db))

for filename, out_img in spec_map.items():
    m_db = spec_cache[filename]["matrix_db"]
    s_freqs = spec_cache[filename]["sub_freqs"]
    dur = spec_cache[filename]["duration"]

    plt.figure(figsize=(12, 5))
    plt.imshow(
        m_db,
        aspect="auto",
        origin="lower",
        extent=[0, dur, s_freqs[0] / 1000.0, s_freqs[-1] / 1000.0],
        cmap="inferno",
        vmin=global_min_db + 20,
        vmax=global_max_db,
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.title(f"Voiceband Spectrogram (0.5-6 kHz) — {filename}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"  Saved: {out_img}")


# ─────────────────────────────────────────────────────────────
# STEP 8/10: SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 10: SUMMARY TABLE")
print("=" * 75)
print(
    f"{'Freq':>6s} | {'Base Mag':>14s} | {'Rec Mag':>14s} | {'Ratio':>8s} | {'dB':>8s} | "
    f"{'Obs Hz':>8s} | {'Prom':>8s} | {'STFT dur':>8s} | Classification"
)
print("-" * 120)

for filename, target_freq in targets:
    r = results[filename]
    print(
        f"{r['target_freq']:6.0f} | {r['base_mag']:14.10f} | {r['rec_mag']:14.10f} | "
        f"{r['ratio']:8.2f} | {r['rel_db']:+8.2f} | {r['obs_freq']:8.1f} | "
        f"{r['prom_ratio']:8.1f} | {r['duration_sec']:7.2f}s | {r['classification']}"
    )

print()
