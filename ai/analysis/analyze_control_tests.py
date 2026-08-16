"""
Diagnostic & Spectral Analysis of Audible Control Physical Test Recordings
=============================================================================
"""

import os
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

# Add project root & dsp dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def analyze_recording_metadata(files):
    print("\n" + "="*75)
    print("STEP 1: CONTROL RECORDINGS METADATA & HASH VERIFICATION")
    print("="*75)

    info_list = []
    hashes = set()

    for f in files:
        sr, raw_data = wavfile.read(f)
        if raw_data.dtype == np.int16:
            data = raw_data.astype(np.float64) / 32768.0
        elif raw_data.dtype == np.int32:
            data = raw_data.astype(np.float64) / 2147483648.0
        else:
            data = raw_data.astype(np.float64)

        if data.ndim > 1:
            data = data[:, 0]
        duration = len(data) / sr
        peak = np.max(np.abs(data))
        rms = np.sqrt(np.mean(data**2))
        sha = calculate_sha256(f)
        hashes.add(sha)

        info = {
            "file": f,
            "sr": sr,
            "duration": duration,
            "shape": data.shape,
            "dtype": raw_data.dtype,
            "peak": peak,
            "rms": rms,
            "sha256": sha,
            "data": data
        }
        info_list.append(info)
        print(f"File        : {f}")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  Duration   : {duration:.2f} s")
        print(f"  Shape      : {data.shape}")
        print(f"  Peak       : {peak:.6f}")
        print(f"  RMS        : {rms:.6f}")
        print(f"  SHA256     : {sha}\n")

    print(f"Unique SHA256 Hashes: {len(hashes)} of {len(files)}")
    assert len(hashes) == len(files), "Error: Recordings are not independent!"
    print("  [OK] All control recordings confirmed independent with distinct SHA256 hashes.")
    return {info["file"]: info for info in info_list}


def analyze_control_carriers(rec_dict):
    print("\n" + "="*75)
    print("STEPS 4 & 5: CARRIER MEASUREMENTS, BASELINE INCREASE & LOCAL PROMINENCE")
    print("="*75)

    base_data = rec_dict["baseline_control.wav"]["data"]
    sr = rec_dict["baseline_control.wav"]["sr"]

    # Compute baseline FFT
    n_base = len(base_data)
    w_base = np.hanning(n_base)
    base_fft = np.abs(np.fft.rfft(base_data * w_base)) * (2.0 / n_base)
    base_freqs = np.fft.rfftfreq(n_base, 1.0 / sr)

    targets = [
        ("rx_1000.wav", 1000.0),
        ("rx_5000.wav", 5000.0),
        ("rx_10000.wav", 10000.0),
        ("rx_12000.wav", 12000.0),
        ("rx_14000.wav", 14000.0)
    ]

    results = {}

    for filename, target_freq in targets:
        data = rec_dict[filename]["data"]
        n = len(data)
        window = np.hanning(n)
        rx_fft = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
        rx_freqs = np.fft.rfftfreq(n, 1.0 / sr)

        target_mask = (rx_freqs >= target_freq - 20) & (rx_freqs <= target_freq + 20)
        base_mask = (base_freqs >= target_freq - 20) & (base_freqs <= target_freq + 20)

        rx_peak_mag = np.max(rx_fft[target_mask]) if np.any(target_mask) else 1e-12
        rx_peak_freq = rx_freqs[target_mask][np.argmax(rx_fft[target_mask])] if np.any(target_mask) else target_freq

        base_peak_mag = np.max(base_fft[base_mask]) if np.any(base_mask) else 1e-12

        ratio = rx_peak_mag / max(base_peak_mag, 1e-12)
        rel_db = 20.0 * np.log10(ratio)

        # Local BG around target_freq ± 500 Hz excluding target_freq ± 100 Hz
        local_bg_mask = (rx_freqs >= max(100, target_freq - 500)) & (rx_freqs <= target_freq + 500) & (np.abs(rx_freqs - target_freq) > 100)
        local_bg_median = np.median(rx_fft[local_bg_mask]) if np.any(local_bg_mask) else 1e-12

        local_prom_ratio = rx_peak_mag / max(local_bg_median, 1e-12)
        local_prom_db = 20.0 * np.log10(local_prom_ratio)

        # STFT Sliding Window
        n_fft = 8192
        hop = 2048
        num_frames = (len(data) - n_fft) // hop + 1
        w_stft = np.hanning(n_fft)
        stft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        frame_energies = []
        for i in range(num_frames):
            frame = data[i * hop : i * hop + n_fft]
            spec = np.abs(np.fft.rfft(frame * w_stft)) * (2.0 / n_fft)
            t_mask = np.abs(stft_freqs - target_freq) <= 30
            frame_energy = np.max(spec[t_mask]) if np.any(t_mask) else 0.0
            frame_energies.append(frame_energy)

        frame_energies = np.array(frame_energies)
        max_stft = np.max(frame_energies)
        median_stft = np.median(frame_energies)
        elevated_frames = np.sum(frame_energies > max(3.0 * median_stft, 1e-7))
        duration_sec = elevated_frames * (hop / sr)

        # Classification
        if ratio >= 5.0 and local_prom_ratio >= 10.0 and duration_sec >= 1.0:
            classification = "CLEARLY CAPTURED"
        elif ratio >= 2.0 or local_prom_ratio >= 3.0:
            classification = "WEAK / UNCERTAIN"
        else:
            classification = "NOT CAPTURED"

        print(f"Target: {target_freq:.0f} Hz in '{filename}'")
        print(f"  Observed Peak Freq     : {rx_peak_freq:.1f} Hz")
        print(f"  Recorded Carrier Mag   : {rx_peak_mag:12.8f}")
        print(f"  Baseline Carrier Mag   : {base_peak_mag:12.8f}")
        print(f"  Relative FFT Increase  : {ratio:8.2f}x ({rel_db:+6.2f} dB)")
        print(f"  Local BG Median Mag    : {local_bg_median:12.8f}")
        print(f"  Local Prominence       : {local_prom_ratio:8.2f}x ({local_prom_db:+6.2f} dB above bg)")
        print(f"  STFT Elevated Duration : {duration_sec:.2f} s (Max STFT: {max_stft:.8f})")
        print(f"  Classification         : {classification}\n")

        results[filename] = {
            "target_freq": target_freq,
            "obs_freq": rx_peak_freq,
            "rec_mag": rx_peak_mag,
            "base_mag": base_peak_mag,
            "ratio": ratio,
            "rel_db": rel_db,
            "local_prom_ratio": local_prom_ratio,
            "local_prom_db": local_prom_db,
            "duration_sec": duration_sec,
            "classification": classification
        }

    return results


def generate_control_spectrograms(rec_dict):
    print("\n" + "="*75)
    print("STEP 7: CONTROL SPECTROGRAM IMAGE GENERATION")
    print("="*75)

    n_fft = 4096
    hop = 1024

    spec_data = {}
    global_max_db = -1000.0
    global_min_db = 1000.0

    out_map = {
        "baseline_control.wav": "hardware_control_baseline.png",
        "rx_1000.wav": "hardware_control_1000.png",
        "rx_5000.wav": "hardware_control_5000.png",
        "rx_10000.wav": "hardware_control_10000.png",
        "rx_12000.wav": "hardware_control_12000.png",
        "rx_14000.wav": "hardware_control_14000.png"
    }

    for filename, info in rec_dict.items():
        data = info["data"]
        sr = info["sr"]

        num_frames = (len(data) - n_fft) // hop + 1
        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        mask = (freqs >= 100) & (freqs <= 16000)
        sub_freqs = freqs[mask]

        matrix = []
        for i in range(num_frames):
            frame = data[i * hop : i * hop + n_fft]
            mag = np.abs(np.fft.rfft(frame * window)) * (2.0 / n_fft)
            matrix.append(mag[mask])

        matrix = np.array(matrix).T
        matrix_db = 20.0 * np.log10(np.maximum(matrix, 1e-8))

        spec_data[filename] = {
            "matrix_db": matrix_db,
            "sub_freqs": sub_freqs,
            "duration": info["duration"]
        }

        global_max_db = max(global_max_db, np.max(matrix_db))
        global_min_db = min(global_min_db, np.min(matrix_db))

    for filename, out_img in out_map.items():
        m_db = spec_data[filename]["matrix_db"]
        s_freqs = spec_data[filename]["sub_freqs"]
        dur = spec_data[filename]["duration"]

        plt.figure(figsize=(10, 5))
        plt.imshow(
            m_db,
            aspect='auto',
            origin='lower',
            extent=[0, dur, s_freqs[0] / 1000.0, s_freqs[-1] / 1000.0],
            cmap='viridis',
            vmin=global_min_db + 10,
            vmax=global_max_db
        )
        plt.colorbar(label='Magnitude (dB relative)')
        plt.title(f"Control Spectrogram (0.1-16 kHz) — {filename}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"  Saved Control Spectrogram: {out_img}")


def main():
    files = ["baseline_control.wav", "rx_1000.wav", "rx_5000.wav", "rx_10000.wav", "rx_12000.wav", "rx_14000.wav"]
    rec_dict = analyze_recording_metadata(files)
    res = analyze_control_carriers(rec_dict)
    generate_control_spectrograms(rec_dict)


if __name__ == "__main__":
    main()
