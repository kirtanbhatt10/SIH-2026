"""
Diagnostic & Spectral Analysis of 15-17.5 kHz Physical Ultrasonic Test Recordings
===================================================================================
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

from dsp.dsp_api import DSPPipeline


def calculate_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def analyze_recording_metadata(files):
    print("\n" + "="*75)
    print("STEP 1: RECORDING METADATA & SHA256 HASH VERIFICATION")
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
            "dtype": data.dtype,
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
    print("  [OK] All physical test recordings confirmed independent with distinct SHA256 hashes.")
    return {info["file"]: info for info in info_list}


def compute_band_fft(data, sr, f_min=14000, f_max=18500):
    n = len(data)
    window = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    
    mask = (freqs >= f_min) & (freqs <= f_max)
    band_freqs = freqs[mask]
    band_mags = fft_mag[mask]
    
    return freqs, fft_mag, band_freqs, band_mags


def analyze_raw_frequencies(rec_dict):
    print("\n" + "="*75)
    print("STEP 3: RAW FREQUENCY ANALYSIS (14.0 kHz - 18.5 kHz)")
    print("="*75)
    
    results = {}
    for name, info in rec_dict.items():
        freqs, fft_mag, b_freqs, b_mags = compute_band_fft(info["data"], info["sr"])
        
        peaks, _ = find_peaks(b_mags, distance=20)
        if len(peaks) == 0:
            top_indices = np.argsort(b_mags)[-10:][::-1]
        else:
            top_indices = peaks[np.argsort(b_mags[peaks])[-10:]][::-1]
            
        print(f"Top High-Frequency Peaks in '{name}':")
        top_list = []
        for idx in top_indices:
            pf = b_freqs[idx]
            pm = b_mags[idx]
            top_list.append((pf, pm))
            print(f"  {pf:8.1f} Hz | Magnitude: {pm:10.8f}")
        print()
        results[name] = {
            "freqs": freqs,
            "fft_mag": fft_mag,
            "b_freqs": b_freqs,
            "b_mags": b_mags,
            "top_peaks": top_list
        }
    return results


def analyze_expected_carrier_and_prominence(rec_dict, band_res):
    print("\n" + "="*75)
    print("STEPS 4 & 5: CARRIER MEASUREMENTS, BASELINE INCREASE & LOCAL PROMINENCE")
    print("="*75)
    
    base_info = rec_dict["baseline_lower.wav"]
    b_freqs = band_res["baseline_lower.wav"]["freqs"]
    b_fft = band_res["baseline_lower.wav"]["fft_mag"]
    
    targets = [
        ("rx_15000.wav", 15000.0),
        ("rx_16000.wav", 16000.0),
        ("rx_17000.wav", 17000.0),
        ("rx_17500.wav", 17500.0)
    ]
    
    measurements = {}
    
    for filename, target_freq in targets:
        rx_freqs = band_res[filename]["freqs"]
        rx_fft = band_res[filename]["fft_mag"]
        
        target_mask = (rx_freqs >= target_freq - 20) & (rx_freqs <= target_freq + 20)
        base_mask = (b_freqs >= target_freq - 20) & (b_freqs <= target_freq + 20)
        
        rx_peak_mag = np.max(rx_fft[target_mask]) if np.any(target_mask) else 1e-12
        rx_peak_freq = rx_freqs[target_mask][np.argmax(rx_fft[target_mask])] if np.any(target_mask) else target_freq
        
        base_peak_mag = np.max(b_fft[base_mask]) if np.any(base_mask) else 1e-12
        
        ratio = rx_peak_mag / max(base_peak_mag, 1e-12)
        rel_db = 20.0 * np.log10(ratio)
        
        local_bg_mask = (rx_freqs >= 14000) & (rx_freqs <= 18500) & (np.abs(rx_freqs - target_freq) > 100)
        local_bg_median = np.median(rx_fft[local_bg_mask]) if np.any(local_bg_mask) else 1e-12
        
        local_prom_ratio = rx_peak_mag / max(local_bg_median, 1e-12)
        local_prom_db = 20.0 * np.log10(local_prom_ratio)
        
        # Classification
        if ratio >= 5.0 and local_prom_ratio >= 10.0:
            result_class = "CLEARLY CAPTURED"
        elif ratio >= 2.0 or local_prom_ratio >= 3.0:
            result_class = "WEAK / UNCERTAIN"
        else:
            result_class = "NOT CAPTURED"
            
        print(f"Target: {target_freq:.0f} Hz in '{filename}'")
        print(f"  Observed Peak Freq     : {rx_peak_freq:.1f} Hz")
        print(f"  Recorded Carrier Mag   : {rx_peak_mag:12.8f}")
        print(f"  Baseline Carrier Mag   : {base_peak_mag:12.8f}")
        print(f"  Relative FFT Increase  : {ratio:8.2f}x ({rel_db:+6.2f} dB)")
        print(f"  Local BG Median Mag    : {local_bg_median:12.8f}")
        print(f"  Local Prominence       : {local_prom_ratio:8.2f}x ({local_prom_db:+6.2f} dB above bg)")
        print(f"  Classification         : {result_class}\n")
        
        measurements[filename] = {
            "target_freq": target_freq,
            "obs_freq": rx_peak_freq,
            "rec_mag": rx_peak_mag,
            "base_mag": base_peak_mag,
            "ratio": ratio,
            "rel_db": rel_db,
            "local_prom_ratio": local_prom_ratio,
            "local_prom_db": local_prom_db,
            "classification": result_class
        }
        
    return measurements


def analyze_time_local_stft(rec_dict):
    print("\n" + "="*75)
    print("STEP 6: TIME-LOCAL STFT SLIDING WINDOW ANALYSIS")
    print("="*75)
    
    stft_results = {}
    targets = [
        ("baseline_lower.wav", 15000.0),
        ("rx_15000.wav", 15000.0),
        ("rx_16000.wav", 16000.0),
        ("rx_17000.wav", 17000.0),
        ("rx_17500.wav", 17500.0)
    ]
    
    n_fft = 8192
    hop = 2048
    
    for filename, target_freq in targets:
        data = rec_dict[filename]["data"]
        sr = rec_dict[filename]["sr"]
        
        num_frames = (len(data) - n_fft) // hop + 1
        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        
        frame_energies = []
        for i in range(num_frames):
            frame = data[i * hop : i * hop + n_fft]
            spec = np.abs(np.fft.rfft(frame * window)) * (2.0 / n_fft)
            t_mask = np.abs(freqs - target_freq) <= 30
            frame_energy = np.max(spec[t_mask]) if np.any(t_mask) else 0.0
            frame_energies.append(frame_energy)
            
        frame_energies = np.array(frame_energies)
        max_energy = np.max(frame_energies)
        median_energy = np.median(frame_energies)
        
        elevated_frames = np.sum(frame_energies > max(3.0 * median_energy, 1e-7))
        duration_sec = elevated_frames * (hop / sr)
        line_visible = (elevated_frames > 2) and (max_energy > 5.0 * max(median_energy, 1e-8))
        
        print(f"File: {filename:18s} | Target: {target_freq:.0f} Hz")
        print(f"  Max STFT Mag Over Time : {max_energy:12.8f}")
        print(f"  Median STFT Mag        : {median_energy:12.8f}")
        print(f"  Elevated Duration      : {duration_sec:.2f} s")
        print(f"  Narrow Line Visible    : {'YES' if line_visible else 'NO'}\n")
        
        stft_results[(filename, target_freq)] = {
            "max_energy": max_energy,
            "median_energy": median_energy,
            "duration_sec": duration_sec,
            "line_visible": line_visible
        }
    return stft_results


def generate_spectrogram_images(rec_dict):
    print("\n" + "="*75)
    print("STEP 7: SPECTROGRAM IMAGE GENERATION (14-18.5 kHz)")
    print("="*75)
    
    n_fft = 4096
    hop = 1024
    
    spec_data = {}
    global_max_db = -1000.0
    global_min_db = 1000.0
    
    for filename, info in rec_dict.items():
        data = info["data"]
        sr = info["sr"]
        
        num_frames = (len(data) - n_fft) // hop + 1
        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        
        mask = (freqs >= 14000) & (freqs <= 18500)
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
        
    out_map = {
        "baseline_lower.wav": "hardware_test_lower_baseline.png",
        "rx_15000.wav": "hardware_test_15000.png",
        "rx_16000.wav": "hardware_test_16000.png",
        "rx_17000.wav": "hardware_test_17000.png",
        "rx_17500.wav": "hardware_test_17500.png"
    }
    
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
        plt.title(f"Spectrogram (14-18.5 kHz) — {filename}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"  Saved Spectrogram: {out_img}")


def main():
    files = ["baseline_lower.wav", "rx_15000.wav", "rx_16000.wav", "rx_17000.wav", "rx_17500.wav"]
    rec_dict = analyze_recording_metadata(files)
    band_res = analyze_raw_frequencies(rec_dict)
    carrier_meas = analyze_expected_carrier_and_prominence(rec_dict, band_res)
    stft_res = analyze_time_local_stft(rec_dict)
    generate_spectrogram_images(rec_dict)


if __name__ == "__main__":
    main()
