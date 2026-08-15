"""
Comprehensive Diagnostic & Spectral Analysis of Physical Ultrasonic Recordings
================================================================================
"""

import os
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
    print("\n" + "="*70)
    print("STEP 1: RECORDING METADATA & HASH VERIFICATION")
    print("="*70)
    
    info_list = []
    hashes = set()
    
    for f in files:
        sr, data = wavfile.read(f)
        data = data.astype(np.float64)
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


def analyze_source_tones(source_files):
    print("\n" + "="*70)
    print("STEP 2: SOURCE TONE ANALYSIS")
    print("="*70)
    
    for f in source_files:
        if not os.path.exists(f):
            print(f"Source file {f} not found.")
            continue
        sr, data = wavfile.read(f)
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]
        
        # Windowed FFT
        n = len(data)
        window = np.hanning(n)
        fft_mag = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        
        peak_idx = np.argmax(fft_mag)
        peak_freq = freqs[peak_idx]
        peak_mag = fft_mag[peak_idx]
        
        print(f"Source WAV   : {f}")
        print(f"  Dominant Peak: {peak_freq:.1f} Hz (magnitude: {peak_mag:.6f})")


def compute_band_fft(data, sr, f_min=17000, f_max=21000):
    n = len(data)
    window = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(data * window)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    
    mask = (freqs >= f_min) & (freqs <= f_max)
    band_freqs = freqs[mask]
    band_mags = fft_mag[mask]
    
    return freqs, fft_mag, band_freqs, band_mags


def analyze_raw_frequencies(rec_dict):
    print("\n" + "="*70)
    print("STEP 3: RAW FREQUENCY ANALYSIS (17 kHz - 21 kHz)")
    print("="*70)
    
    results = {}
    for name, info in rec_dict.items():
        freqs, fft_mag, b_freqs, b_mags = compute_band_fft(info["data"], info["sr"])
        
        # Top 10 peaks in 17k-21k band
        # Find local peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(b_mags, distance=20)
        if len(peaks) == 0:
            top_indices = np.argsort(b_mags)[-10:][::-1]
        else:
            top_indices = peaks[np.argsort(b_mags[peaks])[-10:]][::-1]
            
        print(f"Top 10 High-Frequency Peaks in '{name}':")
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
    print("\n" + "="*70)
    print("STEPS 4, 5, 6: CARRIER MEASUREMENTS, BASELINE INCREASE & LOCAL PROMINENCE")
    print("="*70)
    
    base_data = rec_dict["baseline_new.wav"]["data"]
    sr = rec_dict["baseline_new.wav"]["sr"]
    b_freqs, b_fft = band_res["baseline_new.wav"]["freqs"], band_res["baseline_new.wav"]["fft_mag"]
    
    targets = [
        ("rx_18000.wav", 18000.0),
        ("rx_19000.wav", 19000.0)
    ]
    
    measurements = {}
    
    for filename, target_freq in targets:
        rx_freqs = band_res[filename]["freqs"]
        rx_fft = band_res[filename]["fft_mag"]
        
        # 1. Target region ± 20 Hz
        target_mask = (rx_freqs >= target_freq - 20) & (rx_freqs <= target_freq + 20)
        base_mask = (b_freqs >= target_freq - 20) & (b_freqs <= target_freq + 20)
        
        rx_peak_mag = np.max(rx_fft[target_mask]) if np.any(target_mask) else 1e-12
        rx_peak_freq = rx_freqs[target_mask][np.argmax(rx_fft[target_mask])] if np.any(target_mask) else target_freq
        
        base_peak_mag = np.max(b_fft[base_mask]) if np.any(base_mask) else 1e-12
        
        # Relative increase
        ratio = rx_peak_mag / max(base_peak_mag, 1e-12)
        rel_db = 20.0 * np.log10(ratio)
        
        # Local background in rx recording: 17k-21k excluding target_freq ± 100 Hz
        local_bg_mask = (rx_freqs >= 17000) & (rx_freqs <= 21000) & (np.abs(rx_freqs - target_freq) > 100)
        local_bg_median = np.median(rx_fft[local_bg_mask]) if np.any(local_bg_mask) else 1e-12
        local_bg_mean = np.mean(rx_fft[local_bg_mask]) if np.any(local_bg_mask) else 1e-12
        
        local_prom_ratio = rx_peak_mag / max(local_bg_median, 1e-12)
        local_prom_db = 20.0 * np.log10(local_prom_ratio)
        
        print(f"Target: {target_freq:.0f} Hz in '{filename}'")
        print(f"  Observed Peak Freq     : {rx_peak_freq:.1f} Hz")
        print(f"  Recorded Carrier Mag   : {rx_peak_mag:12.8f}")
        print(f"  Baseline Carrier Mag   : {base_peak_mag:12.8f}")
        print(f"  Relative FFT Increase  : {ratio:8.2f}x ({rel_db:+6.2f} dB)")
        print(f"  Local BG Median Mag    : {local_bg_median:12.8f}")
        print(f"  Local Prominence       : {local_prom_ratio:8.2f}x ({local_prom_db:+6.2f} dB above local bg)\n")
        
        measurements[filename] = {
            "target_freq": target_freq,
            "obs_freq": rx_peak_freq,
            "rec_mag": rx_peak_mag,
            "base_mag": base_peak_mag,
            "ratio": ratio,
            "rel_db": rel_db,
            "local_prom_ratio": local_prom_ratio,
            "local_prom_db": local_prom_db
        }
        
    return measurements


def analyze_time_local_stft(rec_dict):
    print("\n" + "="*70)
    print("STEP 7: TIME-LOCAL STFT SLIDING WINDOW ANALYSIS")
    print("="*70)
    
    stft_results = {}
    targets = [
        ("baseline_new.wav", 18000.0),
        ("baseline_new.wav", 19000.0),
        ("rx_18000.wav", 18000.0),
        ("rx_19000.wav", 19000.0)
    ]
    
    n_fft = 8192
    hop = 2048
    
    for filename, target_freq in targets:
        data = rec_dict[filename]["data"]
        sr = rec_dict[filename]["sr"]
        
        num_frames = (len(data) - n_fft) // hop + 1
        window = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        
        target_idx = np.argmin(np.abs(freqs - target_freq))
        
        frame_energies = []
        for i in range(num_frames):
            frame = data[i * hop : i * hop + n_fft]
            spec = np.abs(np.fft.rfft(frame * window)) * (2.0 / n_fft)
            # Energy near target_freq ± 30 Hz
            t_mask = np.abs(freqs - target_freq) <= 30
            frame_energy = np.max(spec[t_mask])
            frame_energies.append(frame_energy)
            
        frame_energies = np.array(frame_energies)
        max_energy = np.max(frame_energies)
        median_energy = np.median(frame_energies)
        
        # Duration elevated (> 3x median energy)
        elevated_frames = np.sum(frame_energies > max(3.0 * median_energy, 1e-7))
        duration_sec = elevated_frames * (hop / sr)
        
        print(f"File: {filename:16s} | Target: {target_freq:.0f} Hz")
        print(f"  Max STFT Mag Over Time : {max_energy:12.8f}")
        print(f"  Median STFT Mag        : {median_energy:12.8f}")
        print(f"  Elevated Duration      : {duration_sec:.2f} s")
        print(f"  Narrow Line Visible    : {'YES' if elevated_frames > 2 and max_energy > 5 * median_energy else 'NO'}\n")
        
        stft_results[(filename, target_freq)] = {
            "max_energy": max_energy,
            "median_energy": median_energy,
            "duration_sec": duration_sec,
            "line_visible": elevated_frames > 2 and max_energy > 5 * median_energy
        }
    return stft_results


def generate_spectrogram_images(rec_dict):
    print("\n" + "="*70)
    print("STEP 8: SPECTROGRAM IMAGE GENERATION")
    print("="*70)
    
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
        
        mask = (freqs >= 17000) & (freqs <= 21000)
        sub_freqs = freqs[mask]
        
        matrix = []
        for i in range(num_frames):
            frame = data[i * hop : i * hop + n_fft]
            mag = np.abs(np.fft.rfft(frame * window)) * (2.0 / n_fft)
            matrix.append(mag[mask])
            
        matrix = np.array(matrix).T  # Shape: (freq_bins, num_frames)
        matrix_db = 20.0 * np.log10(np.maximum(matrix, 1e-8))
        
        spec_data[filename] = {
            "matrix_db": matrix_db,
            "sub_freqs": sub_freqs,
            "duration": info["duration"]
        }
        
        global_max_db = max(global_max_db, np.max(matrix_db))
        global_min_db = min(global_min_db, np.min(matrix_db))
        
    print(f"Global Spectrogram Dynamic Range: {global_min_db:.1f} dB to {global_max_db:.1f} dB")
    
    # Save individual plots with shared colorbar scale
    out_map = {
        "baseline_new.wav": "hardware_test_baseline.png",
        "rx_18000.wav": "hardware_test_18000.png",
        "rx_19000.wav": "hardware_test_19000.png"
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
        plt.title(f"Spectrogram (17-21 kHz) — {filename}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (kHz)")
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"  Saved Spectrogram: {out_img}")


def check_baseline_artifact(rec_dict, band_res):
    print("\n" + "="*70)
    print("STEP 10: BASELINE ARTIFACT CHECK (~19.4 kHz)")
    print("="*70)
    
    for filename, info in rec_dict.items():
        freqs = band_res[filename]["freqs"]
        fft_mag = band_res[filename]["fft_mag"]
        
        artifact_mask = (freqs >= 19350) & (freqs <= 19450)
        if np.any(artifact_mask):
            peak_mag = np.max(fft_mag[artifact_mask])
            peak_freq = freqs[artifact_mask][np.argmax(fft_mag[artifact_mask])]
            print(f"File: {filename:16s} | ~19.4 kHz Region Peak: {peak_freq:.1f} Hz (mag: {peak_mag:12.8f})")


def observe_dsp_pipeline(rec_dict):
    print("\n" + "="*70)
    print("STEP 11: OPTIONAL DSP PIPELINE OBSERVATION")
    print("="*70)
    
    for filename, info in rec_dict.items():
        pipeline = DSPPipeline(sample_rate=info["sr"])
        data = info["data"]
        chunk_size = 2048
        num_chunks = len(data) // chunk_size
        
        for i in range(num_chunks):
            chunk = data[i * chunk_size : (i + 1) * chunk_size]
            pipeline.process(chunk)
            
        verdict = pipeline.get_verdict()
        print(f"File: {filename}")
        print(f"  Accumulated Threat Verdict : {verdict['is_threat']}")
        print(f"  Average Suspicion Score   : {verdict['average_suspicion']:.4f}")
        print(f"  Consistent Carriers (Hz)  : {verdict['consistent_carriers']}")
        print(f"  Chunks Analyzed           : {verdict['chunks_analyzed']}\n")


def main():
    files = ["baseline_new.wav", "rx_18000.wav", "rx_19000.wav"]
    rec_dict = analyze_recording_metadata(files)
    
    source_files = ["18000hz.wav", "19000hz.wav"]
    analyze_source_tones(source_files)
    
    band_res = analyze_raw_frequencies(rec_dict)
    carrier_meas = analyze_expected_carrier_and_prominence(rec_dict, band_res)
    stft_res = analyze_time_local_stft(rec_dict)
    generate_spectrogram_images(rec_dict)
    check_baseline_artifact(rec_dict, band_res)
    observe_dsp_pipeline(rec_dict)

if __name__ == "__main__":
    main()
