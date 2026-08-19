"""Peak SNR in the 17-21 kHz band.

Usage: python analysis/analyze_peak_snr.py [recording.wav]
"""
import numpy as np
from scipy.io import wavfile

import sys
FILE = sys.argv[1] if len(sys.argv) > 1 else "victim_recording.wav"

sr, audio = wavfile.read(FILE)

# Stereo -> mono if necessary
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float64)

# Remove DC
audio -= np.mean(audio)

# Hann window
window = np.hanning(len(audio))

# FFT
spectrum = np.abs(np.fft.rfft(audio * window))
freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)

# Examine 17–21 kHz
mask = (freqs >= 17000) & (freqs <= 21000)

band_freqs = freqs[mask]
band_spectrum = spectrum[mask]

peak_index = np.argmax(band_spectrum)

peak_freq = float(band_freqs[peak_index])
peak_mag = float(band_spectrum[peak_index])

# Estimate local spectral floor
noise_floor = float(np.median(band_spectrum))

ratio = peak_mag / (noise_floor + 1e-12)

snr_like_db = 20 * np.log10(ratio + 1e-12)

print("=" * 50)
print("PHYSICAL ULTRASONIC TEST")
print("=" * 50)

print(f"Sample rate           : {sr} Hz")
print(f"Duration              : {len(audio)/sr:.2f} sec")
print(f"Strongest 17-21k peak : {peak_freq:.1f} Hz")
print(f"Peak / median ratio   : {ratio:.2f}")
print(f"Peak prominence       : {snr_like_db:.1f} dB")

# Specifically inspect around expected 19 kHz
expected_mask = (freqs >= 18950) & (freqs <= 19050)
expected_peak = np.max(spectrum[expected_mask])

print(f"Energy near 19 kHz    : {expected_peak:.2f}")

if abs(peak_freq - 19000) <= 50 and snr_like_db > 10:
    print("\nRESULT: 19 kHz physical transmission detected!")
elif abs(peak_freq - 19000) <= 100:
    print("\nRESULT: Signal near 19 kHz found, but it is weak.")
else:
    print("\nRESULT: No dominant 19 kHz transmission found.")