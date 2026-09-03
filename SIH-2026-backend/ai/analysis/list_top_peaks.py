"""Ten strongest peaks in the 17-21 kHz band.

Usage: python analysis/list_top_peaks.py [recording.wav]
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

import sys
FILE = sys.argv[1] if len(sys.argv) > 1 else "victim_recording.wav"
sr, audio = wavfile.read(FILE)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float64)
audio -= np.mean(audio)

spectrum = np.abs(
    np.fft.rfft(audio * np.hanning(len(audio)))
)

freqs = np.fft.rfftfreq(
    len(audio),
    1 / sr
)

mask = (freqs >= 17000) & (freqs <= 21000)

bf = freqs[mask]
bs = spectrum[mask]

peaks, _ = find_peaks(
    bs,
    distance=100
)

# Ten strongest high-frequency peaks
strongest = peaks[
    np.argsort(bs[peaks])[-10:]
][::-1]

print("Top high-frequency peaks:")
for i, p in enumerate(strongest, 1):
    print(
        f"{i:2d}. {bf[p]:9.1f} Hz   magnitude={bs[p]:.3f}"
    )