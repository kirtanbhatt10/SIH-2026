import numpy as np
from scipy.io import wavfile

TESTS = [
    ("rx_18000.wav", 18000),
    ("rx_18200.wav", 18200),
    ("rx_18400.wav", 18400),
    ("rx_18600.wav", 18600),
    ("rx_18800.wav", 18800),
    ("rx_19000.wav", 19000),
]

def load_spectrum(filename):
    sr, audio = wavfile.read(filename)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)
    audio -= np.mean(audio)

    # Normalize integer recordings for numerical consistency
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    window = np.hanning(len(audio))

    spectrum = np.abs(
        np.fft.rfft(audio * window)
    )

    # Compensate for different recording lengths
    spectrum /= np.sum(window)

    freqs = np.fft.rfftfreq(
        len(audio),
        1 / sr
    )

    return sr, freqs, spectrum


def magnitude_near(freqs, spectrum, target, width=10):
    mask = (
        (freqs >= target - width)
        & (freqs <= target + width)
    )

    if not np.any(mask):
        return 0.0

    return float(np.max(spectrum[mask]))


_, base_freqs, base_spec = load_spectrum(
    "baseline_quiet.wav"
)

print("=" * 78)
print("BASELINE VS TRANSMISSION COMPARISON")
print("=" * 78)
print()
print(
    f"{'TX':>8} "
    f"{'Baseline':>12} "
    f"{'Recorded':>12} "
    f"{'Ratio':>10} "
    f"{'Increase dB':>14}"
)
print("-" * 78)

for filename, target in TESTS:

    _, freqs, spectrum = load_spectrum(
        filename
    )

    # Frequency axes can differ slightly if lengths differ,
    # so measure each recording independently.
    baseline_mag = magnitude_near(
        base_freqs,
        base_spec,
        target
    )

    recorded_mag = magnitude_near(
        freqs,
        spectrum,
        target
    )

    ratio = recorded_mag / (
        baseline_mag + 1e-12
    )

    increase_db = 20 * np.log10(
        ratio + 1e-12
    )

    print(
        f"{target:8.0f} "
        f"{baseline_mag:12.6e} "
        f"{recorded_mag:12.6e} "
        f"{ratio:10.2f} "
        f"{increase_db:14.2f}"
    )