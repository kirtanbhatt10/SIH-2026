import numpy as np
from scipy.io import wavfile
from pathlib import Path

TESTS = [
    ("baseline_quiet.wav", None),
    ("rx_18000.wav", 18000),
    ("rx_18200.wav", 18200),
    ("rx_18400.wav", 18400),
    ("rx_18600.wav", 18600),
    ("rx_18800.wav", 18800),
    ("rx_19000.wav", 19000),
]

LOW = 17000
HIGH = 21000

print("=" * 78)
print("PHYSICAL FREQUENCY RESPONSE TEST")
print("=" * 78)

for filename, expected in TESTS:

    path = Path(filename)

    if not path.exists():
        print(f"{filename:22s} MISSING")
        continue

    sr, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)
    audio -= np.mean(audio)

    window = np.hanning(len(audio))

    spectrum = np.abs(
        np.fft.rfft(audio * window)
    )

    freqs = np.fft.rfftfreq(
        len(audio),
        d=1.0 / sr
    )

    band = (freqs >= LOW) & (freqs <= HIGH)

    bf = freqs[band]
    bs = spectrum[band]

    idx = np.argmax(bs)

    strongest_freq = float(bf[idx])
    strongest_mag = float(bs[idx])

    if expected is None:
        print(
            f"{filename:22s} "
            f"baseline peak={strongest_freq:8.1f} Hz  "
            f"mag={strongest_mag:.3g}"
        )
        continue

    # Search within +/- 50 Hz of expected carrier
    expected_band = (
        (freqs >= expected - 50)
        & (freqs <= expected + 50)
    )

    expected_mag = float(
        np.max(spectrum[expected_band])
    )

    # Local reference excluding expected carrier region
    reference_band = (
        band
        & (
            (freqs < expected - 150)
            | (freqs > expected + 150)
        )
    )

    reference = float(
        np.median(spectrum[reference_band])
    )

    relative_db = 20 * np.log10(
        (expected_mag + 1e-12)
        / (reference + 1e-12)
    )

    error = abs(strongest_freq - expected)

    print(
        f"{filename:22s} "
        f"TX={expected:7.0f}  "
        f"strongest={strongest_freq:8.1f}  "
        f"error={error:6.1f}  "
        f"carrier/ref={relative_db:6.1f} dB"
    )