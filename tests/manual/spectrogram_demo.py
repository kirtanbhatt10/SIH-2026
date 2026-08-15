import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..' if 'manual' in __file__ else '..')))

import os
import pprint
import numpy as np
from dsp.spectrogram_generator import SpectrogramGenerator

print("--- Step 7: Generating Synthetic FSK Signal ---")
sr = 48000
n_fft = 2048
t = np.linspace(0, 1, sr)
fsk = np.zeros(sr)
for i in range(20):
    freq = 19500 if i % 2 else 18500
    start = i * 2400
    fsk[start:start+2400] = np.sin(2 * np.pi * freq * t[start:start+2400])

print("\n--- Step 8: Instantiating SpectrogramGenerator ---")
spec_gen = SpectrogramGenerator(
    sample_rate=48000,
    n_fft=2048,
    history_seconds=5.0,
    ui_bins=64
)
print("SpectrogramGenerator initialized OK.")

print("\n--- Step 9: Testing add_chunk_dict() ---")
chunk = fsk[:n_fft]
frame = spec_gen.add_chunk_dict(chunk)

assert isinstance(frame, dict), "Returned object is not a dict"
assert len(frame["freq_axis"]) == 64, f"Expected 64 freq_axis bins, got {len(frame['freq_axis'])}"
assert len(frame["magnitudes_db"]) == 64, f"Expected 64 magnitudes_db bins, got {len(frame['magnitudes_db'])}"
assert len(frame["magnitudes_norm"]) == 64, f"Expected 64 magnitudes_norm bins, got {len(frame['magnitudes_norm'])}"
assert all(0.0 <= v <= 1.0 for v in frame["magnitudes_norm"]), "magnitudes_norm value out of range [0, 1]"
assert 18000 <= frame["peak_freq_hz"] <= 22000, f"peak_freq_hz {frame['peak_freq_hz']} out of range"
assert isinstance(frame["ultrasonic_active"], bool), "ultrasonic_active is not bool"
assert np.isfinite(frame["peak_snr_db"]), "peak_snr_db is not finite"

print("Important Values Extracted:")
print(f"  peak_freq_hz      : {frame['peak_freq_hz']} Hz")
print(f"  peak_magnitude_db : {frame['peak_magnitude_db']:.2f} dB")
print(f"  peak_snr_db       : {frame['peak_snr_db']:.2f} dB")
print(f"  ultrasonic_active : {frame['ultrasonic_active']}")
print(f"  freq_axis range   : [{frame['freq_axis'][0]:.1f} Hz ... {frame['freq_axis'][-1]:.1f} Hz]")
print(f"  magnitudes_norm   : min={min(frame['magnitudes_norm']):.3f}, max={max(frame['magnitudes_norm']):.3f}")

print("[PASS] Step 9 assertions verified.")

print("\n--- Step 10: Adding 10 Frames and Testing get_full_spectrogram() ---")
spec_gen.clear_history()
chunk_len = n_fft
for i in range(10):
    sub_chunk = fsk[i * chunk_len : (i + 1) * chunk_len]
    spec_gen.add_chunk(sub_chunk)

full_data = spec_gen.get_full_spectrogram()
spec_shape = full_data["spectrogram_db"].shape
print(f"Full spectrogram shape: {spec_shape}")
assert spec_shape == (64, 10), f"Expected shape (64, 10), got {spec_shape}"
print("[PASS] Step 10 shape (64, 10) verified.")

print("\n--- Step 11: Testing save_spectrogram_image() ---")
img_filename = "test_spectrogram.png"
if os.path.exists(img_filename):
    os.remove(img_filename)

spec_gen.save_spectrogram_image(img_filename)
assert os.path.exists(img_filename), f"Image file {img_filename} was not created!"
file_size = os.path.getsize(img_filename)
print(f"Verified image file '{img_filename}' exists (size: {file_size} bytes).")
print("[PASS] Step 11 image saving verified.")

print("\n--- Step 12: Testing clear_history() ---")
spec_gen.clear_history()
print(f"Frame count after clear_history(): {spec_gen.frame_count}")
assert spec_gen.frame_count == 0, f"Expected frame_count == 0, got {spec_gen.frame_count}"
print("[PASS] Step 12 history clearing verified.")

print("\n[ALL SPECTROGRAM MANUAL TESTS PASSED SUCCESSFULLY!]")
