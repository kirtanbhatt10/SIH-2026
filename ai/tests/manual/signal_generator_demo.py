import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..' if 'manual' in __file__ else '..')))

import os
import shutil
import numpy as np
from dsp.utils import SignalGenerator

print("==================================================")
print("  Manual Test Suite for dsp/utils.py")
print("==================================================")

sr = 48000
gen = SignalGenerator(sample_rate=sr)

# Helper check
def check_signal(name, sig, expected_len=48000):
    is_valid = (
        len(sig) == expected_len and
        sig.ndim == 1 and
        np.all(np.isfinite(sig)) and
        np.max(np.abs(sig)) <= 1.01
    )
    if is_valid:
        print(f"[PASS] Test {name}: length={len(sig)}, 1D=True, finite=True, peak={np.max(np.abs(sig)):.3f} <= 1.01")
    else:
        print(f"[FAIL] Test {name}: length={len(sig)}, 1D={sig.ndim==1}, finite={np.all(np.isfinite(sig))}, peak={np.max(np.abs(sig))}")
    assert is_valid

# 1. FSK
fsk = gen.generate_fsk(duration_sec=1.0)
check_signal("1 (generate_fsk)", fsk)

# 2. OOK
ook = gen.generate_ook(duration_sec=1.0)
check_signal("2 (generate_ook)", ook)

# 3. Chirp
chirp = gen.generate_chirp(duration_sec=1.0)
check_signal("3 (generate_chirp)", chirp)

# 4. Tone
tone = gen.generate_tone(duration_sec=1.0, frequency=19000)
check_signal("4 (generate_tone)", tone)

# 5. White noise
wn = gen.generate_white_noise(duration_sec=1.0)
check_wn = len(wn) == sr and wn.ndim == 1 and np.all(np.isfinite(wn))
print(f"[{'PASS' if check_wn else 'FAIL'}] Test 5 (generate_white_noise): length={len(wn)}, finite={check_wn}")
assert check_wn

# 6. Ambient noise
an = gen.generate_ambient_noise(duration_sec=1.0)
check_an = len(an) == sr and an.ndim == 1 and np.all(np.isfinite(an))
print(f"[{'PASS' if check_an else 'FAIL'}] Test 6 (generate_ambient_noise): length={len(an)}, finite={check_an}")
assert check_an

# 7. FSK frequency content
fsk_custom = gen.generate_fsk(duration_sec=1.0, freq_mark=18500, freq_space=19500, baud_rate=20)
fft_mag = np.abs(np.fft.rfft(fsk_custom * np.hanning(len(fsk_custom))))
freqs = np.fft.rfftfreq(len(fsk_custom), d=1.0/sr)

# Check energy near 18500 and 19500
idx_mark = np.argmin(np.abs(freqs - 18500))
idx_space = np.argmin(np.abs(freqs - 19500))
peak_mark = np.max(fft_mag[max(0, idx_mark-10):idx_mark+10])
peak_space = np.max(fft_mag[max(0, idx_space-10):idx_space+10])
mean_bg = np.mean(fft_mag)

fsk_freq_ok = peak_mark > mean_bg * 5 and peak_space > mean_bg * 5
print(f"[{'PASS' if fsk_freq_ok else 'FAIL'}] Test 7 (FSK frequency content): mark_peak={peak_mark:.1f}, space_peak={peak_space:.1f}, bg={mean_bg:.1f}")
assert fsk_freq_ok

# 8. Tone frequency content
tone_19k = gen.generate_tone(duration_sec=1.0, frequency=19000)
fft_mag_tone = np.abs(np.fft.rfft(tone_19k * np.hanning(len(tone_19k))))
peak_freq = freqs[np.argmax(fft_mag_tone)]
tone_freq_ok = abs(peak_freq - 19000) < 10
print(f"[{'PASS' if tone_freq_ok else 'FAIL'}] Test 8 (Tone frequency content): peak_freq={peak_freq:.1f} Hz (expected ~19000 Hz)")
assert tone_freq_ok

# 9. mix_with_noise
snrs = [20, 10, 0]
snr_ok = True
for snr in snrs:
    mixed = gen.mix_with_noise(tone_19k, snr_db=snr)
    if len(mixed) != len(tone_19k) or not np.all(np.isfinite(mixed)):
        snr_ok = False
        break
print(f"[{'PASS' if snr_ok else 'FAIL'}] Test 9 (mix_with_noise at 20dB, 10dB, 0dB): length unchanged & finite={snr_ok}")
assert snr_ok

# 10. WAV round trip
wav_path = "temp_test_tone.wav"
if os.path.exists(wav_path):
    os.remove(wav_path)
gen.save_wav(tone_19k, wav_path)
loaded = gen.load_wav(wav_path)
wav_ok = len(loaded) == len(tone_19k) and loaded.ndim == tone_19k.ndim
if os.path.exists(wav_path):
    os.remove(wav_path)
print(f"[{'PASS' if wav_ok else 'FAIL'}] Test 10 (WAV round trip): loaded length={len(loaded)}, shape={loaded.shape}")
assert wav_ok

# 11. Small dataset generation
out_dir = "test_training_data"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

X, y = gen.generate_dataset(
    output_dir=out_dir,
    samples_per_class=3,
    chunk_duration_sec=0.1,
    n_fft=2048
)

dataset_ok = (
    X.shape[1] == 32 and
    len(X) == len(y) and
    np.all(np.isfinite(X)) and
    os.path.exists(os.path.join(out_dir, "metadata.json")) and
    os.path.exists(os.path.join(out_dir, "label_map.json")) and
    os.path.exists(os.path.join(out_dir, "features.npy")) and
    os.path.exists(os.path.join(out_dir, "labels.npy"))
)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

print(f"[{'PASS' if dataset_ok else 'FAIL'}] Test 11 (Dataset Generation): X.shape={X.shape}, y.shape={y.shape}, metadata/npy files verified")
assert dataset_ok

print("\n==================================================")
print("  [ALL UTILS MANUAL TESTS PASSED SUCCESSFULLY!]")
print("==================================================")
