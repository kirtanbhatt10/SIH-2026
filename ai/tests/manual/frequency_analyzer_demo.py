import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..' if 'manual' in __file__ else '..')))

import numpy as np
from dsp.frequency_analyzer import FrequencyAnalyzer

print("--- Step 6: Initializing FrequencyAnalyzer ---")
analyzer = FrequencyAnalyzer(sample_rate=48000, n_fft=2048)

sr = 48000
t = np.linspace(0, 1, sr)
fsk = np.zeros(sr)
for i in range(20):
    freq = 19500 if i % 2 else 18500
    start = i * 2400
    fsk[start:start+2400] = np.sin(2 * np.pi * freq * t[start:start+2400])

print("\n--- Testing Synthetic FSK Signal ---")
result_fsk = analyzer.analyze(fsk)
print("Complete returned dictionary:")
import pprint
pprint.pprint(result_fsk)

required_keys = [
    "modulation_type",
    "confidence",
    "carrier_freqs",
    "num_peaks",
    "is_suspicious",
    "suspicion_score",
    "energy_ratio",
    "estimated_baud"
]

for key in required_keys:
    assert key in result_fsk, f"Missing key '{key}' in FSK result"

print("[PASS] All required keys present in FSK result.")

print("\n--- Step 7: Testing Additional Signals ---")

# 19000 Hz steady tone
tone = np.sin(2 * np.pi * 19000 * t)
result_tone = analyzer.analyze(tone)
print("\n[Steady 19000 Hz Tone Result]:")
pprint.pprint(result_tone)
for key in required_keys:
    assert key in result_tone, f"Missing key '{key}' in tone result"

# Random noise
noise = np.random.randn(sr) * 0.01
result_noise = analyzer.analyze(noise)
print("\n[Random Noise Result]:")
pprint.pprint(result_noise)
for key in required_keys:
    assert key in result_noise, f"Missing key '{key}' in noise result"

# Empty array
empty = np.array([])
result_empty = analyzer.analyze(empty)
print("\n[Empty Array Result]:")
pprint.pprint(result_empty)
for key in required_keys:
    assert key in result_empty, f"Missing key '{key}' in empty array result"

print("\n[ALL MANUAL TESTS PASSED SUCCESSFULLY!]")
