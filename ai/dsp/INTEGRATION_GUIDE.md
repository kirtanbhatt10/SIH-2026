# Acoustic Cybersecurity System — DSP Module Integration Guide

> **Author:** AI 1 (DSP / Signal Processing Lead)
> **Last Updated:** August 2026
> **Status:** ✅ Ready for integration

---

## 🏗 Architecture Overview

```
Microphone → [Backend 2: Audio Stream] → raw PCM chunks
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   DSP Pipeline   │ ← YOU ARE HERE
                                    │   (dsp_api.py)   │
                                    └──┬──────┬──────┬─┘
                                       │      │      │
                         ┌─────────────┘      │      └──────────────┐
                         ▼                    ▼                     ▼
                  32 Features          Spectrogram JSON        Analysis
                  (for AI 2)          (for Frontend 1&2)    (for Backend 1)
```

---

## 📋 Quick Start (For Every Team Member)

### Install Dependencies

```bash
pip install numpy scipy
# Optional (only for spectrogram images):
pip install matplotlib
```

### One-Line Test

```bash
python tests/run_dsp_suite.py
```

If all tests pass ✓, the module is working correctly.

---

## 👤 For Backend 1 — System Architect

### What to import

```python
from dsp import DSPPipeline

pipeline = DSPPipeline()  # uses default settings (48 kHz, 2048 FFT)
```

### Processing audio chunks

```python
# In your audio processing loop / WebSocket handler:
result = pipeline.process(raw_audio_chunk)

# result contains EVERYTHING:
result["features"]         # → dict with feature_names, feature_values, feature_dict
result["spectrogram"]      # → dict ready for WebSocket JSON
result["analysis"]         # → dict with modulation_type, suspicion_score, etc.
result["is_suspicious"]    # → bool (alias of frame_is_suspicious)
result["suspicion_score"]  # → float 0.0 – 1.0 (alias of frame_suspicion_score)
result["snr_db"]           # → float, peak SNR in dB for this chunk
result["peak_frequency_hz"]# → float, dominant frequency this chunk
result["event_active"]     # → bool, is a signal event currently open
result["event_closed"]     # → dict or None, event that just ended
result["timestamp"]        # → Unix timestamp
result["chunk_index"]      # → sequential chunk number
```

### Building the threat event for your API (charter §14)

```python
event = pipeline.to_threat_event()
```

Returns exactly the shape you persist and serve:

```json
{
  "schema_version": "1.0.0-dsp",
  "detected": true,
  "confidence": 0.89,
  "risk": "MEDIUM",
  "suspicion_score": 0.6365,
  "frequency_start": 19007.8,
  "frequency_end": 20507.8,
  "carrier_freqs": [19007.8, 20507.8],
  "duration": 2.987,
  "pattern": "fsk",
  "snr": 47.59,
  "chunks_analyzed": 70,
  "timestamp": 1786778426.16
}
```

**Call this periodically or at end of capture — NOT once per `process()`.**
It is built from the accumulated verdict. A single chunk is ~42.7 ms, so
`duration` is meaningless per-frame.

**Two different confidences.** `confidence` is how sure the modulation
classifier is of its label. `suspicion_score` is how threat-like the signal is
overall. They are not interchangeable.

**Keep `carrier_freqs`.** Collapsing `[19000, 20500]` into `frequency_start/end`
discards the two-carrier structure that *is* the FSK signature. Persist the
array.

**Risk bands** (DSP-side defaults; AI 2 owns final calibration):
`HIGH >= 0.75`, `MEDIUM >= 0.45`, else `LOW`.

### Getting accumulated verdict (after many chunks)

```python
verdict = pipeline.get_verdict()
# verdict["is_threat"]             → bool
# verdict["average_suspicion"]     → float
# verdict["consistent_carriers"]   → [19000.0, 20500.0]
# verdict["threat_duration_sec"]   → float
```

### API config for your setup

```python
config = DSPPipeline.get_default_config()   # class-level, no instance needed
# or: DSPPipeline().get_config()
# {
#   "sample_rate": 48000,
#   "n_fft": 2048,
#   "recommended_stream_chunk_size": 2048,
#   "ultrasonic_band_hz": [18000.0, 21000.0],
#   "filter_band_hz": [17500.0, 21500.0],
#   "num_features": 32,
#   "input_channels": 1,
#   "preferred_input_dtype": "float32",
#   "normalized_range": [-1.0, 1.0],
#   "preprocessor_output_dtype": "float32"
# }
```

---

## 👤 For Backend 2 — Signal / Audio Engineer

### Audio input requirements

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample Rate | **48000 Hz** | Nyquist ceiling at 24 kHz covers our 18–21 kHz band |
| Chunk Size | **2048 samples** | Matches FFT window for best frequency resolution |
| Format | **float32 or float64** | Normalized to [-1.0, 1.0] |
| Channels | **Mono** | If stereo, average both channels first |

### Sending audio to DSP

```python
from dsp import DSPPipeline

pipeline = DSPPipeline(sample_rate=48000)

# Your audio callback:
def on_audio_chunk(raw_pcm_buffer):
    # raw_pcm_buffer: np.ndarray, shape (2048,), dtype float32/64
    result = pipeline.process(raw_pcm_buffer)
    return result
```

### Attack Simulator

```python
# Generate attack signals for the simulator UI
signal = pipeline.generate_attack_signal("fsk", duration_sec=5.0,
    freq_mark=19000, freq_space=20500, baud_rate=25)
pipeline.save_wav(signal, "attack.wav")

# Available types: "fsk", "ook", "chirp", "tone"
# Add noise: snr_db=10 (lower = harder to detect)
```

---

## 👤 For AI 2 — Detection / ML Lead

### THIS IS YOUR MOST IMPORTANT SECTION

### Getting the 32-feature vector

```python
from dsp import DSPPipeline

pipeline = DSPPipeline()
result = pipeline.process(audio_chunk)

# For your model:
feature_vector = result["features"]["feature_values"]   # list of 32 floats
feature_names  = result["features"]["feature_names"]     # list of 32 strings
feature_dict   = result["features"]["feature_dict"]      # {"name": value, ...}

# As numpy array for sklearn/torch:
import numpy as np
X = np.array(feature_vector).reshape(1, -1)  # shape (1, 32)
```

### Feature list (32 features, fixed order)

| # | Name | Group | What it measures |
|---|------|-------|------------------|
| 1 | ultrasonic_energy | Energy | Total power in 18–21 kHz |
| 2 | total_energy | Energy | Total power across full spectrum |
| 3 | energy_ratio | Energy | ultrasonic / total (key indicator) |
| 4 | energy_db | Energy | Ultrasonic energy in dB |
| 5 | spectral_centroid | Shape | Center of mass frequency |
| 6 | spectral_bandwidth | Shape | Spread around centroid |
| 7 | spectral_flatness | Shape | Low = tonal (signal), High = noise |
| 8 | spectral_rolloff | Shape | Freq below which 85% energy lies |
| 9 | spectral_skewness | Shape | Asymmetry of spectrum |
| 10 | spectral_kurtosis | Shape | Peakedness of spectrum |
| 11 | peak_frequency | Peak | Dominant frequency in Hz |
| 12 | peak_magnitude | Peak | Magnitude of dominant peak |
| 13 | peak_prominence | Peak | Peak height above local baseline |
| 14 | num_peaks | Peak | Number of significant peaks |
| 15 | peak_spacing_mean | Peak | Mean spacing between peaks (FSK!) |
| 16 | peak_spacing_std | Peak | Regularity of peak spacing |
| 17 | tonal_prominence | Tonal | Peak-to-mean ratio |
| 18 | harmonic_ratio | Tonal | Top-3 peak energy / total |
| 19 | crest_factor | Tonal | Peak / RMS ratio |
| 20 | zero_crossing_rate | Tonal | Rate of sign changes |
| 21 | rms_amplitude | Temporal | Root-mean-square amplitude |
| 22 | amplitude_envelope_std | Temporal | Variability of envelope |
| 23 | onset_strength | Temporal | Max energy jump between frames |
| 24 | temporal_flatness | Temporal | Consistency of energy over time |
| 25 | duty_cycle | Temporal | Fraction of time signal is "on" |
| 26 | bit_rate_estimate | Temporal | Estimated symbol rate |
| 27 | mean_magnitude | Stats | Mean of ultrasonic magnitudes |
| 28 | std_magnitude | Stats | Std dev of magnitudes |
| 29 | max_magnitude | Stats | Maximum magnitude |
| 30 | min_magnitude | Stats | Minimum magnitude |
| 31 | magnitude_range | Stats | max - min |
| 32 | coefficient_of_variation | Stats | std / mean |

### Generate training dataset (ONE COMMAND)

```python
from dsp import DSPPipeline

pipeline = DSPPipeline()
features, labels = pipeline.generate_training_dataset(
    output_dir="training_data",
    samples_per_class=1000,    # 5000 total samples (5 classes × 1000)
)

# Output files:
#   training_data/features.npy    — shape (5000, 32) float64
#   training_data/labels.npy      — shape (5000,)    int32
#   training_data/label_map.json  — {"0": "benign", "1": "fsk", "2": "ook", ...}
#   training_data/metadata.json   — generation parameters
```

### Label map

| Label | Class | Description |
|-------|-------|-------------|
| 0 | benign | Ambient noise, no attack |
| 1 | fsk | Frequency Shift Keying |
| 2 | ook | On-Off Keying |
| 3 | chirp | Frequency sweep |
| 4 | tone | Steady ultrasonic tone |

### Quick training example (sklearn)

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load generated dataset
X = np.load("training_data/features.npy")
y = np.load("training_data/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 👤 For Frontend 1 — Security Dashboard

### Spectrogram data format (WebSocket JSON)

Every chunk produces this JSON payload:

```json
{
    "timestamp": 1720000000.123,
    "freq_axis": [18000.0, 18046.9, ..., 21000.0],
    "magnitudes_db": [-80.2, -45.1, ..., -70.5],
    "magnitudes_norm": [0.0, 0.6, ..., 0.1],
    "peak_freq_hz": 19500.0,
    "peak_magnitude_db": -12.3,
    "peak_snr_db": 24.7,
    "ultrasonic_active": true
}
```

| Field | Type | Use |
|-------|------|-----|
| `freq_axis` | float[] (64 values) | X-axis labels for frequency chart |
| `magnitudes_norm` | float[] (64 values, 0–1) | Bar heights / heatmap values |
| `magnitudes_db` | float[] (64 values) | Actual dB values for tooltips |
| `peak_freq_hz` | float | Highlight the dominant frequency |
| `peak_snr_db` | float | Signal strength indicator (charter §15) |
| `ultrasonic_active` | bool | Show/hide alert indicator |

### Risk indicator data

From the main `result`:

```json
{
    "is_suspicious": true,
    "suspicion_score": 0.823,
    "analysis": {
        "modulation_type": "fsk",
        "confidence": 0.91,
        "carrier_freqs": [19000.0, 20500.0]
    }
}
```

---

## 👤 For Frontend 2 — Attack Simulator UI

### Triggering attacks from the UI

Call Backend's API endpoint, which internally uses:

```python
signal = pipeline.generate_attack_signal(
    attack_type="fsk",       # "fsk", "ook", "chirp", "tone"
    duration_sec=5.0,
    snr_db=15,               # optional noise mixing
    freq_mark=19000,          # FSK-specific
    freq_space=20500,         # FSK-specific
    baud_rate=25,
)
```

### Available attack parameters

| Attack | Parameters |
|--------|-----------|
| `fsk` | `freq_mark`, `freq_space`, `baud_rate` |
| `ook` | `carrier_freq`, `baud_rate` |
| `chirp` | `freq_start`, `freq_end`, `num_sweeps` |
| `tone` | `frequency`, `amplitude` |

---

## 👤 For PPT Lead — Presentation

### Demo script

```bash
# Run the live terminal demo:
python demo/live_demo.py --attack fsk --duration 8

# Generate pretty spectrogram images for slides:
python demo/generate_demo_assets.py
# → Output in demo/assets/
```

### Key talking points

1. **18 kHz – 21 kHz** — above human hearing, below microphone/speaker limits
2. **32 spectral features** extracted per audio chunk in real time
3. **< 10 ms processing** per chunk — runs on phones and laptops
4. **4 modulation schemes** targeted: FSK, OOK, Chirp, steady tone
   (⚠ OOK is only ~70% reliable on synthetic input — see `docs/KNOWN_ISSUES.md`.
    Do not quote a detection-accuracy figure until AI 2 has measured it.)
5. **Zero network traffic** — this attack is invisible to every firewall and antivirus

---

## ⚙ Configuration Reference

| Constant | Value | Defined In |
|----------|-------|-----------|
| `SAMPLE_RATE` | 48000 Hz | `dsp/__init__.py` |
| `CHUNK_SIZE` | 2048 samples | `dsp/__init__.py` |
| `N_FFT` | 2048 | `dsp/__init__.py` |
| `HOP_LENGTH` | 2048 | `dsp/__init__.py` (= CHUNK_SIZE, frames do not overlap) |
| `ULTRASONIC_LOW` | 18000 Hz | `dsp/__init__.py` |
| `ULTRASONIC_HIGH` | 21000 Hz | `dsp/__init__.py` |
| `FILTER_LOW` | 17500 Hz | `dsp/__init__.py` |
| `FILTER_HIGH` | 21500 Hz | `dsp/__init__.py` |

---

## 🧪 Running Tests

```bash
# Full test suite with colored output:
python tests/run_dsp_suite.py

# With pytest (if installed):
python -m pytest tests/run_dsp_suite.py -v
```

Expected: **All 24 tests pass**, processing benchmark < 10 ms/chunk.

Full pytest suite (70 tests):

```bash
python -m pytest tests/ -q
```
