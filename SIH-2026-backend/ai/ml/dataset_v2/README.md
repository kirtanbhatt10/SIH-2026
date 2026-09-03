# Dataset V2 — Synthetic Training Data Generator (Phase 2)

> **SOURCE = SYNTHETIC.**
>
> This dataset is generated synthetically through the project's
> signal-generation and DSP pipeline. It is intended for reproducible ML
> development and does not establish real-world / OTA detection performance.
>
> It is NOT a real recording, NOT an over-the-air capture, and NOT a
> microphone recording.

## What this is

Phase 2 of the AI/ML roadmap. It delivers the **synthetic dataset-generation
system** — the framework that Phase 5 will later execute to build and persist
the canonical `training_data_v2/` artifact.

> **Phase boundary:** The final Dataset V2 artifact is intentionally **NOT
> built in Phase 2. Dataset V2 will be generated/persisted during Phase 5.**

Phase 2 produces labeled, reproducible **32-feature vectors** extracted from
synthetic ultrasonic signals by running the **exact same DSP components** the
live detection pipeline uses:

```
dsp.SignalGenerator   →  synthetic audio waveform  (varied parameters)
        ↓
dsp.AudioPreprocessor →  real bandpass / pre-emphasis pipeline
        ↓
dsp.FeatureExtractor  →  frozen 32-feature extraction
        ↓
(features, label, rich metadata)
```

No features are fabricated. No feature vectors are invented outside this
pipeline. The Tier-2 fallback in `ai/ml/data_generation.py` is not touched
and cannot silently enter Dataset V2.

## Quick start

```bash
cd ai/ml/dataset_v2

# Generate (takes ~8 s for 2500 samples)
python build_dataset.py --samples-per-class 500 --seed 42 --no-audio

# Validate (exits 0 only if all 15 checks pass)
python validate_dataset.py --dataset-dir ../../training_data_v2

# Inspect feature distributions + example plots
python inspect_dataset.py --dataset-dir ../../training_data_v2

# Reproducibility test (same-seed identity + different-seed divergence)
python test_reproducibility.py
```

## Output structure

```
training_data_v2/
├── features.npy        (N, 32) float64 — ML-ready feature matrix
├── labels.npy          (N,)   int32   — {0=benign, 1=fsk, 2=ook, 3=chirp, 4=tone}
├── metadata.csv        one row per sample (see schema below)
├── dataset_info.json   dataset-level summary + provenance
├── label_map.json      {"0": "benign", ...}
├── README.md           local copy of synthetic-source disclaimer
└── audio/              (optional, only if --no-audio is omitted)
    ├── benign/
    ├── fsk/
    ├── ook/
    ├── chirp/
    └── tone/
```

`training_data_v2/` is gitignored (same policy as `training_data/`).
Regenerate from seed rather than committing artifacts.

## Classes

| Label | Name   | What it is |
|-------|--------|------------|
| 0     | benign | Ambient noise, broadband hiss, environmental artifacts — no structured ultrasonic communication. Three sub-variants: `ambient`, `ambient_hf_boost` (loud non-communication high-frequency energy), `white`. |
| 1     | fsk    | Frequency Shift Keying — two alternating tones encoding bits |
| 2     | ook    | On-Off Keying — single carrier toggled on/off |
| 3     | chirp  | Frequency sweeps across the detection band |
| 4     | tone   | Steady single-frequency sinusoid |

## Variation axes (what Dataset V2 varies per sample)

| Axis | FSK | OOK | Chirp | Tone | Benign |
|------|-----|-----|-------|------|--------|
| Carrier / center freq | 18.2–19.8 kHz (mark) | 18.3–21.0 kHz | 18.0–19.3 kHz (start), 20.2–21.0 kHz (end) | 18.3–21.0 kHz | n/a |
| Frequency deviation | 400–2200 Hz (space ≤ 21 kHz) | n/a | sweep range | n/a | n/a |
| Bit / baud rate | 15–90 bps | 30–70 bps | n/a | n/a | n/a |
| Chirp sweeps | n/a | n/a | 2–8 | n/a | n/a |
| Amplitude | 0.55–1.0 | 0.55–1.0 | 0.55–1.0 | 0.55–1.0 | 0.15–0.6 |
| SNR (dB) | -5 to 30 (4 tiers) | -5 to 30 | -5 to 30 | -5 to 30 | n/a or -5 to 25 (hf_boost) |
| Noise type | white / ambient | white / ambient | white / ambient | white / ambient | ambient / white |
| Crop offset (phase) | random | random | random | random | random (5-attempt loudest) |
| Source duration | 5–10× chunk | 5–10× chunk | 5–10× chunk | 5–10× chunk | 5–10× chunk |

SNR is stratified into four equally-weighted tiers: very_noisy (-5 to 2 dB),
noisy (2–10 dB), moderate (10–20 dB), clean (20–30 dB).

## Metadata schema (metadata.csv)

| Field | Type | Description |
|-------|------|-------------|
| sample_id | string | Unique ID: `dataset_v2.0_<class>_<index>` |
| label | int | Class index (0–4) |
| signal_type | string | Class name |
| sample_rate | int | Always 48000 |
| duration | float | Always ~0.04267 s (fixed chunk) |
| frequency | float/null | Carrier or center frequency |
| snr | float/null | Signal-to-noise ratio in dB |
| noise_level | string/null | SNR tier name |
| amplitude | float/null | Generation-time amplitude parameter |
| peak_amplitude | float | Post-mix peak absolute amplitude |
| noise_type | string | "white" or "ambient" |
| seed | int | Dataset-level seed |
| local_seed | int | Per-sample local RNG seed |
| global_seed | int | Per-sample global RNG seed (for SignalGenerator) |
| generator_version | string | Generator code version tag |
| dataset_version | string | Dataset version tag |
| bit_rate | float/null | Baud rate (FSK, OOK) |
| frequency_deviation | float/null | FSK freq deviation |
| start_frequency | float/null | Chirp start freq |
| end_frequency | float/null | Chirp end freq |
| duty_cycle | null | Not independently controllable (see Known Limitations) |
| chirp_sweeps | int/null | Number of chirp sweeps |
| benign_variant | string/null | "ambient" / "ambient_hf_boost" / "white" |
| source_duration_sec | float | Length of generated source waveform |
| crop_offset_sec | float | Where the 2048-sample window was cropped |
| crop_attempts | int | Retry count (0 = first try succeeded) |
| generation_group | string | Composite key for grouped train/test splits (see below) |

### generation_group design

Groups related synthetic scenarios for leakage-aware splitting:

| Class | Format | Example |
|-------|--------|---------|
| benign (ambient/white) | `benign\|<variant>\|<amp_bucket>` | `benign\|ambient\|a30` |
| benign (hf_boost) | `benign\|ambient_hf_boost\|<snr_tier>` | `benign\|ambient_hf_boost\|moderate` |
| tone / fsk / ook | `<class>\|<snr_tier>\|<freq_500Hz>` | `fsk\|noisy\|19000` |
| chirp | `chirp\|<snr_tier>\|<start>_<end>` | `chirp\|clean\|18000_20500` |

Null values are represented as empty strings in CSV.

## Reproducibility

Every sample is deterministic given (dataset_seed, class_index, sample_index):

* A `SeedSequence` derives two independent seeds per sample.
* `local_seed` drives Dataset V2's own parameter choices (frequency, SNR, amplitude, crop offset).
* `global_seed` seeds `np.random` immediately before calling `SignalGenerator`, isolating its internal random state.

```
Same (dataset_version + seed + config) → identical features, labels, metadata.
```

Verified by `test_reproducibility.py` and `tests/test_phase2_dataset_v2_generator.py`.

## Phase 5 dependency

Phase 5 will execute `build_dataset.py` to persist `training_data_v2/`.
Phase 2 only validates that the generator system is correct and ready.

## Audit findings (Task 1)

Traced from actual code, not documentation:

1. **Fabricated fallback clearly labeled.** `ai/ml/data_generation.py` already
   has a loud `RuntimeWarning` and `source='fabricated'` tag. Dataset V2
   never imports or touches it.

2. **FeatureExtractor `ultrasonic_high` default mismatch.** The class default
   is 22000 Hz but `DSPPipeline` constructs it with 21000 Hz. Dataset V2
   matches `DSPPipeline`'s real construction, not the misleading default.

3. **AudioPreprocessor squelch.** `normalize()` returns all zeros if the
   post-bandpass peak is below 0.05. This is real, frozen behavior.
   Dataset V2 handles it with a deterministic crop-retry for communication
   classes (up to 20 attempts from the same generated audio). Benign samples
   are allowed to be squelched (that IS what "no ultrasonic content" is).

4. **OOK sparse bits.** At very low baud rates (<~25 bps), a single OOK
   bit period exceeds the 42.7 ms crop window, making it likely for a
   window to land entirely in an "off" period. Fixed by setting OOK
   min baud to 30 bps.

5. **Fixed chunk size.** The feature-extraction window is always 2048
   samples (~42.7 ms). "Duration variation" is achieved via random
   crop offset from a longer source waveform, not by changing the FFT
   window (which would break train/serve consistency).

## Known limitations

* **OOK duty cycle** is not independently configurable by the existing
  `SignalGenerator.generate_ook()`. Bits are drawn uniformly random
  (~50% expected duty cycle). Recorded as null in metadata.

* **Chirp direction** is always up-sweep. The existing generator has no
  down-sweep parameter.

* **No channel effects.** No room acoustics, speaker/mic frequency
  response, multipath, or distance attenuation. This is a limitation of
  the existing `SignalGenerator` (documented in `dsp/utils.py` line 32–42).

* **Benign samples can be all-zero** after preprocessing. This is correct
  behavior (squelched noise), not a bug.

* **No NOISY_FSK / NOISY_OOK / NOISY_CHIRP sub-classes.** These are not
  separate classes — all communication samples already receive stratified
  SNR variation across four noise tiers. Creating separate "noisy" classes
  would add labels without teaching anything the SNR metadata + existing
  labels don't already capture.

## Phase 1 feature contract

**Unchanged.** The 32 features in `config.py:FEATURE_NAMES` and
`dsp/feature_extraction.py:FeatureExtractor.FEATURE_NAMES` are identical.
Dataset V2 asserts this at build time. No features were added, removed,
reordered, or renamed.

## Files in this package

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `config_v2.py` | All parameter ranges, thresholds, version tags |
| `generator.py` | Core per-sample generation (SignalGen → AudioPreproc → FeatureExtract) |
| `build_dataset.py` | CLI: builds the full dataset to disk |
| `validate_dataset.py` | CLI: 15-check quality validator |
| `inspect_dataset.py` | CLI: feature distributions + example waveform/spectrum plots |
| `test_reproducibility.py` | Same-seed identity + different-seed divergence test |
| `README.md` | This file |
