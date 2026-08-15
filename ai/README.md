# Acoustic Cybersecurity System — AI 1 (DSP / Signal Processing)

Ultrasonic signal processing for the SIH 2026 Acoustic Cybersecurity System.

Takes raw microphone audio and produces three things: a 32-feature vector for
AI 2's classifier, spectrogram frames for the dashboard, and a structured
threat event for Backend 1.

```
Microphone → Preprocess → FFT / STFT → Features → Analysis → ThreatEvent
                              │            │           │
                        Spectrogram   (AI 2 model)  (Backend 1)
                        (Frontend)
```

---

## Quick start

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for running the test suite

python -m pytest tests/ -q        # 87 automated tests
python tests/run_dsp_suite.py     # 24 module tests + latency benchmark
```

`tests/run_dsp_suite.py` needs no pytest — run it for a quick check.

```python
from dsp import DSPPipeline

pipeline = DSPPipeline()

for chunk in audio_stream:            # 2048 samples, mono, float32, 48 kHz
    result = pipeline.process(chunk)

event = pipeline.to_threat_event()    # AI → Backend contract
```

| I need… | Read |
|---|---|
| To integrate with this module | `dsp/INTEGRATION_GUIDE.md` |
| The exact contract + failure modes | `docs/AI_DETECTION_SPEC.md` |
| What's broken / unverified | `docs/KNOWN_ISSUES.md` |
| Why the physical tests were redone | `docs/RECORDING_FAILURE_2026-08-15.md` |

---

## Project structure

```
.
├── README.md                         you are here
├── requirements.txt
├── .gitignore                        ignores *.wav/*.png except docs/evidence/
│
├── dsp/                              ── the library ──
│   ├── __init__.py                   shared constants (SINGLE SOURCE OF TRUTH)
│   ├── dsp_api.py                    DSPPipeline — the only import you need
│   ├── audio_preprocessing.py        DC removal, pre-emphasis, bandpass
│   ├── feature_extraction.py         32 features per chunk
│   ├── stft.py                       explicit STFT with configurable overlap
│   ├── spectrogram_generator.py      rolling spectrogram, JSON frames, PNG
│   ├── frequency_analyzer.py         peaks, FSK/OOK/chirp/tone, suspicion
│   ├── segmentation.py               onset/offset → event durations
│   ├── recording_quality.py          rejects dead microphone captures
│   ├── utils.py                      signal generation, dataset builder
│   └── INTEGRATION_GUIDE.md          per-teammate integration instructions
│
├── tests/                            ── verification ──
│   ├── run_dsp_suite.py              24 module tests + benchmark
│   ├── test_dsp_api.py               public interface
│   ├── test_dsp_pipeline.py          end-to-end integration
│   ├── test_threat_event_contract.py AI → Backend contract shape
│   ├── test_segmentation.py          onset/offset detection
│   ├── test_stft.py                  STFT + round-trip
│   ├── test_recording_quality.py     dead-capture rejection
│   ├── test_device_selection.py      microphone selection ranking
│   ├── test_training_serving_consistency.py
│   └── manual/                       interactive demos, not run by pytest
│
├── scripts/                          ── operator tools ──
│   ├── record_wav.py                 physical test recorder (quality-gated)
│   ├── diagnose_capture.py           diagnose a failing microphone stream
│   ├── enum_devices.py               list audio devices
│   └── make_evidence.py              recordings → committable evidence
│
├── analysis/                         ── one-off analysis scripts ──
│   ├── analyze_peak_snr.py
│   ├── list_top_peaks.py
│   ├── analyze_control_tests.py
│   ├── analyze_voiceband_tests.py
│   ├── analyze_lower_ultrasonic_tests.py
│   ├── analyze_physical_tests.py
│   └── compare_hardware_response.py
│
├── demo/                             ── presentation ──
│   ├── live_demo.py                  terminal live detection demo
│   └── generate_demo_assets.py       spectrogram images for slides
│
└── docs/
    ├── AI_DETECTION_SPEC.md          charter §11 handoff to Backend 1
    ├── KNOWN_ISSUES.md               honest list of open gaps
    ├── RECORDING_FAILURE_2026-08-15.md
    ├── AI1_STATUS_AUDIT.md
    └── evidence/                     measured artifacts (committed)
```

---

## Configuration

`dsp/__init__.py` is the **single source of truth**. Do not hardcode these
anywhere else.

| Constant | Value |
|---|---|
| `SAMPLE_RATE` | 48000 Hz |
| `CHUNK_SIZE` / `N_FFT` / `HOP_LENGTH` | 2048 |
| `ULTRASONIC_LOW` / `ULTRASONIC_HIGH` | 18000 / 21000 Hz |
| `FILTER_LOW` / `FILTER_HIGH` | 17500 / 21500 Hz |

Detection band is **18–21 kHz**. Earlier documentation said 18–22 kHz in some
places and 17.5–23 kHz in others; all of it is now unified on the values above.

---

## Recording physical tests

**Verify capture works before recording anything you intend to keep.**

```bash
python scripts/diagnose_capture.py --all      # find a device that survives
python scripts/record_wav.py --list-devices
python scripts/record_wav.py --device 11
```

A capture is rejected automatically if it is dead (< 50 LSB peak), contains
only quantisation noise, is clipped, or lacks the tone named in its label.

Check files you already have:

```bash
python -m dsp.recording_quality *.wav
```

Then turn good recordings into committable evidence plus the hardware report:

```bash
python scripts/make_evidence.py --curated --input-dir .
```

This machinery exists because an entire test sweep was silently recorded as
digital silence on 15 Aug 2026 — see `docs/RECORDING_FAILURE_2026-08-15.md`.

---

## Status

**Working** — preprocessing, FFT, STFT, spectrogram, 32 features, frequency
analysis, segmentation, threat-event contract, capture quality gate.
**111 tests pass** (87 pytest + 24 module suite) at **~1.2 ms per chunk**,
roughly 35× faster than real time.

**Open** — see `docs/KNOWN_ISSUES.md`:

- OOK classification measured at only ~70% on synthetic input
- Validation is almost entirely synthetic
- **No hardware frequency-response claim is supported by data** until the
  physical sweep is re-recorded

**Owned by AI 2, not this module** — the classifier, dataset labelling, risk
threshold calibration, and false-positive evaluation.
