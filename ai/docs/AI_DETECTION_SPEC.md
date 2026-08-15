# AI Detection Specification — DSP layer

**Owner:** AI 1 (DSP / Signal Processing)
**Consumer:** Backend 1 (integration), AI 2 (classification)
**Status:** DSP fields final. Classifier-dependent fields provisional until AI 2 lands.
**Schema version:** `1.0.0-dsp`

This is the charter §11 handoff document. It states exactly what the DSP layer
measures, what it does not, and how it fails.

---

## 1. Audio input requirements

| Parameter | Value | Enforced |
|---|---|---|
| Sample rate | **48000 Hz** | Yes — `< 44100` raises `ValueError` |
| Channels | Mono | Stereo is averaged |
| Chunk size | 2048 samples (~42.7 ms) | Recommended; other sizes work |
| Format | float32/float64 in [-1.0, 1.0] | Integer PCM is scaled |

48 kHz is required: the 24 kHz Nyquist ceiling must clear the 21 kHz band edge.

---

## 2. Preprocessing performed

1. DC offset removal
2. First-order pre-emphasis
3. Butterworth bandpass, order 6, **17.5–21.5 kHz**, SOS form

The filter is **stateful across chunks** (`sosfilt` with retained `zi`).
Filtering each chunk independently produces edge discontinuities at every
boundary that look like OOK keying — i.e. it manufactures false positives.

Training data is preprocessed through this same path
(`generate_training_dataset` → `preprocess_fn`), enforced by
`tests/test_training_serving_consistency.py`.

---

## 3. Features calculated

**32 features per chunk, fixed order, stable names.** Groups: energy (4),
spectral shape (6), peaks (6), tonal (4), temporal (6), statistics (6).
Full table in `dsp/INTEGRATION_GUIDE.md`.

Guarantees: exactly 32 values; never NaN or Inf; order never changes without a
schema-version bump.

```python
X = np.array(result["features"]["feature_values"]).reshape(1, -1)   # (1, 32)
```

---

## 4. Model type

**None at the DSP layer.** Modulation classification is rule-based over
spectral peak structure — deliberately interpretable, no training required.

The learned classifier is **AI 2's** deliverable. This layer supplies features
and a heuristic suspicion score so Backend 1 is not blocked in the meantime.

---

## 5. Output classes

| `pattern` | Meaning |
|---|---|
| `none` | No structured signal |
| `tone` | Steady single carrier |
| `fsk` | Two or more alternating carriers |
| `ook` | On-off keyed single carrier |
| `chirp` | Swept carrier |

`pattern` is decided by **majority vote across frames**, not from the last
frame. At 25 baud a symbol is 40 ms and a frame is 42.7 ms, so most individual
frames of an FSK signal contain one steady carrier and score as `tone`
(measured: 55 tone vs 15 fsk over a 3 s FSK signal). If the accumulated verdict
holds ≥ 2 consistent carriers, the event is reported as `fsk` regardless.

---

## 6. Confidence definition

**Two distinct numbers. Do not conflate them.**

| Field | Meaning | Range |
|---|---|---|
| `confidence` | How sure the modulation classifier is of its label | 0.0–1.0 |
| `suspicion_score` | How threat-like the signal is overall | 0.0–1.0 |

`confidence` blends the cross-frame vote share with the classifier's own
per-frame certainty. **It is not calibrated** — 0.9 does not mean 90% correct.
Calibration is AI 2's evaluation deliverable.

---

## 7. Risk methodology

```
HIGH    suspicion_score >= 0.75
MEDIUM  suspicion_score >= 0.45
LOW     otherwise, or detected == false
```

Thresholds live in `DSPPipeline.RISK_THRESHOLDS`. **These are DSP-side
defaults, not validated cutoffs.** AI 2 owns final calibration against measured
false-positive rates.

---

## 8. Reliably measured fields

| Field | Source | Notes |
|---|---|---|
| `frequency_start` / `frequency_end` | FFT peak detection | ±23 Hz bin resolution |
| `carrier_freqs` | Consistent peaks across frames | **Preserve this array** |
| `duration` | Onset/offset segmentation | Seconds |
| `snr` | Peak vs local median floor | dB |
| `chunks_analyzed` | Counter | Exact |

### Why `carrier_freqs` must survive

Charter §14 asks for a min/max range. This pipeline detects **discrete
carriers**. Collapsing `[19000, 20500]` into `{start: 19000, end: 20500}`
discards the fact that there were two separate carriers — which is precisely
the FSK signature. Backend 1 should persist the array alongside the range.

### Why `duration` cannot come from one frame

A chunk is ~42.7 ms. `to_threat_event()` must be called **periodically or at
end of capture**, not once per `process()`.

---

## 9. Experimental / provisional fields

| Field | Why |
|---|---|
| `pattern` | OOK measured at only ~70% on synthetic input (`docs/KNOWN_ISSUES.md`) |
| `confidence` | Uncalibrated |
| `risk` | Thresholds not yet validated against measured FPR |
| `suspicion_score` | Hand-tuned weights, not learned |

---

## 10. Failure behaviour

| Condition | Behaviour |
|---|---|
| Silence / no signal | `detected: false`, `risk: "LOW"`, `carrier_freqs: []` |
| No chunks processed | Valid event, `chunks_analyzed: 0`, `duration: 0.0` |
| Sample rate < 44100 | `ValueError` at construction |
| Empty chunk | Zeros returned, no exception |
| NaN/Inf input | Features finite; never propagated |

**Never raises during `process()`.** Backend 1 does not need a try/except in
the audio loop.

Dead microphone captures are rejected upstream by `dsp.recording_quality`
(< 50 LSB peak, < 64 unique values, clipping, or missing expected tone).

---

## 11. Measured performance

| Metric | Value | Method |
|---|---|---|
| Processing latency | **~1.2 ms/chunk** (max 1.44) | `tests/run_dsp_suite.py` benchmark |
| Real-time factor | ~35× faster than realtime | 1.2 ms per 42.7 ms of audio |
| Test coverage | 94 tests (70 pytest + 24 DSP) | All passing |
| Frequency accuracy | ±23 Hz | FFT bin resolution at 2048/48 kHz |
| FSK carrier recovery | 19007.8 / 20507.8 Hz for 19000 / 20500 sent | Synthetic |
| Duration accuracy | 2.987 s for 3.0 s sent | Synthetic |

### Not measured — do not claim

- Detection accuracy / precision / recall / F1 — **AI 2, not yet done**
- False-positive rate — **AI 2, not yet done**
- Real over-the-air performance — physical sweep must be re-recorded
- Hardware frequency ceiling — no valid data (see `docs/RECORDING_FAILURE_2026-08-15.md`)
- Maximum range or bitrate — never tested

**All numbers above are from synthetic signals.** Real speaker → air →
microphone performance is expected to be worse and has not been measured.

---

## 12. Contract example

```python
from dsp import DSPPipeline

pipeline = DSPPipeline()
for chunk in stream:
    pipeline.process(chunk)

event = pipeline.to_threat_event()
```

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

Shape is pinned by `tests/test_threat_event_contract.py`. Any field rename is a
breaking change requiring a `schema_version` bump and coordination with
Backend 1.
