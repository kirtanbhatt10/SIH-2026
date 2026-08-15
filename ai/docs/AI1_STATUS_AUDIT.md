# AI 1 — DSP / Signal Processing Lead — Scoped Status Audit

> Audit date: 15 August 2026
> Repo: `Kashish-patodiya/ai` @ `94f9a06`
> Scope: **AI 1 ownership only**, per the AI-team charter (§2) and team allocation
> Method: full source read + live execution of both test suites

**Supersedes the earlier draft of this file.** That version graded you against the *whole* AI-team deliverable list. Under the charter, several of those items belong to AI 2. Corrected below.

---

## 1. Scope correction — what is NOT yours

The charter splits AI into two roles. These are **AI 2's**, not yours:

| Item | Owner |
|---|---|
| Dataset, labels, metadata schema | **AI 2** |
| Baseline classifier + model selection | **AI 2** |
| Threat scoring / risk methodology | **AI 2** |
| Evaluation (accuracy/precision/recall/F1/FPR) | **AI 2** |
| False-positive analysis | **AI 2** |
| Anomaly detection | **AI 2** |

So my earlier "🔴 missing baseline classifier with measured metrics" was **wrongly charged to you**. You are not behind on that — AI 2 is, and you have already handed them the tooling to do it.

Your actual charter ownership (§2, AI Member 1):

> Audio acquisition · Sampling · Filtering · FFT · STFT · Spectrogram · Frequency analysis · **SNR** · **Signal segmentation** · Feature extraction · Signal visualization

Plus the team-allocation line: *FFT/STFT, band-pass filtering, frequency detection, spectrogram, signal features.*

---

## 2. Scorecard against **your** charter items

| # | AI 1 charter item | Status | Evidence |
|---|---|---|---|
| 1 | Audio acquisition | ✅ Done | `scripts/record_wav.py` — device selection, filters out virtual/webcam mics, 48 kHz mono float32 |
| 2 | Sampling / rate policy | ✅ Done | 48 kHz enforced; `sample_rate < 44100` raises with a clear message |
| 3 | Band-pass filtering | ✅ Done | Butterworth SOS order 6, **stateful across chunks** (correct for streaming), DC removal, pre-emphasis |
| 4 | FFT | ✅ Done | Hann-windowed `rfft` throughout |
| 5 | **STFT** | ⚠️ **Partial** | No `stft` anywhere in `dsp/`. You build a *rolling per-chunk FFT history*, which is functionally an STFT with hop = chunk size — but it isn't named, configurable, or exposed as one. `HOP_LENGTH` is documented in the guide and **does not exist in code** |
| 6 | Spectrogram | ✅ Done | `SpectrogramGenerator` — rolling history, JSON frames, PNG export |
| 7 | Frequency detection | ✅ Done | `FrequencyAnalyzer` — peak finding, carrier extraction, FSK/OOK/chirp/tone classification |
| 8 | **SNR** | ⚠️ **Partial** | `peak_snr_db` computed in `spectrogram_generator.py:263`. But: **not in the 32-feature vector**, **not documented in the guide**, and the backend contract (§14) explicitly asks for an `snr` field |
| 9 | **Signal segmentation** | ❌ **Missing** | `grep -rn "segment" dsp/` → **zero hits**. No onset/offset detection, no "signal starts here, ends here" boundary logic |
| 10 | Feature extraction | ✅ Done | 32 features, fixed order, no NaN/Inf, stable names |
| 11 | Signal visualization | ⚠️ Code exists, output discarded | 4 `analysis/` analysers + `demo/generate_demo_assets.py` all `savefig()` → PNG → **`.gitignore`d** |
| 12 | **Hardware frequency measurement** (charter §3, §10) | ⚠️ **Code only** | `analysis/hardware_frequency_response.py`, `analysis/compare_hardware_response.py`, `enum_devices.py` exist and sweep 18.0–19.0 kHz. **No committed report, no numbers in repo** |

**7 done · 4 partial · 1 missing.**

---

## 3. What you did well

Verified by execution, not by reading:

| Check | Result |
|---|---|
| `python tests/run_dsp_suite.py` | ✅ **24/24 passed** |
| `python -m pytest tests/ -q` | ✅ **12/12 passed** |
| Processing latency | ✅ **avg 1.23 ms/chunk** (charter target: real-time) |
| Feature vector | ✅ exactly 32, no NaN/Inf, stable ordering |

Three things worth specific credit:

**Train/serve consistency is enforced by a test.** `generate_training_dataset()` runs examples through the same `AudioPreprocessor` used at runtime via `preprocess_fn`, and `tests/test_training_serving_consistency.py` asserts it. This is the #1 silent killer of hackathon ML demos and you pre-empted it. AI 2 inherits a correct dataset because of this.

**Stateful streaming filter.** You used `sosfilt` with retained state across chunks rather than re-filtering each chunk independently. Naïve implementations produce edge artifacts at every chunk boundary that look exactly like OOK keying — i.e. they manufacture false positives. You avoided that.

**You already satisfy the charter's hardware-fallback requirement in code.** §3 says: *if the hardware can't do ultrasonic, prove the pipeline at a supported test frequency.* I verified your pipeline reconfigures cleanly:

```python
DSPPipeline(ultrasonic_low=3000, ultrasonic_high=5000,
            filter_low=2500, filter_high=5500)
# → works, peak detected at 3984 Hz
```

Your `analysis/analyze_voiceband_tests.py` (1–5 kHz) and `analyze_lower_ultrasonic_tests.py` (15–17.5 kHz) show you actually ran that fallback strategy. **That is exactly the discipline the charter asks for — and there is no written record of it.** See §5.

---

## 4. 🔴 Bugs — fix today

Verified by running them.

### 4.1 The guide documents result keys that don't exist — INTEGRATION BREAKER

Guide tells Backend 1:
```python
result["is_suspicious"]     # → KeyError
result["suspicion_score"]   # → KeyError
```
Actual: `frame_is_suspicious`, `frame_suspicion_score`. Confirmed `'is_suspicious' in result` → **False**.

Backend 1 copies your documented snippet, it crashes, and it reads as *your* module being broken.

### 4.2 `DSPPipeline.get_config()` documented as classmethod, isn't
```
TypeError: get_config() missing 1 required positional argument: 'self'
```
Its documented return value is stale too — advertises `chunk_size`/`audio_format`; really returns `recommended_stream_chunk_size`, `ultrasonic_band_hz`, `filter_band_hz`, `input_channels`, `preferred_input_dtype`, `normalized_range`, `preprocessor_output_dtype`.

### 4.3 Frequency constants contradict the code

| Constant | Guide | Actual `dsp/__init__.py` |
|---|---|---|
| `ULTRASONIC_HIGH` | 22000 | **21000** |
| `FILTER_HIGH` | 23000 | **21500** |
| `HOP_LENGTH` | 512 | **doesn't exist** |

`audio_preprocessing.py`'s own docstring says "17.5 kHz – 23 kHz" — a **third** number. Backend 2 builds a 21.5 kHz simulator and detection silently fails.

### 4.4 `peak_snr_db` is undocumented
You compute it; the guide never mentions it. Frontend 1 wants "signal strength" (§15) and Backend wants `snr` (§14) — you're already producing it and nobody knows.

### 4.5 Test count wrong
Guide says "All 22 tests pass". Actual: **24**.

### 4.6 `requirements.txt` incomplete
`scripts/record_wav.py` and `scripts/enum_devices.py` import `sounddevice`; not listed. Charter §5 names it in the stack. Add `sounddevice>=0.4.6`.

### 4.7 `dsp/run_test.py` bare import (file since removed)
`from feature_extraction import ...` only resolves via script-dir injection; breaks under pytest from root. Fix or delete (superseded by `test_dsp.py`).

---

## 5. ⚠️ Your biggest real gap: evidence is invisible

`.gitignore` excludes `*.wav` and `*.png`. Every analyser you wrote outputs PNGs. Net result:

**You have run physical hardware experiments across voiceband, 15–17.5 kHz, and 18–19 kHz — and the repository contains zero record of what you found.**

The charter is unusually blunt about this:

> §3 — *"Report experimentally measured limits."*
> §13 — *"Do NOT claim theoretical maximum data rates as your measured performance."*
> §10 — AI 1 must have *"Hardware frequency capability measurement."*

Right now the PPT lead has nothing measurable to cite from you, and if a judge asks *"what's your mic's actual usable ceiling?"* the answer exists only on your local disk.

**Fix:** each analyser should also emit a committed `.json`/`.md` of numeric findings (peak freq, peak_snr_db, noise floor, pass/fail per test frequency), and force-add the 3–4 decisive spectrograms:
```bash
git add -f docs/evidence/*.png
```

---

## 6. 🔴 The contract gap that is genuinely yours

Charter §14 specifies the AI→Backend payload. Compare:

**Charter wants:**
```json
{ "detected": true, "confidence": 0.94, "risk": "HIGH",
  "frequency_start": 19800, "frequency_end": 21200,
  "duration": 3.2, "pattern": "communication_like", "snr": 18.4 }
```

**You emit:**
```json
{ "frame_is_suspicious": true, "frame_suspicion_score": 0.82,
  "analysis": { "modulation_type": "fsk", "confidence": 0.91,
                "carrier_freqs": [19000.0, 20500.0],
                "energy_ratio": ..., "estimated_baud": ... } }
```

Split by owner:

- `risk`, and the score→LOW/MEDIUM/HIGH thresholds → **AI 2's call**, not yours
- `snr` → **yours.** You have `peak_snr_db` but never surface it in `process()`'s top level or the feature vector
- `frequency_start` / `frequency_end` → **yours.** You emit `carrier_freqs` as discrete peaks. Note collapsing `[19000, 20500]` into a min/max range **destroys the two-distinct-carriers evidence that is the FSK signature**. Push back on that field shape — propose keeping `carrier_freqs` alongside
- `duration` → **yours, architecturally.** Per-frame is ~42.7 ms; duration only exists in `get_verdict()["threat_duration_sec"]`. **Therefore backend events must be built from the accumulated verdict, not a single frame.** Nobody has been told this
- `pattern: "communication_like"` vs your `modulation_type: "fsk"` → agree exact enum strings with AI 2 + Backend 1

Also: two different confidences exist (`analysis.confidence` = modulation-classifier certainty; `frame_suspicion_score` = threat heuristic). They are **not interchangeable** and the guide doesn't distinguish them. That's a documentation duty on you.

---

## 7. Action plan

### P0 — today (~1 hour, unblocks Backend 1 + Frontend 1)
1. Fix §4.1 and §4.2 — documented API must match code.
2. Reconcile the 18–21 / 18–22 / 17.5–23 kHz mess (§4.3). One number, everywhere.
3. Document `peak_snr_db` and surface SNR at the top level of `process()` (§4.4).

### P1 — this week (your remaining charter items)
4. **Signal segmentation** (§2 gap #9) — onset/offset detection so a "signal event" has real start/end and a measurable `duration`. This is your only wholly-missing charter item, and it's the prerequisite for the `duration` field.
5. **Commit hardware feasibility findings** as `docs/hardware-feasibility.md` with the numbers `analysis/hardware_frequency_response.py` already prints — measured usable ceiling, per-frequency SNR, chosen MVP band, stated limitation. Charter §10 deliverable.
6. **Commit spectrogram evidence** (§5), force-added.
7. **Name the STFT properly** — expose hop length as a real parameter, or drop `HOP_LENGTH` from the doc.

### P2 — before Day 4 integration
8. Write your half of the AI→Backend spec: input requirements, preprocessing performed, features produced, SNR/frequency/duration semantics, and **failure behaviour** (silence, clipping, wrong sample rate). Hand AI 2 the rest.
9. Add `ai/README.md`; rename `a.py`/`i.py` to meaningful names; add `sounddevice` to requirements.
10. Consolidate the split test dirs (`dsp/test_*.py` vs `tests/`).

---

## 8. Bottom line

**Did you do your job?** Against **your** charter — largely yes. 7 of 12 items complete, DSP core is clean and genuinely well-engineered, 36 tests green, 1.23 ms/chunk, and you shipped the callable inference interface (`DSPPipeline.process()`) ahead of the roadmap so AI 2 and Backend 1 aren't blocked on you.

**What's actually left, all of it yours:**

- 🔴 Docs that lie about the API (`is_suspicious`, `get_config()`, three different band definitions) — will break on first contact with Backend 1
- 🔴 **Signal segmentation** — the one charter item with no code at all
- 🔴 Hardware feasibility findings committed as measured numbers — charter §10, and your Day-1 deliverable
- 🟠 SNR plumbed through to the contract and the feature vector
- 🟠 Spectrogram evidence surviving `.gitignore`

**Not your problem, stop worrying about it:** the classifier, the dataset labelling strategy, risk thresholds, and the FPR evaluation. That's AI 2. You've already given them clean features and a consistency guarantee — which is more than they were owed at this stage.
