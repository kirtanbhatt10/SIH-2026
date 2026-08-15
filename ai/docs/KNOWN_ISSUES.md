# Known Issues — AI 1 (DSP)

Open technical debt in the DSP subsystem. Kept honest so nobody builds a claim
on something we have not actually verified.

---

## 1. OOK classification is only ~70% reliable

**Severity:** Medium — affects a headline claim
**Found:** 15 Aug 2026, while investigating a flaky test

`FrequencyAnalyzer` classifies a synthetic 50-baud OOK signal at 19 kHz
correctly in **28 of 40 runs**. Measured over 40 unseeded trials:

| Signal | Result |
|---|---|
| 19 kHz steady tone | `tone` **40/40** |
| OOK 50 baud @ 19 kHz | **`ook` 28/40**, `fsk` 9/40, `none` 3/40 |
| FSK single chunk | `tone` 40/40 (correctly never `ook`) |

The variation comes from `generate_ook()` using random bits: some bit patterns
put too few on/off transitions inside the analysis window, and the classifier
falls back to `fsk` or `none`.

**Why it matters:** `INTEGRATION_GUIDE.md` advertises *"4 modulation schemes
detected: FSK, OOK, Chirp, steady tone"* without qualification. On this
evidence OOK detection is roughly a 70% proposition on *synthetic, noise-free*
input. Real over-the-air audio will be worse.

**Do not** present modulation classification accuracy in the PPT until it is
measured properly across many trials and SNRs — that measurement is AI 2's
evaluation deliverable.

**Mitigation in place:** `tests/test_dsp_pipeline.py::test_6_regression_modulation_classification`
is now seeded (`np.random.seed(42)`) so CI is deterministic. **The seed hides
the flakiness; it does not fix it.** This entry exists so the gap is not lost.

**Likely fix:** require a minimum number of on/off transitions in the window
before committing to `ook`, and/or analyse a longer segment for OOK than the
single 2048-sample frame.

---

## 2. Tone-presence threshold was calibrated, not guessed

**Status:** Resolved 15 Aug 2026 — recorded so the number is defensible.

`recording_quality.MIN_TONE_SNR_DB` was initially set to 10 dB by intuition.
Measuring the peak-to-median ratio of **pure white noise** in a ±200 Hz band
over 40 trials:

```
mean 9.4 dB   max 11.0 dB   p99 10.9 dB
=> a 10 dB threshold fires on pure noise ~20% of the time
=> a 15 dB threshold fired 0% of the time
```

Threshold is now **15 dB**. If a judge asks "why 15?", the answer is this
measurement, not a guess.

---

## 3. STFT — RESOLVED 15 Aug 2026

`dsp/stft.py` provides `stft()` / `istft()` with configurable hop, four window
types, and optional centering. Interior reconstruction is exact (measured
relative RMS error 2.3e-16); the first and last `n_fft` samples are degraded
because fewer frames overlap there, which is inherent to overlap-add.

`HOP_LENGTH = 2048` in `dsp/__init__.py` documents the *streaming* framing
used by `SpectrogramGenerator` (one frame per chunk, no overlap). Offline
analysis should use `stft()` with a smaller hop.

Original note follows.

`SpectrogramGenerator` accumulates per-chunk FFT frames, which is functionally
an STFT with hop = chunk size. But:

- there is no `stft()` function anywhere in `dsp/`
- hop length is not configurable
- `INTEGRATION_GUIDE.md` documents `HOP_LENGTH = 512`, which **does not exist
  in the code**

The charter lists STFT as an AI 1 deliverable. Either expose a proper
configurable STFT or correct the documentation.

---

## 4. Signal segmentation — RESOLVED 15 Aug 2026

`dsp/segmentation.py` provides `segment_signal()` (offline) and
`StreamingSegmenter` (real-time), wired into `DSPPipeline.process()`.
`duration` in the threat event now comes from measured onset/offset.

Note on the noise-floor estimator: a median-based floor silently returned zero
events when the signal covered more than half the recording (the median landed
inside the tone). It now splits low/high percentiles and takes the floor from
the quiet side. Signals covering ~95%+ of a recording are still missed — there
is no quiet reference left to compare against.

Original note follows.

`grep -rn "segment" dsp/` returns nothing. The charter lists *signal
segmentation* under AI 1, and the backend contract wants a `duration` field.
Per-frame results are ~42.7 ms, so duration can only come from accumulated
verdicts until onset/offset detection exists.

---

## 5. Documented API does not match the code — RESOLVED 15 Aug 2026

`is_suspicious` / `suspicion_score` now exist as aliases;
`DSPPipeline.get_default_config()` added as the class-level call; all band
constants unified on `dsp/__init__.py` (18–21 kHz detection, 17.5–21.5 kHz
filter). `audio_preprocessing.py` default `filter_high` was 21000 while
`__init__.py` said 21500 — now both 21500.

Original note follows.

`INTEGRATION_GUIDE.md` tells Backend 1 to use:

```python
result["is_suspicious"]     # KeyError — actual key: frame_is_suspicious
result["suspicion_score"]   # KeyError — actual key: frame_suspicion_score
DSPPipeline.get_config()    # TypeError — not a classmethod
```

Frequency constants disagree in three places:

| | Guide | `dsp/__init__.py` | `audio_preprocessing.py` docstring |
|---|---|---|---|
| Upper band | 22 kHz | **21 kHz** | 23 kHz |

Unresolved as of this commit. Highest-priority fix before Backend 1 integrates.

---

## 6. No hardware feasibility data — BLOCKED ON RE-RECORDING

`scripts/make_evidence.py` now generates `docs/hardware-feasibility.md`
automatically, including the measured frequency-response table, noise floor,
limitations and SHA-256 provenance.

**It has not been run on real data.** The 15 Aug capture failure destroyed the
sweep, so no measured frequency ceiling exists. Until the sweep is re-recorded:

- no hardware frequency-response claim may appear in the report or PPT
- the answer to *"what is your microphone's usable ceiling?"* is "not yet
  measured", not a number

Run after re-recording:

```bash
python -m dsp.recording_quality *.wav          # verify captures first
python scripts/make_evidence.py --curated --input-dir .
```

---

## 7. Demo reports synthetic results — by design

`demo/live_demo.py` runs on **generated** signals, not live microphone input.
It is a presentation tool, not evidence.

Two claims were corrected on 15 Aug 2026:

- It printed `Processing speed: avg 75.22 ms/chunk`, computed as wall-clock
  uptime / chunks — which included the `time.sleep()` used for visual pacing.
  Real DSP time is ~1.2–2 ms. It now times `process()` directly and states
  that display pacing is excluded.
- It asserted *"Conventional antivirus and network monitors cannot detect this
  channel"* and *"consistent with air-gap acoustic data exfiltration"*. Neither
  was tested. Softened to what is actually supported: the signal carries no
  network traffic, and detection here was on a synthetic signal.

If asked during the demo whether this is live audio, the answer is no — the
microphone path is blocked on re-recording (issue 6).

---

## 8. All validation is on synthetic signals

Every passing test uses `SignalGenerator` output. The only real recordings that
survived the 15 Aug capture failure are `clap.wav`, `rx_1000.wav`,
`rx_1000_v2.wav`, `rx_5000_v2.wav` — see `docs/RECORDING_FAILURE_2026-08-15.md`.

A classifier validated purely on synthetic FSK may not survive real
speaker → air → microphone transmission. This is the largest technical risk in
the DSP subsystem.
