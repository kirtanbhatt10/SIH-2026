# Known Issues — AI 1 (DSP)

Open technical debt in the DSP subsystem. Kept honest so nobody builds a claim
on something we have not actually verified.

---

## 1. OOK classification — improved, still imperfect

**Severity:** Low-Medium
**Updated:** 15 Aug 2026 — `transitions >= 2` relaxed to `>= 1`

A single analysis window covers only a few bit periods, so many genuine OOK
windows contain exactly one ON/OFF transition and were being discarded as
`tone` or `none`.

**Window size matters.** The pipeline analyses **8192 samples**
(`test_dsp_pipeline` passes 8192), not a single 2048 frame. Measuring on one
frame understates performance badly:

| Window | `transitions >= 2` (old) | `transitions >= 1` (new) |
|---|---|---|
| **8192 (pipeline)** | 27/40 — 68% | **30/40 — 75%** |
| 2048 (single frame) | 15/40 — 38% | 20/40 — 50% |

*40 unseeded trials, clean 50-baud OOK @ 19 kHz.*

**No false-positive cost**, verified after the change:

| Check | Result |
|---|---|
| Ambient noise, per-frame | 100/100 `none` |
| Ambient noise, final verdict | 0/30 false threats |
| Speech-like signal, final verdict | 0/20 false threats |
| Steady tone | 40/40 still `tone` — no regression |

Remaining ~25% are windows where the carrier occupies too little of the window
to form a strong FFT peak (duty far from 0.5). Fixing those needs multi-frame
transition counting — **follow-up, not done.**

**Attempted fix, 18 Aug — did not work, recorded so nobody repeats it.**

The Phase 2/3 plan suggested two changes: a longer analysis window, and
rejecting OOK sidebands that masquerade as FSK. Both were measured.

*Longer window* — helps, but plateaus well below target:

| Window | OOK correct |
|---|---|
| 2048 (42.7 ms) | 22/40 |
| 4096 | 28/40 |
| **8192 (170 ms)** | **33/40 — best** |
| 16384 | 26/40 |
| 24000 | 30/40 |

Longer is not monotonically better: past ~170 ms the window spans so many
symbols that the envelope averages out and the keying signature weakens.

*Sideband rejection* — the diagnosis was right, the fix was not. OOK at
50 baud produces sidebands at carrier ± n·baud, and 4/40 misreads were `fsk`
with an identical 586 Hz separation, confirming they were keying artefacts
rather than two carriers. But adding an envelope-plus-symmetry check to
reclassify them made things **worse**: OOK fell from 33/40 to 23/40, because
genuinely-keyed OOK frames that were already correct got rerouted through the
new branch and rejected as `none` (12/40).

Reverted. Reaching >90% needs a proper cepstral or autocorrelation-based
keying detector, not another threshold on the existing peak logic. That is a
larger piece of work than the remaining schedule allows.

**For the PPT:** OOK detection is ~75% on clean synthetic input. Do not quote a
single accuracy figure for "modulation detection" across all four schemes —
tone and FSK are far more reliable than OOK. Real over-the-air performance is
unmeasured.

Determinism: `tests/test_dsp_pipeline.py::test_6...` is seeded so CI is stable.
The seed hides run-to-run variance; it does not remove it.

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
