# Incident: physical test recordings contained no audio

**Date found:** 15 August 2026
**Found by:** AI 1 (DSP) during evidence curation
**Severity:** High — these files were nearly published as measured hardware evidence
**Status:** Root cause not yet confirmed; detection guard shipped

---

## Summary

Nearly every `rx_*.wav` in the physical test set contains **no audio**. The files
are float32 WAVs whose peak amplitude is 1–2 LSB — the smallest non-zero step a
16-bit converter can produce — with only 3–5 distinct sample values across 10–12
seconds. The transmitted tone is absent, not merely quiet.

The microphone hardware is fine: `clap.wav` from the same machine measured
8064 LSB.

---

## Evidence

1 LSB = 1/32768 = 3.0518e-05.

| File | Peak | LSB | Unique values | Verdict |
|---|---|---|---|---|
| `clap.wav` | 2.46e-01 | **8064** | 7364 | valid |
| `rx_1000.wav` | 1.70e-02 | **557** | 651 | valid |
| `rx_5000_v2.wav` | 3.57e-03 | 117 | 213 | valid |
| `rx_1000_v2.wav` | 3.30e-03 | 108 | 163 | valid |
| `rx_4000_v2.wav` | 6.10e-05 | **2** | 4 | dead |
| `rx_17000.wav` | 6.10e-05 | **2** | 4 | dead |
| `rx_18000.wav` | 6.10e-05 | **2** | 5 | dead |
| `rx_18200.wav` | 3.05e-05 | **1** | 3 | dead |
| `baseline_quiet.wav` | 3.05e-05 | **1** | 3 | dead |

`rx_18000.wav`, 576 000 samples over 12 s, entire content:

```
0.000e+00   443895 samples
-3.052e-05   66122
 3.052e-05   65977
 6.104e-05       5
-6.104e-05       1
```

**The tone is not there.** In `rx_4000_v2.wav` the 4 kHz bin sits at `4.5e-09`
while the file's own maximum is at 375 Hz. Same for `rx_18000.wav` at 18 kHz.
These are not quiet recordings of a tone; they are recordings of nothing.

---

## Why it was nearly missed

`scripts/record_wav.py` computed peak and RMS, then printed `[SUCCESS]` **unconditionally**.
A capture with peak = 1 LSB reported success and wrote a plausible-looking file.

Worse, analysing these files produces *impressive-looking* numbers. With a
near-zero noise floor, `20*log10(peak/floor)` explodes:

```
rx_18200.wav   "SNR = 108.85 dB"      <- from 3 distinct sample values
rx_18400.wav   "SNR = 107.05 dB"
```

The same script also reported a peak at 18035 Hz for a file where 18200 Hz was
transmitted — it was locking onto quantisation noise.

Had these been published, charter §13 (*do not present unmeasured values as
results*) would have been violated, and any judge opening the WAV in Audacity
would have seen a flat line.

Separately: the `*hz.wav` files show ~232 dB SNR because they are the
**synthesised transmit files**, not recordings. Real-world SNR cannot be 232 dB.
Transmit and receive files must never be mixed in the same results table.

---

## The spectrograms confirm it visually

The committed PNGs were generated *from* these dead recordings, so they are
pictures of nothing.

`hardware_test_18000.png` (from `rx_18000.wav`) shows a uniform dither texture
across the whole 17–21 kHz band at **−125 to −150 dB**, with no horizontal line
at 18 kHz. A received tone would be an unmistakable bright horizontal stripe.

More revealing is `hardware_voiceband_1000.png` (from `rx_1000_v2.wav`, one of
the *valid* files): a broadband burst from ~0.8 s to ~2.4 s with a clear 1 kHz
line — and then **the recording goes flat and stays flat for the remaining
7.6 seconds.** Not quiet: featureless.

## Root cause: the capture stream dies ~1–2 s in

Measuring per-0.1 s block activity (peak > 20 LSB) across the set:

| File | Duration | Seconds with audio | Last activity |
|---|---|---|---|
| `clap.wav` | 5.0 s | 1.3 s | 4.0 s |
| `rx_1000_v2.wav` | 10.0 s | 0.6 s | **2.0 s** |
| `rx_1000.wav` | 10.0 s | 0.3 s | **1.6 s** |
| `rx_5000_v2.wav` | 10.0 s | 0.1 s | **1.5 s** |
| `rx_4000_v2.wav` | 10.0 s | 0.0 s | never |
| `rx_15000.wav` | 12.0 s | 0.0 s | never |
| `rx_18000.wav` | 12.0 s | 0.0 s | never |
| `baseline_quiet.wav` | 12.0 s | 0.0 s | never |

Per-0.5 s peaks for `rx_1000.wav` (LSB):

```
0.0s:1   0.5s:557   1.0s:7   1.5s:58   2.0s:7   2.5s:1   3.0s:2  ...  9.0s:1
```

The decisive detail: **ambient room noise disappears too.** A real microphone in
a real room never returns 1 LSB for eight consecutive seconds — even a silent
room produces a noise floor. The stream stops delivering samples and the buffer
is filled with near-zeros.

This reframes the problem. It is not "the mic can't hear 18 kHz" and not "the
room was quiet". **The capture stream dies roughly 1–2 seconds after it starts.**
The higher-frequency files show nothing at all simply because by the time the
tone played, the stream was already dead.

`clap.wav` survived because the clap happened at 3.0 s during a session where
the stream stayed alive — the exception that proves the mic hardware is fine.

## ROOT CAUSE CONFIRMED — sample-rate mismatch on a legacy host API

Measured with `scripts/diagnose_capture.py` on the affected laptop:

| Device | Host API | Native | Result |
|---|---|---|---|
| `[18]` Microphone Array (Senary Audio capture) | WDM-KS | **48000** | **8.02 s captured, audio throughout** |
| `[22]` Microphone (Senary Audio capture) | WDM-KS | 48000 | `PaErrorCode -9996` invalid device |
| `[0]` Sound Mapper | MME | 44100 | flat 1 LSB — digital silence |
| `[1]` Iriun Webcam | MME | 44100 | flat 1 LSB — digital silence |

Device 18, 8-second run, per-second peak in LSB:

```
0s 163   1s 175   2s 207   3s 182   4s 11209   5s 14516   6s 17844   7s 16094
```

Room noise floor ~163–207 LSB with claps at 11 000–17 800. Healthy capture.

**The cause:** `select_input_device()` matched on device *name* only and
returned the first hit, which on this machine is:

```
[2] Microphone Array (Senary Audio)   MME   native 44100 Hz
```

`scripts/record_wav.py` requests **48000 Hz**. Windows accepted the mismatched request
via MME, resampled, and the stream stalled 1–2 seconds in — then delivered
digital silence for the remainder. Because `[SUCCESS]` printed unconditionally,
every recording looked fine.

The three Senary Audio entries at 44100 Hz (MME `[2]`, DirectSound `[7]`) and
the ones at 48000 Hz (WASAPI `[11]`, WDM-KS `[18]`) are the *same physical
microphone* exposed through different host APIs. Only the native-48 kHz paths
work reliably.

**Fix:** `select_input_device()` now ranks by
`native rate match > host API (WASAPI > WDM-KS > DirectSound > MME) > known
hardware name`, and warns when no device runs natively at `SAMPLE_RATE`.
On this laptop it now selects `[11]` (WASAPI, 48 kHz) instead of `[2]`.

`--device N` forces a specific index; `--list-devices` prints the table.
Ranking is locked down by `tests/test_device_selection.py` using this exact
device list.

---

## Earlier suspicions (superseded, kept for the record)

1. **`sd.rec()` buffer/stream failure** — likely an underrun or a device timeout
   that `sounddevice` does not raise. `sd.rec()` pre-allocates and returns
   silently even if the stream stalls. Check `sd.CallbackFlags` / use
   `InputStream` with an explicit status callback to catch this.
2. **Sample-rate mismatch.** If the device's native rate is 44 100 Hz and we
   force 48 000 Hz, some Windows WDM/MME drivers accept the call and then stall.
   `enum_devices.py` prints `default_samplerate` per device — compare it.
3. **Host API.** MME and WDM-KS are less reliable than WASAPI on Windows.
   Try WASAPI explicitly.
4. **Exclusive-mode conflict** — another app grabbing the device mid-session.

Wrong-device and permissions are now *unlikely*: those produce zeros for the
whole duration, not 1–2 s of genuine audio followed by death.

---

## Fix shipped

**`dsp/recording_quality.py`** — `check_recording()` rejects a capture when:

| Check | Threshold |
|---|---|
| Dead capture | peak < 50 LSB |
| Quantisation-only content | < 64 unique sample values |
| Digital silence | < 1% non-zero samples |
| Clipping | > 0.01% samples at full scale |
| Tone absent | tone SNR < 10 dB (when expected frequency known) |

**`scripts/record_wav.py`** now runs this gate **before writing** and refuses to save a
failing capture, printing troubleshooting steps. `--force` overrides with an
explicit warning not to use the file as evidence.

The expected tone is inferred from the label (`rx_18000` → 18000 Hz), so the
recorder verifies the tone actually arrived rather than trusting the filename.

**`tests/test_recording_quality.py`** — 11 tests that reproduce the exact
failure signature and confirm valid audio still passes.

Check existing files at any time:

```bash
python -m dsp.recording_quality *.wav
python -m dsp.recording_quality rx_18000.wav --expected 18000
```

---

## Next steps

1. `python scripts/enum_devices.py` — confirm the correct device index.
2. Record a 3 s test **while clapping**. A working mic gives > 1000 LSB.
3. Once capture is confirmed, re-run the full frequency sweep.
4. Re-run `python -m dsp.recording_quality *.wav` and only then curate evidence.

## Data that remains usable

`clap.wav`, `rx_1000.wav`, `rx_1000_v2.wav`, `rx_5000_v2.wav`.

Enough to demonstrate the analysis pipeline on real audio — **not** enough to
support any frequency-response or hardware-ceiling claim. No such claim should
appear in the report or presentation until the sweep is re-recorded.
