# Backend 2 — AI Handoff Sample Package

This directory contains the **official reference sample** produced by the Backend 2 acoustic communication simulator for AI/DSP testing.

## Contents

| File | Description |
|------|-------------|
| `sample.wav` | Mono PCM WAV of BFSK-modulated payload |
| `sample.json` | Simulator metadata (frequencies, timing, encoding, measured audio stats) |
| `README.md` | This file |

## Payload

**Text:** `SIH 2026 BACKEND 2`

This is a controlled, non-sensitive test string suitable for pipeline validation.

## How the sample was generated

```bash
python backend/scripts/generate_sample.py
```

Optional overrides:

```bash
python backend/scripts/generate_sample.py --payload "HELLO" --out-dir samples/backend2
```

## Signal parameters (from simulator implementation)

| Parameter | Value |
|-----------|-------|
| Modulation | BFSK (binary frequency-shift keying) |
| Bit 0 frequency | 18500 Hz |
| Bit 1 frequency | 20500 Hz |
| Symbol / bit duration | 0.05 s (50 ms) |
| Sample rate | 48000 Hz |
| Channels | 1 (mono) |
| Sample width | 16-bit PCM |
| Tone amplitude | 0.5 peak (before envelope) |
| Envelope | 10% linear ramp per symbol |
| Preamble | `10101010` (8 bits, alternating) |
| Silence padding | 0.5 s before and after tone |

## Encoding

1. Payload is UTF-8 text interpreted as ASCII bytes.
2. Each byte → 8 bits, MSB first (`ord(c):08b`).
3. Preamble `10101010` is prepended before payload bits.
4. No checksum or framing beyond preamble.

## Expected signal characteristics

- Ultrasonic band: primary energy at 18500 Hz and 20500 Hz during active symbols.
- Leading/trailing 0.5 s silence.
- Non-zero RMS during modulated region; near-zero during padding.
- Decoder self-test (simulator only): WAV → demodulate → decode recovers exact payload text.

## How AI should consume `sample.wav`

1. Load as mono float or int16 PCM at **48000 Hz** (do not resample unless your pipeline documents a different rate).
2. Segment or analyze the full file; active BFSK region starts after ~0.5 s silence.
3. Use `sample.json` for ground-truth modulation parameters—not inferred guesses.
4. **Do not** use the Backend 2 decoder output as the production threat detection result. Production path is: microphone → detector → DSP/AI → `POST /api/analyze`.

## Decoder self-test (simulator integrity only)

```bash
python -c "
from backend.services.payload_service import decode_audio_file
r = decode_audio_file('samples/backend2/sample.wav')
print(r.text, r.success, r.confidence)
"
```

Expected: `SIH 2026 BACKEND 2`, `True`, high confidence.

## Hardware note

Frequencies and timing above are **software simulator parameters**. Whether 18500/20500 Hz propagate faithfully through a given speaker/microphone pair is **TO BE MEASURED** in lab conditions—not assumed from this file.

## AI pipeline verification

```bash
python -m pytest tests/test_backend2_ai_integration.py -v
```

Expected: AI DSP detects `fsk` with two carriers near 18500 and 20500 Hz.

## Related code

| Component | Path |
|-----------|------|
| Encode / modulate / decode | `backend/services/payload_service.py` |
| Default config | `backend/core/config.py` |
| Sample generator | `backend/scripts/generate_sample.py` |
| Simulator API | `backend/api/simulator.py` |
| Full documentation | `docs/backend2-simulator.md` |
