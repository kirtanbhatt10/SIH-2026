# Backend 2 Simulator

**Owner:** Backend 2 (Attack Simulator)
**Status:** Stabilized reference implementation in unified `backend/` package

---

## Architecture

Backend 2 lives inside the unified FastAPI application (`backend/main.py`). It does **not** run a separate API server or database.

```
Payload (UTF-8 text)
    → ASCII bit encoding (8 bits/char, MSB first)
    → Preamble "10101010"
    → BFSK modulation (Goertzel-classified tones)
    → float32 PCM + WAV export
    → [optional] speaker playback / WebSocket stream
```

**Decoder path** (self-test only):

```
WAV / float32 buffer
    → Preamble search (Goertzel per bit window)
    → BFSK demodulation
    → Bit string → ASCII text
```

The decoder validates simulator integrity. It is **not** the production threat detector.

---

## Input

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | string | — | Max 256 chars via API |
| `freq_0_hz` | int | 18500 | Frequency for bit `0` |
| `freq_1_hz` | int | 20500 | Frequency for bit `1` (within AI 18–21 kHz band; 21000 Hz fails AI detection) |
| `bit_duration_ms` | float | 50.0 | Symbol duration |
| `pad_silence_sec` | float | 0.5 | Silence before/after tone |

---

## Output

| Output | Description |
|--------|-------------|
| `np.ndarray` float32 | Normalized mono signal (−1..1) |
| WAV file | 16-bit PCM mono, 48000 Hz |
| `sample.json` | Metadata from `generate_sample.py` |

---

## Modulation (BFSK)

| Bit | Frequency | Implementation |
|-----|-----------|----------------|
| `0` | 18500 Hz | `FREQ_0` in `backend/core/config.py` |
| `1` | 20500 Hz | `FREQ_1` in `backend/core/config.py` |

- **Type:** Binary FSK (two discrete tones per bit window)
- **Symbol duration:** 0.05 s
- **Amplitude:** `0.5 * sin(2π f t)` with 10% linear ramp envelope per symbol
- **Channels:** 1 (mono)
- **Phase:** Continuous sine per symbol (no explicit phase continuity across symbols)

---

## WAV format

| Property | Value |
|----------|-------|
| Format | RIFF WAV |
| Channels | 1 |
| Sample rate | 48000 Hz |
| Sample width | 16-bit signed PCM |
| Clipping | Values clipped to ±1.0 before int16 conversion |

---

## Metadata format

See `samples/backend2/sample.json` for a populated example. Key fields:

- `modulation`, `carrier_frequency_0_hz`, `carrier_frequency_1_hz`
- `symbol_duration_seconds`, `sample_rate`, `channels`, `duration_seconds`
- `encoding.preamble_bits`, `encoded_bit_count`
- `decoder_self_test` (populated by generator script)
- `audio_quality` (peak, RMS, NaN/inf/clipping flags)

---

## Decoder self-test

```bash
python -m pytest tests/test_payload_roundtrip.py tests/test_backend2_simulator.py -v
```

Round-trip via script:

```bash
python backend/scripts/generate_sample.py
python -c "from backend.services.payload_service import decode_audio_file; print(decode_audio_file('samples/backend2/sample.wav'))"
```

---

## AI handoff

Package location: `samples/backend2/`

| File | Purpose |
|------|---------|
| `sample.wav` | Feed directly into AI/DSP pipeline |
| `sample.json` | Ground-truth simulator parameters |
| `README.md` | Consumption instructions |

### Verified AI compatibility (2026-08-15)

| Requirement | Backend 2 value | AI expectation |
|-------------|-----------------|----------------|
| Sample rate | 48000 Hz | 48000 Hz (required) |
| Channels | 1 (mono) | Mono (stereo averaged) |
| Format | 16-bit PCM WAV | float32/64 in [-1, 1] after load |
| Chunk size | N/A (file) | 2048 samples recommended for streaming |
| Carrier frequencies | 18500 / 20500 Hz | Must fall within 18000–21000 Hz detection band |

**Note:** `FREQ_1` was changed from 21000 to 20500 Hz because the AI DSP ultrasonic band ends at 21000 Hz and pure tones at exactly 21000 Hz are not detected (verified with `ai/dsp/DSPPipeline`). Both carriers must be recoverable for FSK classification.

Run the integration test:

```bash
python -m pytest tests/test_backend2_ai_integration.py -v
```

AI output schema for Backend 1 is documented in `docs/integration-contract.md` Section 5 (DSP schema **VERIFIED**).

---

## API endpoints (Backend 2)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/simulate/generate` | Encode payload → WAV |
| POST | `/simulate/transmit` | Virtual or speaker transmit |
| POST | `/simulate/acoustic-exfiltrate` | Demo exfiltration scenario |
| GET | `/simulate/status/{id}` | Payload status |
| GET | `/simulate/list` | List payloads |
| WS | `/stream/audio` | Live chunk broadcast |
| WS | `/stream/forensics` | Decoder events (simulator forensic stream) |

---

## Limitations

1. No checksum or error correction on payload bits.
2. Decoder uses fixed preamble search—fragile under heavy noise (expected; not production detector).
3. Ultrasonic frequencies not validated on physical hardware in this package.
4. `acoustic-exfiltrate` `keylog` source type returns **simulated** demo text, not a real keylogger.
5. In-memory payload store (`system_status._payloads`)—not persistent.

---

## File map

| Path | Role |
|------|------|
| `backend/services/payload_service.py` | Encode, decode, WAV I/O, Goertzel |
| `backend/core/config.py` | Default frequencies and timing |
| `backend/api/simulator.py` | REST simulator routes |
| `backend/api/audio.py` | WebSocket streaming |
| `backend/services/audio_service.py` | Stream sources and forensic decode hook |
| `backend/services/acoustic_service.py` | CLI microphone listener |
| `backend/scripts/generate_sample.py` | Reference sample generator |
| `samples/backend2/` | AI handoff package |

---

## Production detection path (not Backend 2 decoder)

```
Microphone → Detector → DSP/AI → POST /api/analyze → Threat Service
```

Backend 2 provides **controlled test signals** for validating that path.
