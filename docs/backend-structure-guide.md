# Backend Structure Guide — SIH 2026

**Purpose:** Explain the full backend layout, how string encode/decode works, where data is captured, and what was added during the AI integration work.

**Local server:** http://localhost:8000/docs

---

## 1. High-Level Architecture

The project has **one FastAPI server** (`backend/main.py`) that combines two roles:

| Role | Responsibility |
|------|----------------|
| **Backend 2** | Attack simulator — encode text → BFSK audio → WAV |
| **Backend 1** | Threat API — receive AI detection results → store history |

```
                    ┌─────────────────────────────────────┐
                    │         backend/main.py             │
                    │      (single FastAPI app)           │
                    └──────────────┬──────────────────────┘
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │  Backend 2    │      │  Backend 1    │      │  AI module    │
   │  /simulate/*  │      │  /api/*       │      │  ai/dsp/      │
   │  /stream/*    │      │               │      │  (separate)   │
   └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
           │                      │                        │
           ▼                      ▼                        ▼
   WAV / audio stream      Threat history           Detection JSON
```

**Important:** The Backend 2 **decoder** is only for simulator self-test. Real threat detection is done by the **AI DSP pipeline** (`ai/dsp/`), not by the Backend 2 decoder.

---

## 2. Folder Structure

```
backend/
├── main.py                    ← App entry point (uvicorn runs this)
├── core/
│   └── config.py              ← Frequencies, sample rate, paths
├── models/
│   └── data_schemas.py        ← Pydantic request/response models
├── api/
│   ├── simulator.py           ← Backend 2: /simulate/*
│   ├── audio.py               ← Backend 2: /stream/*
│   ├── analysis.py            ← Backend 1: POST /api/analyze
│   ├── threats.py             ← Backend 1: GET /api/threats
│   └── system_status.py       ← In-memory state (payloads, streamer)
├── services/
│   ├── payload_service.py     ← ★ ENCODE + DECODE (string ↔ audio)
│   ├── audio_service.py       ← Microphone/file streaming + forensic decode
│   ├── acoustic_service.py    ← CLI microphone listener (standalone script)
│   └── threat_service.py      ← In-memory threat storage
└── scripts/
    ├── generate_sample.py     ← Creates samples/backend2/sample.wav
    └── test_ai_detection.py   ← Runs AI pipeline on sample WAV

ai/                            ← AI DSP module (added from feature/ai-dsp)
samples/backend2/              ← Reference WAV + JSON for AI testing
docs/                          ← Integration and structure documentation
tests/                         ← Automated tests
```

---

## 3. String Encode and Decode — Full Explanation

### 3.1 Where encode happens

**File:** `backend/services/payload_service.py`

| Function | What it does |
|----------|--------------|
| `encode_text_to_signal(text)` | **String → audio signal** |
| `_text_to_bits(text)` | Each character → 8 bits (ASCII, MSB first) |
| `_bits_to_tone(bits)` | Each bit → BFSK tone (0 = 18500 Hz, 1 = 20500 Hz) |
| `save_signal_to_wav(path, signal)` | float32 signal → 16-bit mono WAV file |

**Encode flow:**

```
Input string (e.g. "SIH 2026 BACKEND 2")
        │
        ▼
UTF-8 text → 8-bit ASCII bits per character
        │
        ▼
Preamble "10101010" prepended (sync pattern)
        │
        ▼
Each bit → tone:
   bit "0" → 18500 Hz sine wave for 50 ms
   bit "1" → 20500 Hz sine wave for 50 ms
        │
        ▼
0.5 s silence + tones + 0.5 s silence
        │
        ▼
float32 PCM array  →  saved as WAV (48000 Hz, mono, 16-bit)
```

**Who calls encode:**

| Caller | Endpoint / Script | Input data |
|--------|-------------------|------------|
| `simulator.py` | `POST /simulate/generate` | User text from API request |
| `simulator.py` | `POST /simulate/acoustic-exfiltrate` | Custom text, sysinfo, or demo keylog string |
| `generate_sample.py` | CLI script | Fixed payload `"SIH 2026 BACKEND 2"` |

**Where encoded output is stored:**

| Location | Content |
|----------|---------|
| `generated_payloads/<id>.wav` | WAV file on disk (default `STORAGE_DIR`) |
| In-memory `_payloads` dict | Signal array + metadata (`system_status.py`) |
| `samples/backend2/sample.wav` | Official reference sample for AI testing |

---

### 3.2 Where decode happens

**File:** `backend/services/payload_service.py`

| Function | What it does |
|----------|--------------|
| `decode_signal_to_text(signal)` | **Audio buffer → string** |
| `decode_audio_file(path)` | Load WAV → call `decode_signal_to_text` |
| `_find_preamble(signal)` | Search for `10101010` pattern using Goertzel |
| `_classify_bit(window)` | Decide if window is bit 0 or 1 by frequency power |
| `_bits_to_text(bits)` | 8 bits → ASCII character |

**Decode flow:**

```
WAV file or float32 audio buffer (from mic or file)
        │
        ▼
Search for preamble "10101010" in signal
        │
        ▼
For each 50 ms window after preamble:
   measure power at 18500 Hz vs 20500 Hz (Goertzel)
   → classify as bit "0" or "1"
        │
        ▼
Collect bits → group into 8-bit bytes → ASCII characters
        │
        ▼
Output: DecodeResult { text, success, confidence, bits, ... }
```

**Who calls decode:**

| Caller | Where | Input audio source |
|--------|-------|-------------------|
| `audio_service.py` | Live stream forensic hook | **Microphone** or **WAV file replay** via `/stream/start` |
| `acoustic_service.py` | CLI listener (`run_listener`) | **Microphone** ring buffer |
| `generate_sample.py` | Self-test after generating sample | `samples/backend2/sample.wav` |
| Tests | `test_payload_roundtrip.py`, etc. | In-memory signals or WAV files |

---

### 3.3 What data is captured — and where

This is the key question: **encode turns a string into sound; decode tries to recover that string from sound.** Different parts of the system capture different things.

| Data type | What is captured | Where stored | Used for |
|-----------|------------------|--------------|----------|
| **Original text (payload)** | The string you want to transmit | API request body (`text` field) | Input to encode |
| **Encoded WAV** | Digital ultrasonic audio | `generated_payloads/*.wav`, `samples/backend2/sample.wav` | Playback, AI testing, file replay |
| **Decoded text** | String recovered from audio | `DecodeResult.text` | Simulator self-test only |
| **Forensic decode events** | Decoded text + confidence from live stream | `AudioStreamer.latest_forensic_events` → WebSocket `/stream/forensics` | Demo / debugging |
| **Microphone captures** | Raw audio from mic | `forensic_captures/capture_*.wav` (CLI listener only) | Offline analysis |
| **Threat events** | AI detection result (not decoded text) | `threat_service._threats` in memory | Backend 1 API `/api/threats` |
| **Stream audio chunks** | Base64 PCM chunks | WebSocket `/stream/audio` | Detector / frontend consumption |

#### Encode path (string → audio)

```
User types: "HELLO"
     │
     ▼
POST /simulate/generate  { "text": "HELLO" }
     │
     ▼
encode_text_to_signal("HELLO")
     │
     ▼
WAV saved → generated_payloads/abc123.wav
Original string also kept in memory: payloads["abc123"]["text"] = "HELLO"
```

#### Decode path (audio → string) — simulator only

```
Microphone or WAV file
     │
     ▼
POST /stream/start?source=mic   OR   source=file&file_path=...
     │
     ▼
AudioStreamer collects 2+ seconds of audio in sliding buffer
     │
     ▼
decode_signal_to_text(buffer)  →  "HELLO" (if signal is clean enough)
     │
     ▼
DecodedPayloadEvent sent to WebSocket /stream/forensics
```

#### AI detection path (audio → threat) — production path

```
samples/backend2/sample.wav  (or live mic via Detector — future)
     │
     ▼
ai/dsp/DSPPipeline.process(chunks)
     │
     ▼
pipeline.to_threat_event()
     │
     ▼
{ "detected": true, "risk": "MEDIUM", "pattern": "tone", ... }
     │
     ▼
(Future) POST /api/analyze  →  threat_service stores event
```

**The AI does NOT decode the original string.** It detects whether a covert ultrasonic signal is present and classifies the modulation pattern. Only the Backend 2 decoder recovers the actual text payload.

---

## 4. API Endpoints Summary

### Backend 2 — Simulator (`/simulate`)

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | `/simulate/generate` | Encode string → WAV |
| POST | `/simulate/transmit` | Play WAV on speaker or mark virtual complete |
| POST | `/simulate/acoustic-exfiltrate` | Demo exfiltration scenario |
| GET | `/simulate/list` | List generated payloads |
| GET | `/simulate/status/{id}` | Payload status |

### Backend 2 — Audio streaming (`/stream`)

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | `/stream/start` | Start mic or file audio stream |
| POST | `/stream/stop` | Stop stream |
| GET | `/stream/status` | Stream info + latest forensic decodes |
| WS | `/stream/audio` | Receive base64 audio chunks |
| WS | `/stream/forensics` | Receive decoded text events |

### Backend 1 — Threat management (`/api`)

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | `/api/analyze` | Store AI threat event |
| GET | `/api/threats` | All stored threats |
| GET | `/api/threats/current` | Latest threat |
| GET | `/api/system-status` | Backend online check |

### System

| Method | Endpoint | Action |
|--------|----------|--------|
| GET | `/health` | Backend 2 health + config values |
| GET | `/docs` | Swagger UI |

---

## 5. Configuration (`backend/core/config.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLE_RATE` | 48000 | Hz |
| `FREQ_0` | 18500 | Bit `0` tone frequency |
| `FREQ_1` | 20500 | Bit `1` tone frequency (was 21000) |
| `BIT_DURATION` | 0.05 | 50 ms per bit |
| `PREAMBLE` | `10101010` | Sync pattern before payload |
| `STORAGE_DIR` | `generated_payloads` | Where API-generated WAVs go |
| `FORENSIC_CAPTURE_DIR` | `forensic_captures` | CLI listener saves mic captures here |

---

## 6. What Was Added in the AI Integration Work

These are the changes made during the Backend 2 + AI integration task:

### 6.1 New files

| File | Purpose |
|------|---------|
| `ai/` (entire folder) | AI DSP pipeline from `origin/feature/ai-dsp` branch |
| `backend/scripts/generate_sample.py` | Generate `samples/backend2/sample.wav` + metadata |
| `backend/scripts/test_ai_detection.py` | Diagnostic: run AI on Backend 2 WAV |
| `tests/test_backend2_ai_integration.py` | Automated test: sample must pass AI detection |
| `tests/test_backend2_simulator.py` | Simulator encoding/reproducibility tests |
| `samples/backend2/sample.wav` | Reference WAV for AI |
| `samples/backend2/sample.json` | Measured metadata (frequencies, duration, decoder self-test) |
| `samples/backend2/README.md` | How AI should consume the sample |
| `docs/backend2-simulator.md` | Backend 2 simulator documentation |
| `docs/backend-structure-guide.md` | This file |
| `isolated/README.md` | Notes on unrelated `Reverse_Shell-main/` code |

### 6.2 Modified files

| File | Change |
|------|--------|
| `backend/core/config.py` | `FREQ_1`: 21000 → **20500** (21000 Hz failed AI detection) |
| `backend/models/data_schemas.py` | Default `freq_1_hz` → 20500; legacy reverse-shell stubs documented |
| `requirements.txt` | Added `scipy` for AI module |
| `docs/integration-contract.md` | Verified AI input/output schemas documented |
| `.gitignore`, `README.md` | Project housekeeping |

### 6.3 Key integration finding

- **WAV format** (48 kHz, mono, 16-bit) was already correct for AI.
- **FREQ_1 = 21000 Hz** was outside the AI detection band edge → changed to **20500 Hz**.
- AI test result on `sample.wav`: `detected: true`, `risk: MEDIUM`.
- Backend 1 `ThreatEvent` schema still differs from AI output — adapter needed before full E2E.

---

## 7. Quick Test Commands

```bash
# Start server
python -m uvicorn backend.main:app --reload --port 8000

# Open API docs
# http://localhost:8000/docs

# Generate reference sample
python backend/scripts/generate_sample.py

# Decoder self-test (string round-trip)
python -c "from backend.services.payload_service import decode_audio_file; print(decode_audio_file('samples/backend2/sample.wav'))"

# AI detection test
python -m pytest tests/test_backend2_ai_integration.py -v

# All tests
python -m pytest tests/ -v
```

### Example: encode via API

```http
POST http://localhost:8000/simulate/generate
Content-Type: application/json

{
  "text": "HELLO",
  "freq_0_hz": 18500,
  "freq_1_hz": 20500,
  "bit_duration_ms": 50.0
}
```

Response includes `wav_url` like `/files/abc123.wav` — download from `http://localhost:8000/files/abc123.wav`.

---

## 8. Data Flow Diagram (Encode vs Decode vs AI)

```
ENCODE (string → audio)                    DECODE (audio → string)
─────────────────────                      ─────────────────────────
User/API text                              Mic or WAV file
     │                                          │
     ▼                                          ▼
payload_service.encode_text_to_signal()    payload_service.decode_signal_to_text()
     │                                          │
     ▼                                          ▼
WAV file on disk                           Recovered string (self-test)
generated_payloads/ or samples/backend2/         │
                                                 ▼
                                          WebSocket /stream/forensics
                                          (forensic demo only)


AI DETECTION (audio → threat JSON)         BACKEND 1 (threat storage)
──────────────────────────────────         ──────────────────────────
WAV or mic chunks                               │
     │                                          ▼
     ▼                                    POST /api/analyze
ai/dsp/DSPPipeline.process()                    │
     │                                          ▼
     ▼                                    threat_service._threats
to_threat_event()                               │
     │                                          ▼
{ detected, risk, pattern, ... }          GET /api/threats
(does NOT return original text)
```

---

## 9. Related Documentation

| Document | Content |
|----------|---------|
| `docs/integration-contract.md` | Cross-team API contracts |
| `docs/backend2-simulator.md` | Simulator technical details |
| `ai/docs/AI_DETECTION_SPEC.md` | AI input/output specification |
| `ai/dsp/INTEGRATION_GUIDE.md` | How to integrate with AI DSP |
| `samples/backend2/README.md` | AI handoff sample package |
