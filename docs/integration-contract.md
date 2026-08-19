# SIH 2026 Acoustic Shield — Integration Contract

**Document owner:** Backend 1 (Backend Lead)
**Last updated:** 2026-08-14
**Purpose:** Central technical handoff between Attack Simulator, Detector, AI, Backend, and Frontend teams.

---

## Status Legend

| Label | Meaning |
|-------|---------|
| **FINAL / IMPLEMENTED** | Currently implemented and in use |
| **TEMPORARY** | Working prototype; subject to change after team agreement |
| **PENDING / TO BE DECIDED** | Not yet defined; requires cross-team coordination |
| **EXPERIMENTAL / TO BE MEASURED** | Value must come from hardware or lab experiments, not assumptions |

---

## 1. System Overview

### Intended Data Flow

```
Controlled Payload
    ->
Encoding
    ->
Modulation
    ->
Audio Signal
    ->
Speaker
    ->
Air
    ->
Microphone
    ->
Detector / Audio Acquisition
    ->
Signal Processing
    ->
Feature Extraction
    ->
AI Classification / Threat Scoring
    ->
Backend
    ->
Frontend Dashboard
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Controlled Payload** | Source data to be transmitted covertly over acoustic channel (simulated attack content). |
| **Encoding** | Converts payload into a bitstream or symbol sequence suitable for modulation. |
| **Modulation** | Maps encoded data onto an acoustic waveform (e.g., FSK, tone patterns). |
| **Audio Signal** | Digital representation of the modulated signal ready for playback or transmission. |
| **Speaker** | Converts electrical signal to airborne sound. |
| **Air** | Physical acoustic propagation channel (attenuation, noise, multipath). |
| **Microphone** | Captures airborne sound as an electrical/digital signal. |
| **Detector / Audio Acquisition** | Records or streams microphone input; prepares audio for downstream processing. |
| **Signal Processing** | Filtering, normalization, segmentation, and conditioning of raw audio. |
| **Feature Extraction** | Derives measurable characteristics (frequency bands, duration, modulation signatures). |
| **AI Classification / Threat Scoring** | Classifies whether covert communication is present; assigns confidence and risk. |
| **Backend** | Validates AI output, stores threat state/history, exposes REST API to Frontend. |
| **Frontend Dashboard** | Displays current threat, history, and system status to operators. |

---

## 2. Team Ownership

### Backend 1

- API architecture
- Data contracts (this document)
- Threat state / history
- AI integration (`POST /api/analyze`)
- Frontend integration
- Logging / configuration *(PENDING)*
- End-to-end integration coordination

### Backend 2 (Attack Simulator)

- Controlled attack simulator
- Payload encoding
- Modulation
- Audio generation (WAV / controlled samples)
- Demodulation
- Decoder
- Communication experiments
- Controlled audio samples for AI and Detector testing

### AI Team

- DSP
- Signal processing
- Feature extraction
- Dataset creation and labeling
- Classification models
- Confidence / risk methodology
- AI inference interface
- **AI -> Backend output specification** *(PENDING — see Section 5)*

### Detector

- Microphone / audio acquisition
- Audio preprocessing
- Audio chunk / file interface to AI pipeline

### Frontend

- Dashboard UI
- Threat visualization
- Threat history display
- System status display
- Backend API integration

---

## 3. Simulator -> Detector Contract

**Status: PARTIAL — Software simulator parameters VERIFIED; hardware propagation TO BE MEASURED**

Backend 2 produces controlled WAV samples and JSON metadata. Values below marked **IMPLEMENTED (simulator)** come from `backend/core/config.py` and `backend/services/payload_service.py` and were verified by round-trip tests and `samples/backend2/sample.json`. They describe the **digital signal** before speaker output—not measured over-the-air performance.

### Verified Simulator Parameters (IMPLEMENTED — software)

| Item | Value | Status |
|------|-------|--------|
| Audio format | RIFF WAV, 16-bit signed PCM mono | **IMPLEMENTED (simulator)** |
| Sample rate (Hz) | 48000 | **IMPLEMENTED (simulator)** |
| Number of channels | 1 (mono) | **IMPLEMENTED (simulator)** |
| Bit depth | 16-bit PCM | **IMPLEMENTED (simulator)** |
| Modulation type | BFSK (binary FSK) | **IMPLEMENTED (simulator)** |
| Bit 0 frequency (Hz) | 18500 | **IMPLEMENTED (simulator)** |
| Bit 1 frequency (Hz) | 20500 | **IMPLEMENTED (simulator)** — changed from 21000; see Section 3 notes |
| Symbol / bit duration | 0.05 s (50 ms) | **IMPLEMENTED (simulator)** |
| Preamble | `10101010` (8 bits) | **IMPLEMENTED (simulator)** |
| Encoding | UTF-8 text → 8-bit ASCII bits, MSB first per byte | **IMPLEMENTED (simulator)** |
| Checksum | None | **IMPLEMENTED (simulator)** |
| Reference sample | `samples/backend2/sample.wav` + `sample.json` | **IMPLEMENTED (simulator)** |
| Reproduction | `python backend/scripts/generate_sample.py` | **IMPLEMENTED (simulator)** |
| Documentation | `docs/backend2-simulator.md`, `samples/backend2/README.md` | **IMPLEMENTED (simulator)** |

### Still Pending / To Be Measured

| Item | Status |
|------|--------|
| Over-the-air sample rate at microphone | **TO BE MEASURED** |
| Duration / chunk size for live detector streams | **PENDING** (stream uses 1.0 s window / 0.5 s hop — not final Detector contract) |
| Experiment conditions (distance, volume, environment) | **TO BE MEASURED** |
| Detector intake API/format | **PENDING** |

### Notes

- Values that depend on speaker hardware, microphone hardware, room acoustics, or distance must be labeled **TO BE MEASURED** and obtained through controlled experiments.
- Backend 2 should document each sample's experimental conditions alongside the audio file.
- **Backend 2** owns simulator-side implementation of this handoff (encoding, modulation, audio generation, metadata).
- **Detector** owns detector-side implementation of this handoff (audio acquisition, intake format, reception behavior).
- **Backend 1** owns and coordinates the shared integration contract in this document. Backend 2 and Detector must agree on technical details; Backend 1 facilitates updates to this document and cross-team alignment.
- **VERIFIED (2026-08-15):** `FREQ_1` was adjusted from 21000 to 20500 Hz. The AI DSP detection band is 18000–21000 Hz (`ai/dsp/__init__.py`); pure tones at exactly 21000 Hz are not recovered by the FFT peak detector. FSK with carriers 18500/21000 Hz fails AI detection; 18500/20500 Hz passes (`tests/test_backend2_ai_integration.py`).

---

## 4. Detector -> AI Contract

**Status: PARTIAL — AI DSP input requirements VERIFIED; Detector acquisition format PENDING**

The AI DSP module (`ai/dsp/`, branch `feature/ai-dsp`) defines verified audio input requirements. Detector acquisition format (microphone, chunking, file packaging) remains **PENDING**.

### AI DSP Input Requirements (VERIFIED)

Source: `ai/docs/AI_DETECTION_SPEC.md`, `ai/dsp/INTEGRATION_GUIDE.md`, `ai/dsp/dsp_api.py`

| Item | Value | Status |
|------|-------|--------|
| Sample rate | **48000 Hz** (raises `ValueError` if < 44100) | **VERIFIED** |
| Channels | Mono (stereo averaged) | **VERIFIED** |
| Format | float32/float64 in [-1.0, 1.0]; int16 PCM scaled on load | **VERIFIED** |
| Recommended chunk size | 2048 samples (~42.7 ms at 48 kHz) | **VERIFIED** |
| WAV file input | Supported via `scipy.io.wavfile` or `SignalGenerator.load_wav()` | **VERIFIED** |
| Detection band | 18000–21000 Hz (tones at exactly 21000 Hz not recovered) | **VERIFIED** |
| Filter band (AI preprocessing) | 17500–21500 Hz Butterworth bandpass, order 6 | **VERIFIED** |

### AI Preprocessing (VERIFIED — performed by AI, not Backend 2)

Backend 2 provides **raw WAV**. The AI DSP pipeline performs:

1. DC offset removal
2. First-order pre-emphasis
3. Stateful Butterworth bandpass 17.5–21.5 kHz

Do **not** duplicate this preprocessing in Backend 2.

### AI Inference Entry Point (VERIFIED)

```python
import sys
sys.path.insert(0, "ai")
from dsp import DSPPipeline

pipeline = DSPPipeline()  # 48 kHz, 2048 FFT
for chunk in audio_stream:  # shape (2048,), float, mono
    pipeline.process(chunk)
event = pipeline.to_threat_event()  # call at end of capture, not per chunk
```

File-based test (Backend 2 handoff):

```bash
python -m pytest tests/test_backend2_ai_integration.py -v
```

### Detector Responsibilities (PENDING)

| Item | Status |
|------|--------|
| Microphone / audio acquisition | **PENDING** |
| Live stream chunking to 2048 samples | **PENDING** |
| Error behavior for dead microphone | **VERIFIED** in `ai/dsp/recording_quality.py` (not Detector code yet) |
| Real-time vs batch | **PENDING** |

### Notes

- Backend 2 `sample.wav` at 48 kHz mono is directly loadable by the AI pipeline without format conversion.
- Detector and AI teams must still agree on live-stream handoff; Backend 1 coordinates contract updates.

---

## 5. AI -> Backend Contract

**Status: IMPLEMENTED — Backend 1 `ThreatEvent` matches the 13-field DSP schema (`1.0.0-dsp`)**

The AI DSP layer (`ai/dsp/dsp_api.py`) produces a threat event via `DSPPipeline.to_threat_event()`. Backend 1 accepts the same 13-field shape via `POST /api/analyze` with Pydantic validation (`backend/models/data_schemas.py`). Backend 1 does not transform or recalculate AI fields — receive → validate → store → serve.

### AI DSP Output Schema (VERIFIED)

Source: `ai/docs/AI_DETECTION_SPEC.md`, pinned by `ai/tests/test_threat_event_contract.py`

```json
{
  "schema_version": "1.0.0-dsp",
  "detected": true,
  "confidence": 0.89,
  "risk": "MEDIUM",
  "suspicion_score": 0.6365,
  "frequency_start": 18492.2,
  "frequency_end": 20507.8,
  "carrier_freqs": [18492.2, 20507.8],
  "duration": 7.68,
  "pattern": "fsk",
  "snr": 55.17,
  "chunks_analyzed": 201,
  "timestamp": 1786794542.14
}
```

### AI DSP Field Summary (VERIFIED)

| Field | Type | Meaning | Status |
|-------|------|---------|--------|
| `schema_version` | string | `"1.0.0-dsp"` | **VERIFIED** |
| `detected` | bool | Accumulated threat verdict | **VERIFIED** |
| `confidence` | float 0–1 | Modulation classifier certainty (uncalibrated) | **VERIFIED** |
| `suspicion_score` | float 0–1 | Threat heuristic (distinct from `confidence`) | **VERIFIED** |
| `risk` | string | `LOW` / `MEDIUM` / `HIGH` from suspicion thresholds | **VERIFIED** (DSP defaults) |
| `frequency_start` | float or null | Min detected carrier Hz | **VERIFIED** |
| `frequency_end` | float or null | Max detected carrier Hz | **VERIFIED** |
| `carrier_freqs` | float[] | Discrete carriers (preserve for FSK) | **VERIFIED** |
| `duration` | float | Segmentation-based seconds | **VERIFIED** |
| `pattern` | string | `none`, `tone`, `fsk`, `ook`, `chirp` | **VERIFIED** (OOK ~70% on synthetic) |
| `snr` | float | Peak SNR in dB | **VERIFIED** |
| `chunks_analyzed` | int | Chunks processed | **VERIFIED** |
| `timestamp` | float | Unix timestamp | **VERIFIED** |

Risk bands (DSP defaults, not calibrated): `HIGH >= 0.75`, `MEDIUM >= 0.45`, else `LOW`.

### Verified Backend 2 → AI Test Result (2026-08-15)

| Item | Value |
|------|-------|
| Input | `samples/backend2/sample.wav` |
| AI method | `DSPPipeline.process()` + `to_threat_event()` |
| `detected` | `true` (after `FREQ_1` adjusted to 20500 Hz) | **VERIFIED** |
| `risk` | `MEDIUM` | **VERIFIED** |
| `suspicion_score` | ~0.58 | **VERIFIED** |
| `pattern` | `tone` (not `fsk` — see note) | **VERIFIED** |
| `carrier_freqs` | `[18492.2]` (20500 Hz not consistently accumulated) | **VERIFIED** |
| Test | `tests/test_backend2_ai_integration.py` | **VERIFIED** |

**Note:** With 50 ms BFSK symbols and a real encoded payload, each 2048-sample AI frame (~42.7 ms) often contains one carrier, so `pattern` may be `tone` even though `detected` is `true`. AI `generate_attack_signal("fsk")` with random bits classifies as `fsk` with two carriers.

### Backend 1 `ThreatEvent` Schema (IMPLEMENTED)

Backend 1 uses the same 13-field contract as the AI DSP output (see above). Source: `backend/models/data_schemas.py`, pinned by `tests/test_backend1_api.py`.

`frequency_start`, `frequency_end`, and `carrier_freqs` are stored separately — Backend 1 does **not** collapse carriers into `frequency.min` / `frequency.max`.

`confidence` (modulation certainty) and `suspicion_score` (threat heuristic) are distinct fields and are stored unchanged.

### Integration wiring (IMPLEMENTED — HTTP client)

| Item | Status |
|------|--------|
| HTTP client (`integration/backend_client.py`) → `POST /api/analyze` | **IMPLEMENTED** |
| Config `BACKEND_API_URL` (default `http://127.0.0.1:8000`) | **IMPLEMENTED** (`integration/config.py`) |
| CLI runner `python -m integration.run_dsp_to_backend` | **IMPLEMENTED** |
| Detector audio stream → `DSPPipeline` → client → Backend | **PENDING** |

### PENDING (AI 2 — not DSP layer)

| Item | Status |
|------|--------|
| Learned classifier model | **PENDING** (AI 2 deliverable) |
| Detection accuracy / FPR / FNR | **PENDING** |
| Calibrated confidence and risk thresholds | **PENDING** |
| Over-the-air performance | **TO BE MEASURED** |

### Rules

- **Backend must not invent AI measurements.** Backend validates structure and stores values; it does not generate confidence, risk, or frequency data.
- **AI owns inference logic.** Backend owns API validation and storage.
- Changes to the production schema require AI team sign-off and an update to this document before implementation changes.

---

## 6. Current Backend API Contract

**Status: IMPLEMENTED**

All endpoints below are live in the current codebase. Base URL during local development: `http://127.0.0.1:8000`.

---

### `GET /api/system-status`

**Purpose:** Backend health and version check.

**Status:** **IMPLEMENTED**

**Response:**

```json
{
  "status": "online",
  "service": "acoustic-shield-backend",
  "version": "0.1.0"
}
```

---

### `POST /api/analyze`

**Purpose:** Receive and validate a `ThreatEvent` from the AI pipeline (or integration test harness).

**Status:** **IMPLEMENTED** — 13-field `ThreatEvent` schema (`1.0.0-dsp`, see Section 5).

**Request body:** `ThreatEvent` (JSON)

**Response:**

```json
{
  "status": "received",
  "event": { ... }
}
```

**Internal flow:**

```
AI / Detector integration layer
    ->
POST /api/analyze
    ->
Pydantic validation (ThreatEvent)
    ->
Threat Service (add_threat)
    ->
In-memory history
```

**Validation rules (current implementation):**

- `schema_version`: string (expected `"1.0.0-dsp"`)
- `confidence`, `suspicion_score`: 0.0–1.0
- `risk`: `LOW` | `MEDIUM` | `HIGH`
- `pattern`: `none` | `tone` | `fsk` | `ook` | `chirp`
- `frequency_start`, `frequency_end`: nullable, ≥ 0 when provided
- `carrier_freqs`: array of floats ≥ 0 (may be empty)
- `duration`: ≥ 0 (event-level seconds)
- `chunks_analyzed`: int ≥ 0
- `timestamp`: Unix epoch float
- All 13 fields required; `frequency_start` / `frequency_end` may be `null`

---

### `GET /api/threats`

**Purpose:** Return all stored threat events in arrival order.

**Status:** **IMPLEMENTED**

**Storage:** In-memory only (data lost on server restart).

**Response:**

```json
{
  "threats": [
    {
      "schema_version": "1.0.0-dsp",
      "detected": true,
      "confidence": 0.94,
      "risk": "HIGH",
      "suspicion_score": 0.91,
      "frequency_start": 19800.0,
      "frequency_end": 21200.0,
      "carrier_freqs": [19800.0, 21200.0],
      "duration": 3.2,
      "pattern": "fsk",
      "snr": 18.5,
      "chunks_analyzed": 75,
      "timestamp": 1786621450.25
    }
  ]
}
```

Empty history:

```json
{
  "threats": []
}
```

---

### `GET /api/threats/current`

**Purpose:** Return the most recently stored threat event.

**Status:** **IMPLEMENTED**

**Response (threat exists):**

```json
{
  "current": {
    "schema_version": "1.0.0-dsp",
    "detected": true,
    "confidence": 0.94,
    "risk": "HIGH",
    "suspicion_score": 0.91,
    "frequency_start": 19800.0,
    "frequency_end": 21200.0,
    "carrier_freqs": [19800.0, 21200.0],
    "duration": 3.2,
    "pattern": "fsk",
    "snr": 18.5,
    "chunks_analyzed": 75,
    "timestamp": 1786621450.25
  }
}
```

**Response (no threats stored):**

```json
{
  "current": null,
  "message": "No threats have been recorded yet"
}
```

HTTP status: `200` in both cases (no 404 or 500 for empty history).

---

## 7. Backend -> Frontend Contract

**Status: BASIC CONTRACT READY**

### Endpoints Frontend Should Consume

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `GET /api/system-status` | Show backend online/offline state | **IMPLEMENTED** |
| `GET /api/threats/current` | Show current / latest threat | **IMPLEMENTED** |
| `GET /api/threats` | Show threat history | **IMPLEMENTED** |

Frontend should **not** call `POST /api/analyze` directly. That endpoint is for the AI / integration layer.

### Fields Frontend Should Display (where available)

| Field | Source | Status |
|-------|--------|--------|
| Current threat state (`detected`) | `GET /api/threats/current` | **IMPLEMENTED** |
| Risk | `GET /api/threats/current` | **IMPLEMENTED** (TEMPORARY schema) |
| Confidence | `GET /api/threats/current` | **IMPLEMENTED** (TEMPORARY schema) |
| Frequency range | `GET /api/threats/current` | **IMPLEMENTED** (TEMPORARY schema) |
| Duration | `GET /api/threats/current` | **IMPLEMENTED** (TEMPORARY schema) |
| Pattern | `GET /api/threats/current` | **IMPLEMENTED** (TEMPORARY schema) |
| Timestamp | — | **PENDING** (not yet in schema) |
| Threat history | `GET /api/threats` | **IMPLEMENTED** |
| Backend / system status | `GET /api/system-status` | **IMPLEMENTED** |

### Rules

- Frontend must **not** depend on AI internals, model outputs, or raw audio.
- Frontend communicates exclusively through Backend REST APIs.
- Frontend must **not** calculate threat scores or re-derive confidence/risk values.
- When the AI schema is finalized, Frontend may need to adapt display fields — coordinate via this document.

---

## 8. Data Ownership Rules

1. **AI owns AI inference logic.** Models, features, confidence, and risk calculations are AI team responsibility.
2. **Backend owns API validation and storage.** Backend validates incoming data against agreed schemas and persists threat state.
3. **Backend does not modify AI predictions arbitrarily.** Backend stores and serves values as received (after validation only).
4. **Frontend does not calculate threat scores.** Display only; no client-side re-scoring.
5. **Simulator (Backend 2) owns signal generation.** Encoding, modulation, and controlled audio sample production.
6. **Detector owns audio acquisition.** Microphone capture, chunking, and preprocessing up to the Detector -> AI boundary.
7. **Experimental measurements must come from experiments, not invention.** See Section 9.
8. **Shared contracts must be coordinated before changing them.** See Section 12.
9. **Temporary contracts must be clearly marked.** See Status Legend at top of this document.
10. **The system detects potential covert acoustic communication; detection alone does not prove malicious intent.** Alerts indicate anomalous acoustic patterns warranting investigation, not confirmed attacks.

---

## 9. Experimental Data Rules

The following values **must be experimentally measured** in controlled lab conditions. Never substitute theoretical or assumed numbers as project measurements.

| Measurement | Status |
|-------------|--------|
| Detection range (distance) | **TO BE MEASURED** |
| Frequency limits (usable band) | **TO BE MEASURED** |
| SNR (signal-to-noise ratio) | **TO BE MEASURED** |
| Detection accuracy | **TO BE MEASURED** |
| False positive rate | **TO BE MEASURED** |
| False negative rate | **TO BE MEASURED** |
| End-to-end latency | **TO BE MEASURED** |
| Payload bit rate / communication bitrate | **TO BE MEASURED** |
| Bit error rate (BER) | **TO BE MEASURED** |
| Packet / message success rate | **TO BE MEASURED** |
| Detection latency | **TO BE MEASURED** |
| Speaker capability (frequency response, SPL) | **TO BE MEASURED** |
| Microphone capability (frequency response, sensitivity) | **TO BE MEASURED** |

### Rules

- Document the experimental setup (hardware, distance, environment, signal parameters) alongside every measurement.
- Publish measured values in team-specific docs or appendices; reference them here once validated.
- Do not use theoretical numbers in demos, dashboards, or reports as if they were measured results.

---

## 10. Current Dependencies

| Handoff | Owner | Receiver | Status |
|---------|-------|----------|--------|
| Simulator -> Detector | Backend 2 | Detector | **PARTIAL** (WAV + JSON reference sample ready) |
| Backend 2 -> AI (file) | Backend 2 | AI DSP | **VERIFIED** (`tests/test_backend2_ai_integration.py`) |
| Detector -> AI (live) | Detector | AI | **PENDING** |
| AI -> Backend | AI DSP | Backend 1 | **MISMATCH** (DSP schema verified; Backend 1 uses temp schema) |
| Backend -> Frontend | Backend 1 | Frontend | **BASIC CONTRACT READY** |
| Full E2E | Backend 1 (coordinates) | All teams | **PENDING** |

---

## 11. Next Handoffs

### Backend 1

- Maintain and update this integration contract
- Coordinate final AI output schema with AI team
- Add timestamps / event IDs to `ThreatEvent` after contract stabilization
- Prepare logging, configuration, and error handling *(PENDING)*

### Backend 2

- Build payload -> modulation -> WAV -> demodulation round-trip
- Produce controlled audio samples for Detector and AI testing
- Record signal metadata and experimental conditions with each sample

### AI Team

- Research signal characteristics of target modulation schemes
- Verify hardware feasibility with available speaker/microphone
- Build DSP / feature extraction pipeline
- Define **AI Detection Specification** (validated output contract)
- Provide validated AI -> Backend output schema

### Detector

- Define audio acquisition format and interface
- Build preprocessing / audio chunk pipeline for AI consumption

### Frontend

- Build dashboard using mock API data against current endpoints
- Integrate with live Backend once API contract is stable
- Adapt display when timestamps and final AI fields are added

---

## 12. Change Control

Before changing any shared contract (API schema, audio format, AI output fields, etc.):

1. **Identify affected teams** — list every team whose code or tests will break.
2. **Explain why the change is required** — link to experiment results, bug, or new requirement.
3. **Update this document** — mark old values as superseded; add new values with status labels.
4. **Update implementation** — Backend, AI, Frontend, Simulator, or Detector as needed.
5. **Test affected components** — verify the full handoff path still works.
6. **Commit with a clear Git message** — reference the contract change and affected teams.

**Do not silently change shared API or data structures.**

---

## 13. MVP Target

### MVP Data Flow

```
Controlled Payload
    ->
Simulator (Backend 2)
    ->
Acoustic Signal
    ->
Microphone
    ->
Detector
    ->
DSP / AI
    ->
Potential Covert Communication Detected
    ->
Backend (Backend 1)
    ->
Threat History / Current Threat
    ->
Frontend Dashboard
    ->
Security Alert
```

### MVP Success Criteria

The MVP is successful when the entire path above works reliably enough for the SIH demonstration:

- Simulator produces a controlled covert acoustic signal
- Detector captures and forwards audio to AI
- AI classifies the signal and posts a `ThreatEvent` to Backend
- Backend stores and serves threat state
- Frontend displays current threat and history with a visible alert

Exact reliability thresholds (accuracy, latency, false positive rate) are **TO BE MEASURED** and **TO BE DECIDED** before the demo.

---

## Appendix: Backend Stack (Reference)

| Item | Value | Status |
|------|-------|--------|
| Language | Python 3.12 | **IMPLEMENTED** |
| Framework | FastAPI | **IMPLEMENTED** |
| Server | Uvicorn | **IMPLEMENTED** |
| Validation | Pydantic | **IMPLEMENTED** |
| Storage | In-memory (Python list) | **IMPLEMENTED** (temporary) |
| Database | None | **PENDING** |
