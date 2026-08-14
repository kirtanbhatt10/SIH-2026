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

**Status: PENDING / TO BE DECIDED**

Backend 2 (Attack Simulator) is expected to produce audio and metadata for the Detector team. Final values have **not** been agreed upon. Do not treat any numeric or format values in this section as final.

### What the Contract Must Eventually Specify

| Item | Status |
|------|--------|
| Audio format (WAV, raw PCM, stream, etc.) | PENDING |
| WAV / stream behavior | PENDING |
| Sample rate (Hz) | **TO BE MEASURED** |
| Number of channels (mono/stereo) | PENDING |
| Bit depth (if relevant) | PENDING |
| Duration / chunk size | PENDING |
| Signal metadata (payload ID, experiment run, etc.) | PENDING |
| Modulation type | PENDING |
| Carrier / frequency information (where available) | **TO BE MEASURED** |
| Experiment conditions (distance, volume, environment) | **TO BE MEASURED** |

### Notes

- Values that depend on speaker hardware, microphone hardware, room acoustics, or distance must be labeled **TO BE MEASURED** and obtained through controlled experiments.
- Backend 2 should document each sample's experimental conditions alongside the audio file.
- **Backend 2** owns simulator-side implementation of this handoff (encoding, modulation, audio generation, metadata).
- **Detector** owns detector-side implementation of this handoff (audio acquisition, intake format, reception behavior).
- **Backend 1** owns and coordinates the shared integration contract in this document. Backend 2 and Detector must agree on technical details; Backend 1 facilitates updates to this document and cross-team alignment.

---

## 4. Detector -> AI Contract

**Status: PENDING / TO BE DECIDED**

The Detector will provide audio to the AI pipeline. Final values have **not** been agreed upon.

### Responsibility Boundary (PENDING)

| Area | Owner | Status |
|------|-------|--------|
| Microphone / audio acquisition | Detector | **PENDING / TO BE DECIDED** |
| Basic acquisition-level preparation (recording, chunking, file/stream packaging) | Detector | **PENDING / TO BE DECIDED** |
| DSP analysis, signal processing, feature extraction, classification | AI | **PENDING / TO BE DECIDED** |
| Exact handoff boundary (what Detector delivers vs what AI computes) | Detector + AI (Backend 1 coordinates contract) | **PENDING / TO BE DECIDED** |

- **Detector** owns microphone/audio acquisition and basic acquisition-level preparation such as recording and chunking.
- **AI** owns DSP analysis, signal processing, feature extraction, and classification unless the teams explicitly agree otherwise.
- The exact boundary between Detector output and AI input is **PENDING** until the AI and Detector teams agree. Backend 1 coordinates and documents the agreed contract here.

### What the Contract Must Eventually Specify

| Item | Status |
|------|--------|
| Input audio format | PENDING |
| Sample rate | **TO BE MEASURED** |
| Channels | PENDING |
| Chunk duration | PENDING |
| Preprocessing responsibilities (Detector vs AI) | PENDING |
| Raw audio vs preprocessed audio | PENDING |
| Handling of silence / noise | PENDING |
| Error behavior (missing audio, corrupt chunk, etc.) | PENDING |
| Real-time vs batch behavior | PENDING |

### Notes

- Detector and AI teams must agree on the exact handoff boundary and who performs each step beyond basic acquisition-level preparation.
- Backend 1 owns and coordinates the shared integration contract; Detector and AI own their respective implementations.

---

## 5. AI -> Backend Contract

**Status: TEMPORARY — THIS IS NOT THE FINAL AI CONTRACT**

The Backend currently accepts a **prototype** `ThreatEvent` schema via `POST /api/analyze`. This schema exists to unblock Frontend and integration work. It will be replaced once the AI team delivers a validated **AI Detection Specification**.

### Current Prototype Schema (TEMPORARY)

```json
{
  "detected": true,
  "confidence": 0.94,
  "risk": "HIGH",
  "frequency": {
    "min": 19800,
    "max": 21200
  },
  "duration": 3.2,
  "pattern": "FSK-like"
}
```

### Prototype Field Summary (TEMPORARY — not validated by AI team)

| Field | Type | Backend Validation | AI Validation |
|-------|------|-------------------|---------------|
| `detected` | `bool` | Required | **PENDING** |
| `confidence` | `float` (0.0–1.0) | Required, `ge=0.0`, `le=1.0` | **PENDING** |
| `risk` | `string` | Required (no enum enforced yet) | **PENDING** |
| `frequency.min` | `float` (≥ 0) | Required | **PENDING** |
| `frequency.max` | `float` (≥ 0) | Required | **PENDING** |
| `duration` | `float` (≥ 0) | Required | **PENDING** |
| `pattern` | `string` | Required | **PENDING** |

### What the AI Team Must Eventually Provide

The AI team must deliver a validated **AI Detection Specification** document covering every output field:

1. **Meaning** — what the field represents
2. **Data type** — bool, float, string, object, etc.
3. **Range** — valid min/max or allowed enum values
4. **Calculation** — how the value is derived from signal processing / model output
5. **Mandatory vs optional** — whether Backend must reject requests missing this field
6. **Experimental validation** — whether the field has been verified against real hardware experiments

Expected topics in the final specification (subject to AI team validation):

- `detected`
- `confidence`
- `risk`
- Frequency information *(if reliably measurable)*
- Duration *(if reliably measurable)*
- Pattern / modulation classification *(if reliably measurable)*
- Any additional features useful to Frontend or Backend (e.g., timestamps, event IDs, SNR estimates)

### Rules

- **Backend must not invent AI measurements.** Backend validates structure and stores values; it does not generate confidence, risk, or frequency data.
- **AI owns inference logic.** Backend owns API validation and storage.
- Changes to this schema require AI team sign-off and an update to this document before implementation changes.

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

**Status:** **IMPLEMENTED** — but `ThreatEvent` schema is **TEMPORARY** (see Section 5).

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

- `confidence`: 0.0–1.0
- `frequency.min`, `frequency.max`: ≥ 0
- `duration`: ≥ 0
- All fields required (no optional fields in current schema)

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
      "detected": true,
      "confidence": 0.94,
      "risk": "HIGH",
      "frequency": { "min": 19800, "max": 21200 },
      "duration": 3.2,
      "pattern": "FSK-like"
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
    "detected": true,
    "confidence": 0.94,
    "risk": "HIGH",
    "frequency": { "min": 19800, "max": 21200 },
    "duration": 3.2,
    "pattern": "FSK-like"
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
| Simulator -> Detector | Backend 2 | Detector | **PENDING** |
| Detector -> AI | Detector | AI | **PENDING** |
| AI -> Backend | AI | Backend 1 | **TEMPORARY CONTRACT** |
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
