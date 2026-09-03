# SIH 2026 — Frontend ↔ Backend Integration Map

**Phase:** Discovery only (no production code modified)  
**Author role:** Backend 1 + Team Lead (Kirtan)  
**Date:** 2026-08-21  
**Target backend URL:** `http://127.0.0.1:8021` (team convention; see port note below)

---

## 1. Repository Structure

The combined workspace root is:

```
SIH-2026-backend-dinl(1)/
├── docs/                                    ← this document
├── SIH-2026-backend/                        ← BACKEND ROOT (+ AI + integration)
│   ├── backend/                             ← FastAPI application
│   │   ├── main.py                          ← unified app entry (uvicorn target)
│   │   ├── api/                             ← REST + WebSocket routers
│   │   ├── core/config.py                   ← sample rate, freqs, storage paths
│   │   ├── models/data_schemas.py           ← Pydantic models (ThreatEvent, etc.)
│   │   ├── services/                        ← threat, payload, audio, acoustic
│   │   └── scripts/                         ← hardware / AI test scripts
│   ├── ai/                                  ← AI ROOT
│   │   ├── dsp/                             ← DSP pipeline (production detection path)
│   │   ├── ml/                              ← ML training/inference (phase3–5)
│   │   ├── docs/                            ← AI specs and audits
│   │   └── tests/                           ← AI unit/integration tests
│   ├── integration/                         ← INTEGRATION FOLDER (Python AI→Backend client)
│   ├── tests/                               ← backend + E2E pytest suite
│   ├── scripts/                             ← PC1/PC2 demo, forensic analysis scripts
│   ├── docs/                                ← backend architecture + integration contract
│   ├── samples/                             ← reference WAV/JSON (Backend 2 handoff)
│   ├── isolated/                            ← non-SIH reference code (documented only)
│   ├── forensic_captures/                   ← runtime capture output
│   ├── generated_payloads/                  ← simulator WAV output (STORAGE_DIR default)
│   ├── requirements.txt                     ← Python dependencies
│   ├── README.md
│   └── .venv/                               ← local virtualenv (may be machine-specific)
└── SIH-2026-Frontend-Final-main ()/
    └── SIH-2026-Frontend-Final-main/        ← FRONTEND ROOT
        ├── src/                             ← React application source
        ├── public/                          ← static assets (3D models, icons)
        ├── package.json                     ← npm dependencies
        ├── vite.config.ts                   ← Vite config (no env/proxy yet)
        ├── tsconfig*.json
        └── README.md
```

| Area | Path | Notes |
|------|------|-------|
| **Frontend root** | `SIH-2026-Frontend-Final-main ()/SIH-2026-Frontend-Final-main/` | React + Vite SPA |
| **Backend root** | `SIH-2026-backend/` | FastAPI unified server |
| **AI root** | `SIH-2026-backend/ai/` | DSP + ML subsystems |
| **DSP root** | `SIH-2026-backend/ai/dsp/` | `DSPPipeline`, feature extraction |
| **ML root** | `SIH-2026-backend/ai/ml/` | phase3–5 training/inference (not API-wired) |
| **Integration folder** | `SIH-2026-backend/integration/` | Python `BackendThreatClient` → `POST /api/analyze` |
| **Scripts** | `SIH-2026-backend/scripts/`, `backend/scripts/` | E2E, hardware, forensic tooling |
| **Tests** | `SIH-2026-backend/tests/`, `ai/tests/` | pytest |
| **Config** | `backend/core/config.py`, `integration/config.py` | env vars: `STORAGE_DIR`, `BACKEND_API_URL` |
| **Environment files** | **None found** (no `.env` in repo) | ports/URLs set via CLI or env at runtime |
| **Python requirements** | `SIH-2026-backend/requirements.txt` | fastapi, uvicorn, numpy, scipy, sounddevice, httpx, pytest |
| **Frontend dependencies** | `package.json` | react 19, react-router-dom 7, vite 8, three.js, framer-motion, tailwind 4 |

**Port note:** README and `integration/config.py` default to port **8000**. Team lead specifies **8021** for integration — no hardcoded 8021 exists in source; use `uvicorn backend.main:app --port 8021` and `VITE_API_BASE_URL` (to be added) at integration time.

---

## 2. Frontend Architecture

| Aspect | Detail |
|--------|--------|
| **Framework** | React 19 + TypeScript |
| **Build tool** | Vite 8 |
| **Entry point** | `src/main.tsx` → wraps `App` in `BrowserRouter`, `IntroProvider`, `FrontendSettingsProvider` |
| **Routing** | React Router v7 (`src/App.tsx`) |
| **State management** | React `useState`/`useContext` only — no Redux/Zustand/React Query |
| **Auth** | **None** — `Login.tsx` is demo-only; any username/password navigates to `/dashboard` |
| **API client** | **Does not exist yet** — README says future adapters under `src/services/` |
| **Environment config** | **None** — no `VITE_*` vars, no `.env`, no proxy in `vite.config.ts` |
| **WebSocket** | **Not used** in frontend |
| **Loading/error states** | Local UI state only (timers, form validation); no network loading patterns |

### Context / hooks

| Module | Purpose |
|--------|---------|
| `context/IntroContext.tsx` | Cinematic intro sequence (`sessionStorage`) |
| `context/FrontendSettingsContext.tsx` | Local UI prefs (viz density, demo labels) |
| `hooks/useFrontendSettings.ts` | Settings accessor |
| `hooks/useIntro.ts` | Intro replay control |
| `hooks/useWebGLSupport.ts` | WebGL capability probe |

### Visualization components (canvas-only, no live audio)

`Waveform`, `Spectrum`, `Spectrogram`, `RadarSweep`, `RadialScore`, `Pipeline` — all render **simulated/procedural** graphics, not backend audio streams.

---

## 3. Backend Architecture

| Aspect | Detail |
|--------|--------|
| **Entry point** | `backend/main.py` — single FastAPI app combining Backend 1 + Backend 2 |
| **Run command** | `python -m uvicorn backend.main:app --reload --port 8021` (from `SIH-2026-backend/`) |
| **Version** | `APP_VERSION = "2.0.0"` (`backend/core/config.py`) |
| **Routers** | `simulator`, `audio`, `analysis`, `threats` + inline system routes |
| **Services** | `threat_service`, `payload_service`, `audio_service`, `acoustic_service` |
| **Models/schemas** | `backend/models/data_schemas.py` |
| **Middleware** | `CORSMiddleware` only |
| **CORS** | `allow_origins=["*"]`, `allow_credentials=True`, all methods/headers |
| **API versioning** | **None** — flat paths (`/api/*`, `/simulate/*`, `/stream/*`) |
| **Storage** | In-memory Python lists/dicts; threats lost on restart |
| **Database** | None |
| **Auth** | None |
| **Static files** | `/files` → `STORAGE_DIR` (generated WAV payloads) |

### Router map

```
backend/main.py
├── GET  /health                         → system_status.health_check()
├── GET  /api/system-status              → Backend 1 status
├── include simulator.router  (/simulate)
├── include audio.router      (/stream)
├── include analysis.router   (/api)     → POST /api/analyze
└── include threats.router      (/api)     → GET /api/threats, /api/threats/current
```

### Service responsibilities

| Service | Role |
|---------|------|
| `threat_service.py` | In-memory `ThreatEvent` list; add/get/latest |
| `payload_service.py` | BFSK encode/decode, WAV save, speaker playback |
| `audio_service.py` | Mic/file streaming, WebSocket broadcast, forensic decode events |
| `acoustic_service.py` | CLI mic listener (standalone script pattern, not mounted as API) |

---

## 4. AI Architecture

### Pipeline (actual code path)

```
RAW AUDIO (48 kHz mono float/int16)
    ↓
ai/dsp/audio_preprocessing.py     DC removal, pre-emphasis, bandpass 17.5–21.5 kHz
    ↓
ai/dsp/feature_extraction.py      32 ML-ready features per 2048-sample chunk
    ↓
ai/dsp/frequency_analyzer.py      Peak detection, modulation classification
    ↓
ai/dsp/segmentation.py            Event duration accumulation
    ↓
ai/dsp/dsp_api.py  DSPPipeline    Accumulated verdict across chunks
    ↓
to_threat_event()                 13-field ThreatEvent dict (schema 1.0.0-dsp)
    ↓
integration/backend_client.py     POST /api/analyze (Python httpx)
    ↓
backend/services/threat_service   In-memory store
    ↓
GET /api/threats, /api/threats/current
```

### ML layer (parallel, not API-connected)

| Path | Status |
|------|--------|
| `ai/ml/phase3/` | Classifier training |
| `ai/ml/phase4/` | Calibration |
| `ai/ml/phase5/` | `InferenceService` — 32-feature vector → calibrated risk |
| Backend API | **No endpoint** exposes ML inference to frontend |

ML output is **not** merged into `ThreatEvent` today. Production path is **DSP-only** via `DSPPipeline.to_threat_event()`.

### ThreatEvent fields (actual — do not invent)

Source: `backend/models/data_schemas.py`, `ai/dsp/dsp_api.py`

| Field | Type | Available to frontend via |
|-------|------|----------------------------|
| `schema_version` | string (`"1.0.0-dsp"`) | `GET /api/threats`, `/api/threats/current` |
| `detected` | bool | same |
| `confidence` | float 0–1 | same |
| `risk` | `"LOW"` \| `"MEDIUM"` \| `"HIGH"` | same |
| `suspicion_score` | float 0–1 | same |
| `frequency_start` | float \| null (Hz) | same |
| `frequency_end` | float \| null (Hz) | same |
| `carrier_freqs` | float[] (Hz) | same |
| `duration` | float (seconds) | same |
| `pattern` | `"none"` \| `"tone"` \| `"fsk"` \| `"ook"` \| `"chirp"` | same |
| `snr` | float (dB) | same |
| `chunks_analyzed` | int | same |
| `timestamp` | float (Unix epoch) | same |

**Not in ThreatEvent:** `event_id`, `classification` (human label), `threatScore` (0–100), `signalStrength` (dBm), `label`, `severity`.

### Forensic decode events (separate from ThreatEvent)

WebSocket `/stream/forensics` and `GET /stream/status` → `latest_forensics` use `DecodedPayloadEvent`:

- `event_type`, `stream_id`, `recovered_text`, `confidence`, `bit_count`, `preamble_found`, `timestamp`

This is **Backend 2 decoder output**, not AI ThreatEvent.

---

## 5. API Inventory

Generated from live FastAPI OpenAPI (`backend.main:app`) + source inspection.  
Base URL: `http://127.0.0.1:8021`

### System

#### `GET /health`

| | |
|---|---|
| **Purpose** | Backend 2 extended health (sample rate, freqs, stream state) |
| **Query params** | none |
| **Response 200** | `{ status, service, version, sample_rate, freq_0, freq_1, bit_duration, storage_dir, total_payloads_generated, stream_active }` |
| **Errors** | none defined |

#### `GET /api/system-status`

| | |
|---|---|
| **Purpose** | Backend 1 lightweight online check for frontend |
| **Response 200** | `{ "status": "online", "service": "acoustic-shield-backend", "version": "2.0.0" }` |

---

### Threats (Backend 1)

#### `GET /api/threats`

| | |
|---|---|
| **Purpose** | Full threat history (arrival order) |
| **Response 200** | `{ "threats": ThreatEvent[] }` — empty array if none |
| **Errors** | none |

#### `GET /api/threats/current`

| | |
|---|---|
| **Purpose** | Latest stored threat |
| **Response 200 (has threat)** | `{ "current": ThreatEvent }` |
| **Response 200 (empty)** | `{ "current": null, "message": "No threats have been recorded yet" }` |

#### `POST /api/analyze`

| | |
|---|---|
| **Purpose** | AI/integration layer submits ThreatEvent |
| **Request body** | `ThreatEvent` (13 required fields) |
| **Response 200** | `{ "status": "received", "event": ThreatEvent }` |
| **Response 422** | Pydantic validation error |
| **Frontend rule** | **Must NOT call directly** (per integration contract) |

---

### Simulator (Backend 2)

#### `POST /simulate/generate`

| | |
|---|---|
| **Request body** | `GeneratePayloadRequest`: `text` (max 256), `freq_0_hz` (default 18500), `freq_1_hz` (default 20500), `bit_duration_ms` (default 50) |
| **Response 200** | `GeneratePayloadResponse`: `payload_id`, `duration_sec`, `bit_count`, `wav_url` |
| **Errors** | 422 validation |

#### `POST /simulate/transmit`

| | |
|---|---|
| **Request body** | `TransmitRequest`: `payload_id`, `mode` (`"virtual"` \| `"audio"`) |
| **Response 200** | `TransmitStatus`: `payload_id`, `state`, `progress_pct` |
| **Errors** | 404 unknown payload, 500 speaker failure |

#### `POST /simulate/acoustic-exfiltrate`

| | |
|---|---|
| **Request body** | `AcousticExfiltrateRequest`: `source_type`, `custom_text`, freqs, `bit_duration_ms`, `emit_audio` |
| **Response 200** | `AcousticExfiltrateResponse`: `status`, `text_length`, `duration_sec`, `bit_count`, `payload_id`, `wav_url`, `preview_text` |

#### `GET /simulate/status/{payload_id}`

| | |
|---|---|
| **Path param** | `payload_id` |
| **Response 200** | `TransmitStatus` |
| **Errors** | 404 |

#### `GET /simulate/list`

| | |
|---|---|
| **Response 200** | Array of `{ payload_id, text, state, wav_url }` |

---

### Audio streaming (Backend 2)

#### `GET /stream/devices`

| | |
|---|---|
| **Response 200** | `{ input_devices: [...], default_input }` or `{ error, input_devices: [] }` |

#### `POST /stream/start`

| | |
|---|---|
| **Query params** | `source` (`mic` \| `file`), `file_path` (required if file), `device_id` (optional) |
| **Response 200** | `{ status, stream_id, source, sample_rate, window_sec }` |
| **Response error** | `{ "error": "file_path parameter is required..." }` (200 with error object) |

#### `POST /stream/stop`

| | |
|---|---|
| **Response 200** | `{ "status": "stopped" }` or `{ "status": "no active stream" }` |

#### `GET /stream/status`

| | |
|---|---|
| **Response 200 (active)** | `{ active: true, stream_id, subscribers, forensic_subscribers, latest_forensics: DecodedPayloadEvent[] }` |
| **Response 200 (idle)** | `{ active: false, subscribers: 0, latest_forensics: [] }` |

#### `WS /stream/audio`

| | |
|---|---|
| **Purpose** | Subscribe to `AudioChunkMessage` JSON (`stream_id`, `seq`, `sample_rate`, `samples_b64`, `timestamp`, `duration_sec`) |
| **Prerequisite** | Active stream via `POST /stream/start` |
| **Close** | 1011 if no active stream |

#### `WS /stream/forensics`

| | |
|---|---|
| **Purpose** | Subscribe to `DecodedPayloadEvent` when BFSK decoder recovers text |
| **Prerequisite** | Active stream |

---

### Static

#### `GET /files/{payload_id}.wav`

Served from `STORAGE_DIR` via StaticFiles mount.

---

## 6. Frontend Route Inventory

| Route | Component | Purpose | Current data source | API dependency | Mock/static |
|-------|-----------|---------|---------------------|----------------|-------------|
| `/` | `Home.tsx` | Marketing + intro + demo telemetry | Hardcoded JSX + canvas viz | **None** | Inline demo event stream, spectrum labels |
| `/login` | `Login.tsx` | Demo gate | Local form state | **None** | No auth API |
| `/dashboard` | `Dashboard.tsx` | Security overview | `mock.ts` | **None** | `dashboardDemo`, `liveEvents` |
| `/intelligence` | `Intelligence.tsx` | AI workstation view | `mock.ts` + hardcoded stats | **None** | `aiReasoning`, `signalCharacteristics`, `threatScore` |
| `/monitor`, `/monitoring`→redirect | `Monitoring.tsx` | Live monitor lifecycle | `mock.ts` + timer FSM | **None** | `monitoringDemoAnalysis`, `monitoringComponentStatuses` |
| `/events` | `Events.tsx` | Threat history table | `mock.ts` | **None** | `liveEvents` |
| `/events/:id` | `EventDetails.tsx` | Single event drill-down | `mock.ts` | **None** | `liveEvents` lookup by `EVT-*` id |
| `/simulator`, `/attack-lab`→redirect | `AttackLab.tsx` | Attack lab simulation | `mock.ts` + timer FSM | **None** | `simulatorDefaults`, `liveEvents` |
| `/system` | `System.tsx` | Module readiness | `mock.ts` | **None** | `systemDemoStatus` |
| `/settings` | `Settings.tsx` | UI preferences | `localStorage` via context | **None** | N/A |
| `/about` | `About.tsx` | Team/info | `team.ts` placeholders | **None** | Placeholder bios |
| `*` | `NotFound.tsx` | 404 | static | **None** | N/A |

**Frontend dev origin:** `http://localhost:5173` (per frontend README).

---

## 7. Frontend → API Mapping

| Frontend feature | Frontend location | Backend endpoint | Method | Data | Status |
|------------------|-------------------|------------------|--------|------|--------|
| Login | `Login.tsx` | — | — | — | **NOT CONNECTED** (demo bypass) |
| Dashboard posture | `Dashboard.tsx` | `GET /api/threats/current` | GET | `detected`, `risk`, `confidence`, `suspicion_score`, carriers | **MOCK ONLY** |
| Dashboard metrics | `Dashboard.tsx` | `GET /api/threats/current` | GET | frequency, duration, pattern | **MOCK ONLY** |
| Dashboard recent events | `Dashboard.tsx` | `GET /api/threats` | GET | threat list | **MOCK ONLY** |
| Live monitor | `Monitoring.tsx` | `POST /stream/start`, `WS /stream/audio`, `GET /api/threats/current` | POST/WS/GET | audio chunks + threat | **MOCK ONLY** |
| Events list | `Events.tsx` | `GET /api/threats` | GET | `threats[]` | **MOCK ONLY** |
| Event details | `EventDetails.tsx` | `GET /api/threats` (by index/timestamp) | GET | single ThreatEvent | **MOCK ONLY** — no backend event IDs |
| Attack simulator | `AttackLab.tsx` | `POST /simulate/generate`, `/simulate/transmit` | POST | payload_id, wav_url, state | **MOCK ONLY** |
| Intelligence / AI view | `Intelligence.tsx` | `GET /api/threats/current` | GET | AI fields | **MOCK ONLY** |
| System status | `System.tsx` | `GET /api/system-status`, `GET /health` | GET | online, version, modules | **MOCK ONLY** |
| Threat history | `Events.tsx` | `GET /api/threats` | GET | full history | **MOCK ONLY** |
| Audio/streaming viz | `Waveform`, `Spectrum`, `Spectrogram` | `WS /stream/audio` | WS | `AudioChunkMessage.samples_b64` | **NOT CONNECTED** |
| Forensic decode feed | — (no UI yet) | `WS /stream/forensics` | WS | `DecodedPayloadEvent` | **NO UI** |
| AI results ingestion | — (backend-side) | `POST /api/analyze` | POST | ThreatEvent | **Python client only** |

---

## 8. Mock Data Locations

| File | Line / component | Current data | Replacement API |
|------|------------------|--------------|-----------------|
| `src/data/mock.ts` | entire file | Central demo dataset | Adapter layer mapping `ThreatEvent` → UI types |
| `mock.ts` | `liveEvents[]` | 8 fake `SignalEvent` records (`EVT-1041`…) | `GET /api/threats` |
| `mock.ts` | `dashboardDemo` | Static high-risk dashboard | `GET /api/threats/current` |
| `mock.ts` | `monitoringDemoAnalysis` | Static analysis rows | `GET /api/threats/current` + stream status |
| `mock.ts` | `monitoringComponentStatuses` | FSM component states | Derive from `/stream/status` + threat state |
| `mock.ts` | `simulatorDefaults` | Payload/carrier defaults | `POST /simulate/generate` defaults from backend config |
| `mock.ts` | `systemDemoStatus` | Module readiness table | `GET /api/system-status` + `GET /health` |
| `mock.ts` | `aiReasoning` | Static reasoning bullets | **No backend field** — omit or future AI explainability |
| `mock.ts` | `signalCharacteristics` | GHz-band fake metrics | Map from `carrier_freqs`, `snr`, `duration` |
| `mock.ts` | `threatScore` (=87) | 0–100 score | Map from `suspicion_score * 100` |
| `Dashboard.tsx` | imports mock | All dashboard content | `/api/threats/current`, `/api/threats` |
| `Monitoring.tsx` | timer FSM | Simulated lifecycle | `/stream/start` + WS + `/api/threats/current` polling |
| `Events.tsx` | `liveEvents` | Event table | `/api/threats` |
| `EventDetails.tsx` | `liveEvents.find(id)` | Event by `EVT-*` | Map threats by `timestamp` or generated client-side ID |
| `AttackLab.tsx` | timer FSM | Simulated pipeline | `/simulate/generate`, `/simulate/transmit`, `/simulate/status/{id}` |
| `Intelligence.tsx` | hardcoded + mock | "2.41 GHz", "94%", score 87 | `/api/threats/current` |
| `System.tsx` | `systemDemoStatus` | "NOT CONNECTED" row | Live fetch to `/api/system-status` |
| `Login.tsx` | `handleSubmit` | Navigates without API | Future auth TBD (no backend auth exists) |
| `Home.tsx` | telemetry section | Inline fake events, GHz spectrum | Optional `/api/threats` teaser |
| `data/team.ts` | all entries | Placeholder team bios | Static (non-API) |

---

## 9. API Client Details

### Frontend API client

**Status: MISSING**

- No `src/services/` directory
- No `fetch`, `axios`, or WebSocket usage anywhere in `src/`
- No TypeScript types for `ThreatEvent`
- Frontend README explicitly defers adapters to a future `src/services/` layer

**Recommended integration target:**

| Setting | Value |
|---------|-------|
| Base URL | `import.meta.env.VITE_API_BASE_URL` → `http://127.0.0.1:8021` |
| HTTP | native `fetch` or lightweight wrapper |
| WebSocket | `new WebSocket(\`${wsBase}/stream/audio\`)` after `/stream/start` |
| Auth | none for MVP (backend has no auth) |
| Error handling | to be implemented (no patterns exist yet) |
| Types | mirror backend `ThreatEvent` 13 fields |

### Existing backend-side client (Python — not frontend)

| Item | Detail |
|------|--------|
| File | `integration/backend_client.py` |
| Library | `httpx` |
| Base URL | `BACKEND_API_URL` env, default `http://127.0.0.1:8000` |
| Endpoints used | `POST /api/analyze`, `GET /api/threats/current` |
| Behavior | No field renaming; returns `SubmitResult` |

---

## 10. CORS Status

### Current backend configuration

```python
# backend/main.py
CORSMiddleware(
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

| Item | Status |
|------|--------|
| Frontend origin | `http://localhost:5173` (Vite default) |
| Backend origin | `http://127.0.0.1:8021` (team target) |
| CORS blocking | **Unlikely** — wildcard `*` allows all origins |
| Credentials + wildcard | Spec-wise awkward combo; works for simple GET/fetch without cookies |
| WebSocket CORS | Separate handshake; wildcard HTTP CORS does not auto-configure WS — test explicitly |

### Integration requirements

1. Add `VITE_API_BASE_URL=http://127.0.0.1:8021` (no code change to CORS needed for dev)
2. Optionally tighten `allow_origins` to `[http://localhost:5173]` before production
3. Verify WebSocket from browser to `ws://127.0.0.1:8021/stream/audio`

---

## 11. Contract Mismatch Report

### Endpoint path mismatches

| Frontend expects (implicit) | Backend provides | Severity |
|----------------------------|------------------|----------|
| `/api/events` | `/api/threats` | **HIGH** — path name differs |
| Event detail `/events/:id` | No server-side IDs | **HIGH** — must synthesize or extend schema |
| `/api/login` or auth | No auth endpoints | **MEDIUM** — demo login only today |

### Field / schema mismatches

| Frontend (`SignalEvent` / UI) | Backend (`ThreatEvent`) | Notes |
|------------------------------|-------------------------|-------|
| `id: "EVT-1041"` | *(no field)* | Use `timestamp` or index as client ID |
| `severity: normal\|anomaly\|threat` | `risk: LOW\|MEDIUM\|HIGH` | Needs mapping function |
| `threatScore: 87` (0–100) | `suspicion_score: 0.91` (0–1) | Scale or rename in adapter |
| `confidence: "94%"` | `confidence: 0.94` | Format vs float |
| `frequency: "2.41 GHz"` | `carrier_freqs: [18492, 20507]` Hz | **Unit/band mismatch** — UI shows RF demo data, backend is ultrasonic kHz |
| `frequencyStart/End` strings | `frequency_start/end` floats (Hz) | Format + unit |
| `signalStrength: "-42 dBm"` | `snr: 18.5` (dB) | Different semantics |
| `classification: string` | `pattern: tone\|fsk\|...` | Different vocabulary |
| `label: "Signal acquired"` | *(no field)* | Client-generated summary |
| `time: "14:52:03"` | `timestamp: unix float` | Format conversion |
| `detected` implicit in severity | `detected: bool` | Align dashboard "safe" with `detected:false` |
| Dashboard `status: safe\|suspicious\|high-risk` | `risk` + `detected` | Composite mapping required |

### Behavioral mismatches

| Topic | Frontend | Backend |
|-------|----------|---------|
| Monitoring lifecycle | JavaScript timers | Real mic stream + AI pipeline |
| Simulator | Local animation | `/simulate/*` generates real WAV + optional speaker |
| Threat persistence | Static array in bundle | In-memory, cleared on restart |
| AI reasoning text | Static bullets in mock | Not produced by backend |
| Event spectrogram/waveform | Procedural canvas | Would need `/stream/audio` or file replay |

### Port / URL mismatches

| Source | URL |
|--------|-----|
| Team integration target | `http://127.0.0.1:8021` |
| `integration/config.py` default | `http://127.0.0.1:8000` |
| Backend README | port 8000 |
| Frontend | no URL configured |

---

## 12. Duplicate Implementations

| Area | Locations | Notes — do not delete |
|------|-----------|----------------------|
| **Threat detection logic** | `ai/dsp/DSPPipeline` vs `payload_service.decode_signal_to_text` | DSP = real detection; decoder = simulator self-test / forensic only |
| **Mic listening** | `audio_service.LiveMicSource` vs `acoustic_service.run_listener` | API stream vs CLI script — overlapping decode path |
| **Forensic decode in stream** | `audio_service` sliding buffer decode | Same BFSK decoder as simulator, not AI ThreatEvent |
| **System health endpoints** | `GET /health` vs `GET /api/system-status` | Different payload shapes; frontend should prefer `/api/system-status` |
| **Backend HTTP clients** | `integration/backend_client.py` (Python) | Only client today; frontend client still to be created — not a duplicate |
| **ThreatEvent types** | `backend/models/data_schemas.py` vs `src/data/mock.ts SignalEvent` | Parallel schemas; adapter must bridge |
| **Integration docs** | `docs/integration-contract.md` vs this map | Complementary; contract is normative for AI↔Backend |
| **Legacy stubs** | `ReverseShellCommandRequest` in schemas | No routes — documented in `isolated/README.md` |
| **ML vs DSP risk scoring** | `DSPPipeline.score_to_risk` vs `ai/ml/phase5 InferenceService` | ML not wired to API; potential future overlap |

---

## 13. Integration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **No frontend API layer** | Blocks all integration | Phase B: create `src/services/apiClient.ts` + types first |
| **Schema/domain mismatch (GHz vs kHz)** | UI will show wrong values if mapped naively | Adapter must reformat Hz → kHz labels; update copy from "2.41 GHz" to ultrasonic band |
| **No event IDs** | `/events/:id` routes break | Derive stable client IDs from `timestamp` or add backend field later |
| **In-memory threats** | Empty UI after backend restart | Document; optional seed via `POST /api/analyze` in demo script |
| **No live AI→Backend→Frontend loop** | Dashboard stays empty without manual POST | Run `integration/run_dsp_to_backend.py` or E2E scripts during demo |
| **Dual health endpoints** | Confusion in System page | Use `/api/system-status` for frontend; `/health` for ops |
| **WebSocket + canvas viz gap** | Monitoring viz won't reflect real audio without decode | Phase H: decode `samples_b64` or keep viz as status-driven |
| **Port 8021 not in repo defaults** | Wrong URL if env omitted | Standardize env docs + startup scripts |
| **ML not in API** | Intelligence page "AI reasoning" has no backend source | Hide or keep static until AI 2 deliverable |
| **CORS credentials + wildcard** | Edge-case browser issues | Tighten origins in production |
| **No auth** | Login page is cosmetic | Accept for SIH demo or defer Phase C |

---

## 14. Recommended Integration Order

### PHASE A — Discovery ✅ (this document)

Complete repository, API, mock, and mismatch inventory.

### PHASE B — API / client connection

1. Add `VITE_API_BASE_URL=http://127.0.0.1:8021` + `src/services/apiClient.ts`
2. Add TypeScript `ThreatEvent` type matching 13 backend fields
3. Add `getSystemStatus`, `getThreats`, `getCurrentThreat` methods
4. Wire health check on app load (optional banner)

### PHASE C — Authentication

- **Skip for MVP** unless backend auth is added — current login remains demo bypass

### PHASE D — Dashboard integration

1. Replace `dashboardDemo` with `GET /api/threats/current`
2. Map `risk`/`detected`/`suspicion_score` → dashboard status + RadialScore
3. Recent events from `GET /api/threats` (slice last N)

### PHASE E — Monitoring integration

1. `POST /stream/start?source=mic`
2. Optional `WS /stream/audio` for future viz
3. Poll `GET /api/threats/current` for classification result
4. Replace timer FSM with real stream + threat state

### PHASE F — Events integration

1. List from `GET /api/threats`
2. Detail page keyed by `timestamp` (or generated `id`)
3. Remove dependency on `EVT-*` mock IDs

### PHASE G — Simulator integration

1. `POST /simulate/generate` with payload text + freqs from UI controls
2. `POST /simulate/transmit` with `mode=audio|virtual`
3. Poll `GET /simulate/status/{payload_id}`
4. Link to threat flow after AI pipeline runs (not automatic today)

### PHASE H — Real-time / WebSocket integration

1. Connect `/stream/audio` and `/stream/forensics`
2. Surface forensic decode in UI (new component or monitor log)
3. Optional: drive Waveform from decoded chunks

### PHASE I — AI / DSP / ML integration

- **Backend-side (existing):** `integration/run_dsp_to_backend.py`
- **Frontend:** display only via `/api/threats*` — never call `/api/analyze`
- ML phase5 remains offline until API exists

### PHASE J — End-to-end validation

1. Start backend on **8021**
2. Start frontend on **5173**
3. Run simulator or mic capture → DSP → POST analyze → verify dashboard/events
4. Checklist: system status online, current threat renders, history populates, simulator generates WAV

---

## Appendix A — Frontend dependency summary

```json
"dependencies": {
  "@react-three/drei", "@react-three/fiber", "framer-motion",
  "react", "react-dom", "react-router-dom", "three"
}
```

No HTTP client library — native `fetch` is sufficient.

---

## Appendix B — Backend startup (integration target port)

```bash
cd SIH-2026-backend
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8021 --reload
```

OpenAPI: `http://127.0.0.1:8021/docs`

---

## Appendix C — Key source references

| Topic | File |
|-------|------|
| FastAPI app | `SIH-2026-backend/backend/main.py` |
| ThreatEvent schema | `SIH-2026-backend/backend/models/data_schemas.py` |
| AI pipeline | `SIH-2026-backend/ai/dsp/dsp_api.py` |
| Python backend client | `SIH-2026-backend/integration/backend_client.py` |
| Frontend routes | `SIH-2026-Frontend-Final-main ()/.../src/App.tsx` |
| Mock data | `SIH-2026-Frontend-Final-main ()/.../src/data/mock.ts` |
| Integration contract | `SIH-2026-backend/docs/integration-contract.md` |

---

*End of discovery document. No production code was modified.*
