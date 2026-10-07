# The Silent Dog's Whistle

**An acoustic cybersecurity system that detects covert data channels hidden in ultrasonic sound. Built for Smart India Hackathon 2026.**

Malware on an air-gapped or locked-down machine can leak data through its speakers at frequencies
people cannot hear. This project builds both sides of that problem in a controlled lab: a simulator
that transmits a known payload over ultrasound, and a detector that listens with an ordinary
microphone, finds the signal, classifies it and raises a threat event on a live dashboard.

> **Scope.** Everything here runs against payloads we generate ourselves, on our own hardware. It
> is a detection and research project, not a tool for exfiltrating data from systems you do not own.

## How it works

```text
Controlled payload
  → BFSK encoding (18.5 kHz / 21 kHz)  → WAV → speaker
  → air
  → microphone → audio acquisition
  → DSP pipeline (preprocess, FFT/STFT, 32-feature vector)
  → classification and threat scoring
  → Backend API (POST /api/analyze) → threat storage
  → Dashboard (REST and WebSocket)
```

## What is in this repository

| Part | Path | What it is |
| --- | --- | --- |
| **Backend** | [`SIH-2026-backend/backend/`](SIH-2026-backend/backend) | FastAPI server: threat intake and storage, payload simulator, audio streaming over WebSocket |
| **DSP** | [`SIH-2026-backend/ai/dsp/`](SIH-2026-backend/ai/dsp) | Signal-processing pipeline that turns microphone audio into features, spectrogram frames and a threat event |
| **ML** | [`SIH-2026-backend/ai/ml/`](SIH-2026-backend/ai/ml) | Dataset generation, model comparison and training experiments |
| **Integration** | [`SIH-2026-backend/integration/`](SIH-2026-backend/integration) | Python client that posts detections from the AI pipeline to the backend |
| **Frontend** | [`SIH-2026-Frontend-Final-main ()/SIH-2026-Frontend-Final-main/`](SIH-2026-Frontend-Final-main%20%28%29/SIH-2026-Frontend-Final-main) | React dashboard: monitoring, events, attack lab, intelligence and system pages |
| **Docs** | [`docs/`](docs), [`SIH-2026-backend/docs/`](SIH-2026-backend/docs), [`SIH-2026-backend/ai/docs/`](SIH-2026-backend/ai/docs) | Integration map, cross-team contract, detection spec, audits and known issues |

## My role

Team lead and Backend 1. I owned the threat-management API (`/api/analyze`, `/api/threats`), the
AI-to-backend contract, and the frontend-to-backend integration plan in
[`docs/frontend-backend-integration-map.md`](docs/frontend-backend-integration-map.md).

## Install and run

**Requirements:** Python 3.12, Node.js 20 or later, and a microphone and speaker for the physical demo.

**1. Backend**

```bash
git clone https://github.com/kirtanbhatt10/SIH-2026.git
cd SIH-2026/SIH-2026-backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8021
```

Interactive API docs are then at <http://127.0.0.1:8021/docs>.

**2. Frontend** (in a second terminal)

```bash
cd "SIH-2026/SIH-2026-Frontend-Final-main ()/SIH-2026-Frontend-Final-main"
npm install
cp .env.example .env               # VITE_API_BASE_URL=http://127.0.0.1:8021
npm run dev                        # http://localhost:5174
```

**3. Tests**

```bash
cd SIH-2026-backend
python -m pytest tests/ -v                      # backend and end-to-end
cd ai && pip install -r requirements-dev.txt
python -m pytest tests/ -q                      # DSP
```

A two-machine transmit and receive demo is described in
[`SIH-2026-backend/MULTI_PC_DEMO_GUIDE.md`](SIH-2026-backend/MULTI_PC_DEMO_GUIDE.md).

## API at a glance

| Area | Endpoints |
| --- | --- |
| System | `GET /health`, `GET /api/system-status` |
| Threats | `POST /api/analyze`, `GET /api/threats`, `GET /api/threats/current` |
| Simulator | `POST /simulate/generate`, `POST /simulate/transmit`, `POST /simulate/acoustic-exfiltrate`, `GET /simulate/list`, `GET /simulate/status/{id}` |
| Audio stream | `WS /stream/audio`, `WS /stream/forensics`, `POST /stream/start`, `POST /stream/stop`, `GET /stream/status`, `GET /stream/devices` |

Full details are in the [backend README](SIH-2026-backend/README.md).

## Tech stack

| Layer | Used |
| --- | --- |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic v2, WebSockets |
| Signal processing | NumPy, SciPy, Goertzel algorithm, sounddevice |
| Frontend | React 19, TypeScript, Vite, React Router, Tailwind CSS 4, three.js, Framer Motion |
| Tests | pytest, httpx |

## Honest limitations

- Threat events are held in memory and are lost on restart.
- The login page is a demo: there is no real authentication.
- The ML models are experiments and are not wired into the API; the live detection path is the DSP pipeline.
- Some detection claims are weaker than they first looked, for example OOK classification is only about 70% reliable on synthetic input. These are written up in [`KNOWN_ISSUES.md`](SIH-2026-backend/ai/docs/KNOWN_ISSUES.md).
