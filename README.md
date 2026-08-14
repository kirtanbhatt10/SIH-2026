# SIH 2026 — Acoustic Cybersecurity System

> **AI-powered detection and analysis of covert communication through acoustic and ultrasonic channels.**

An interdisciplinary cybersecurity system developed for **Smart India Hackathon (SIH) 2026** to detect, analyze, classify, and demonstrate potential covert communication carried through acoustic and ultrasonic signals.

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/TejasPadia-21/SIH--2026.git
cd SIH--2026
git checkout backend

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python -m uvicorn backend.main:app --reload --port 8000

# 5. Open Swagger docs
# http://localhost:8000/docs
```

---

## Project Architecture

```
SIH--2026/
├── backend/
│   ├── main.py              ← Unified FastAPI application
│   ├── api/
│   │   ├── analysis.py      ← POST /api/analyze (threat intake)
│   │   ├── threats.py       ← GET /api/threats, /api/threats/current
│   │   ├── simulator.py     ← /simulate/* (payload generation)
│   │   ├── audio.py         ← /stream/* (WebSocket audio streaming)
│   │   └── system_status.py ← Health check & state management
│   ├── core/
│   │   └── config.py        ← Configuration (sample rate, frequencies)
│   ├── models/
│   │   └── data_schemas.py  ← All Pydantic models
│   └── services/
│       ├── threat_service.py    ← In-memory threat storage
│       ├── payload_service.py   ← Encode/decode/Goertzel (BFSK codec)
│       ├── audio_service.py     ← Audio streaming & forensic detection
│       └── acoustic_service.py  ← CLI acoustic listener
├── docs/
│   └── integration-contract.md  ← Cross-team integration contract
├── tests/
│   ├── test_payload_roundtrip.py
│   ├── test_backend1_api.py
│   └── test_integration.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## API Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/system-status` | Backend status (online/offline) |
| `GET` | `/health` | Detailed health check (frequencies, payload count) |
| `GET` | `/docs` | Swagger/OpenAPI documentation |

### Threat Management (Backend 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Submit a ThreatEvent from AI pipeline |
| `GET` | `/api/threats` | List all recorded threat events |
| `GET` | `/api/threats/current` | Get the most recent threat event |

### Simulator (Backend 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulate/generate` | Generate ultrasonic payload WAV |
| `POST` | `/simulate/transmit` | Transmit payload (audio/virtual) |
| `POST` | `/simulate/acoustic-exfiltrate` | High-level acoustic exfiltration |
| `GET` | `/simulate/status/{id}` | Check payload transmission status |
| `GET` | `/simulate/list` | List all generated payloads |

### Audio Streaming (Backend 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/stream/audio` | Live audio chunk broadcast |
| `WS` | `/stream/forensics` | Decoded payload event stream |
| `POST` | `/stream/start` | Start audio capture (mic/file) |
| `POST` | `/stream/stop` | Stop audio capture |
| `GET` | `/stream/status` | Stream status & subscriber count |
| `GET` | `/stream/devices` | List audio input devices |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Data Flow

```
Controlled Payload
    → Encoding (BFSK: 18.5kHz / 21kHz)
    → Modulation
    → Audio Signal / WAV
    → Speaker
    → Air
    → Microphone
    → Detector / Audio Acquisition
    → Signal Processing
    → Feature Extraction
    → AI Classification / Threat Scoring
    → Backend API (POST /api/analyze)
    → Threat Storage
    → Frontend Dashboard
```

---

## Integration Contract

See [docs/integration-contract.md](docs/integration-contract.md) for the full cross-team data contract covering:
- Simulator → Detector handoff
- Detector → AI handoff
- AI → Backend contract (ThreatEvent schema — TEMPORARY)
- Backend → Frontend endpoints

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Framework | FastAPI |
| Server | Uvicorn |
| Validation | Pydantic v2 |
| DSP | NumPy + Goertzel algorithm |
| Audio | sounddevice |
| Storage | In-memory (temporary) |

---

**Built for SIH 2026 — where signal processing meets cybersecurity.**
