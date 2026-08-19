# PC1 → PC2 Acoustic Test Guide

Complete terminal-based testing for the SIH 2026 Acoustic Shield pipeline.

---

## Architecture Summary

| Component | Location |
|-----------|----------|
| Simulator API | `backend/api/simulator.py` |
| Payload encoder | `backend/services/payload_service.py` → `encode_text_to_signal()` |
| BFSK modulator | `backend/services/payload_service.py` → `_bits_to_tone()` |
| WAV generator | `backend/services/payload_service.py` → `save_signal_to_wav()` |
| Speaker playback | `backend/services/payload_service.py` → `play_ultrasonic_signal()` |
| Microphone capture | `scripts/pipeline_common.py` → `capture_microphone()` |
| Audio streaming (API) | `backend/api/audio.py` + `audio_service.py` |
| AI/DSP pipeline | `ai/dsp/dsp_api.py` → `DSPPipeline` |
| ThreatEvent schema | `backend/models/data_schemas.py` |
| POST /api/analyze | `backend/api/analysis.py` |
| GET /api/threats | `backend/api/threats.py` |
| AI → Backend client | `integration/backend_client.py` (no duplicate `ai_integration.py`) |

---

## 1. PC1 Setup

PC1 is the **attack simulator** — encodes text, modulates BFSK, saves WAV, plays speaker.

### Python environment

```powershell
cd g:\Tryhackme\SIH-2026-backend
conda activate sih
pip install -r requirements.txt
```

### Backend startup (optional on PC1 if using API)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### PC1 transmitter command

```powershell
python scripts/pc1_transmitter_test.py --payload "SIH_PC1_PC2_TEST"
python scripts/pc1_transmitter_test.py --payload "SIH_PC1_PC2_TEST" --play
```

### Physical PC1 command

```powershell
powershell -ExecutionPolicy Bypass -File scripts/pc1_physical_transmitter.ps1 -Payload "SIH_PC1_PC2_TEST"
```

---

## 2. PC2 Setup

PC2 is the **detector** — captures raw mic audio, runs DSPPipeline, posts ThreatEvent.

### Python environment (same as PC1)

```powershell
cd g:\Tryhackme\SIH-2026-backend
conda activate sih
pip install -r requirements.txt
```

### Backend startup (required on PC2 for POST /api/analyze)

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### PC2 receiver — microphone mode

```powershell
python scripts/pc2_receiver_test.py --mode mic --duration 15
python scripts/pc2_receiver_test.py --mode mic --list-devices
python scripts/pc2_receiver_test.py --mode mic --device 1 --duration 20
```

### PC2 receiver — file replay mode

```powershell
python scripts/pc2_receiver_test.py --mode file --wav generated_payloads/abc123.wav
```

### Physical PC2 command (start BEFORE PC1 plays)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/pc2_physical_receiver.ps1 -Duration 20
```

---

## 3. File Replay E2E (single machine)

Automated test without physical speaker/mic:

```powershell
python scripts/run_file_replay_e2e.py
python scripts/run_file_replay_e2e.py --wav samples/backend2/sample.wav
```

Orchestrated PowerShell runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pc1_pc2_test.ps1
```

---

## 4. Automated pytest

```powershell
python -m pytest tests/test_pc1_pc2_pipeline.py -v
python -m pytest tests/ -v
```

---

## 5. Expected Logs

### PC1 transmitter

```
========================================
PC1 ACOUSTIC TRANSMITTER
========================================
Payload        : SIH_PC1_PC2_TEST
Encoding       : PASS
BFSK Modulation: PASS
WAV Generation : PASS
Speaker        : PASS/NOT_AVAILABLE
WAV Path       : ...
========================================
```

### PC2 receiver (mic)

```
========================================
PC2 ACOUSTIC RECEIVER
========================================
Mode           : MICROPHONE
Device         : [0] ...
Capture        : RUNNING
========================================
[PC2] chunk=1 samples=2048
...
========================================
AI/DSP ANALYSIS
========================================
Detected            : True
...
========================================
BACKEND RESULT
========================================
POST /api/analyze : PASS
...
========================================
```

### File replay E2E

```
========================================
FILE REPLAY E2E TEST
========================================
Simulator encoding       : PASS
...
FINAL RESULT: PASS
========================================
```

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| Backend not reachable | Start uvicorn on port 8000 |
| `No module named uvicorn` | `pip install -r requirements.txt` |
| Speaker FAIL | Check audio output device; try without `--play` |
| Mic capture fails | `python scripts/pc2_receiver_test.py --mode mic --list-devices` |
| AI detected=False on physical test | PC1 must play during PC2 capture window; reduce distance |
| POST /api/analyze FAIL | Backend must be running before PC2 receiver |
| DSP not detecting | Use freq_1=20500 Hz (not 21000) |

---

## 7. Terminology

### What "encoded" means

```
TEXT → BITS (ASCII) → PREAMBLE → BFSK tones → float32 signal → WAV
```

Location: `payload_service.encode_text_to_signal()`

### What "decoded" means (simulator only)

```
WAV/signal → preamble search → Goertzel bit classification → BITS → TEXT
```

Location: `payload_service.decode_signal_to_text()`

Used for **simulator self-test only**. Not the security detection path.

### What AI detection means

```
RAW MICROPHONE AUDIO → preprocess → FFT/features → pattern/SNR → ThreatEvent
```

Location: `DSPPipeline.process()` + `to_threat_event()`

**Does NOT recover secret payload text.**

---

## 8. Simulator decode vs AI detection

| | Simulator decode | AI detection |
|--|------------------|--------------|
| Input | Known clean signal | Raw mic/noisy audio |
| Output | Recovered **text** | **ThreatEvent** (metadata) |
| Function | `decode_signal_to_text()` | `DSPPipeline.to_threat_event()` |
| Purpose | Codec validation | Security alert |
| Used in PC2 test? | **No** | **Yes** |

---

## 9. Known limitations

1. **Physical 2-PC test** cannot be automated from one machine — run scripts independently.
2. **In-memory threat store** resets when backend restarts.
3. **No checksum** on simulator payload bits.
4. **AI does not decode secret text** — only threat metadata.
5. **`/stream/forensics` WebSocket** uses simulator decode, not DSPPipeline — use `pc2_receiver_test.py` for AI path.
6. **Ultrasonic hardware** varies — speaker/mic quality affects physical test success.

---

## 10. Test result categories

Report each separately:

| # | Test | Command |
|---|------|---------|
| 1 | Simulator software | `python scripts/pc1_transmitter_test.py` |
| 2 | Encode/decode round-trip | `pytest tests/test_backend2_simulator.py` |
| 3 | File replay AI | `python scripts/run_file_replay_e2e.py` |
| 4 | API integration | `pytest tests/test_pc1_pc2_pipeline.py` |
| 5 | Physical PC1→PC2 | `pc2_physical_receiver.ps1` + `pc1_physical_transmitter.ps1` |

Do not claim physical test PASS unless speaker and microphone were actually used.
