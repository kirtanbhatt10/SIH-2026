# Your Role Guide — Backend 2 (Signal/Backend Engineer)

This is your personal playbook. Everything below has working, tested code already built for you in the `backend2/` folder — treat it as your starting point, not a spec to build from scratch.

## Your ownership, restated precisely

| You own | You do NOT own (but you feed data to them) |
|---|---|
| Audio streaming interface (capture + chunk + transport) | FFT/STFT/spectral features → **AI 1** |
| Attack simulator backend (encode + "transmit") | Threat classification/scoring → **AI 2** |
| Payload encode/decode pipeline | Persistent event storage, main API, deployment → **Backend 1** |
| Data formats (the message contracts) | Dashboard UI, simulator UI → **Frontend 1 & 2** |
| Performance optimization | PPT/docs → **PPT Lead** |

Your mental model: **you are the plumbing.** You get raw sound into the pipeline fast and cleanly, you get encoded/decoded payloads right, and you make sure nothing you build becomes a bottleneck for AI1/AI2 downstream. You are not responsible for deciding "is this an attack" — that's AI2's job. You just need to deliver them clean, well-formed, on-time data.

---

## What's already built and tested for you

All four files are in `backend2/` and have been run successfully:

| File | What it does | Verified result |
|---|---|---|
| `data_schemas.py` | The message contracts — **read this first and share it with the whole team on Day 0** | N/A (schema definitions) |
| `payload_codec.py` | Encode text → ultrasonic tone, AND decode tone → text back | Round-trip test passed: `SIH2026-DEMO` → tone → `SIH2026-DEMO`, 92.3% confidence |
| `audio_streamer.py` | Chunks audio (mic or file-replay) into `AudioChunkMessage`, broadcasts over WebSocket | Streamed 13 chunks from `payload.wav`, base64 round-trip verified |
| `attack_simulator_api.py` | FastAPI service: generate payload → transmit → status/list | Full API test passed: generate (200), transmit (200), list (200) |

Run them yourself to confirm:
```bash
cd backend2
pip install -r requirements.txt

python payload_codec.py                        # round-trip encode/decode self-test
python audio_streamer.py --source file --file ../payload.wav   # chunking self-test
uvicorn attack_simulator_api:app --reload --port 8002           # then hit /docs in browser
```

---

## The four things you're actually building, explained

### 1. Payload encoding/decoding pipeline (`payload_codec.py`)
- **Encode** (already had this from the prototype): text → BFSK ultrasonic waveform, `FREQ_0=18500Hz` for bit 0, `FREQ_1=21000Hz` for bit 1.
- **Decode** (new — this is your main technical-depth contribution): uses a **Goertzel algorithm** to test only the two known frequencies per bit-window, instead of a full FFT — much cheaper computationally, and this is a legitimate "we optimized for the specific structure of our problem" talking point for judges.
- This decode function is what powers the "forensic recovery" feature on the dashboard — instead of just saying "attack detected," you can show the SOC analyst *what the attacker was likely sending*, which is a strong differentiator in your demo.

**Why Goertzel over a generic FFT here:** Alternative = reuse AI1's full FFT/STFT pipeline for this too. Your choice is superior for this specific task because you only need the energy at two known frequencies, not the full spectrum — Goertzel is O(N) per frequency vs. paying for an entire FFT's worth of bins you'd throw away anyway.

### 2. Audio streaming interface (`audio_streamer.py`)
- Two pluggable sources: `LiveMicSource` (real mic via `sounddevice`) and `FileReplaySource` (replays a WAV as if it were live — **this is your team's safety net** if a laptop's mic driver acts up mid-hackathon; everyone can keep developing/demoing against a file).
- Broadcasts fixed-size, fixed-overlap chunks (`window_sec=1.0`, `hop_sec=0.5` by default) as `AudioChunkMessage` over a WebSocket at `/stream/audio`.
- **Both AI1's DSP service and Frontend1's live waveform view subscribe to this same WebSocket** — you only stream the audio once, they each consume it independently. Don't let either of them build their own mic-capture code; that causes device-lock conflicts (only one process can usually hold the mic at a time).

**Why WebSocket broadcast over point-to-point calls:** Alternative = AI1 polls you via REST for the latest chunk. Your choice is superior because polling adds latency and wastes cycles when nothing new has arrived; a push-based broadcast delivers each chunk to every subscriber the instant it's captured, which matters for your ~1-2 second detection latency target.

### 3. Attack simulator backend (`attack_simulator_api.py`)
- `/simulate/generate` — Frontend2 calls this when the demo operator types a message.
- `/simulate/transmit` — has two modes:
  - `mode="audio"` — actually plays the tone through the speaker (the real, theatrical demo).
  - `mode="virtual"` — injects the signal directly into the pipeline without needing a physical speaker/mic pair at all.
- **Always build and test `virtual` mode even if `audio` mode is your headline demo.** Venue audio is the single most common hackathon demo failure — echoey halls, cheap laptop mics with poor >18kHz response, background noise from other teams' booths. `virtual` mode is your insurance policy.

**Why offer both modes instead of committing to only the "real" acoustic demo:** Alternative = only support real speaker/mic transmission, which is more impressive when it works. Your choice is superior because a working "quieter" fallback demo beats a broken "impressive" one every time — judges remember failures, not ambition.

### 4. Data formats (`data_schemas.py`)
- This file **is** the integration contract. Every teammate should import from it rather than inventing their own JSON shapes.
- Get AI1's sign-off specifically on `StreamConfig` (sample rate, window/hop size, band range) **before either of you writes real code** — if your chunk size doesn't match what their FFT expects, you'll get silent garbage results that are painful to debug under time pressure.

### 5. Performance optimization (ongoing responsibility, not a single file)
Concrete things to actually do, in priority order:
1. **Profile the chunking loop first** — `python -c "import cProfile; cProfile.run('...')"` on `audio_streamer.py` before optimizing blindly.
2. **Keep chunk size matched to your latency budget.** `window_sec=1.0` gives AI1 a full second of signal (good for accuracy) but adds a full second of latency; if the demo needs faster visual response, coordinate with AI1 on whether a shorter window with more overlap still gives their classifier enough signal.
3. **Base64 overhead is ~33%** — fine for a hackathon over localhost/LAN, but if bandwidth ever becomes an issue (e.g. WAN judging), swap to raw binary WebSocket frames instead of JSON+base64.
4. **Goertzel algorithm in `payload_codec.py`** is already the optimized choice over full FFT for the two-frequency decode case (see above) — don't let anyone "simplify" this back to a full FFT call, it's a legitimate design decision worth defending in Q&A.
5. **Don't block the event loop** — `LiveMicSource` uses `sounddevice`'s callback + `asyncio.Queue` specifically so mic capture never blocks the async broadcast loop. Keep this pattern if you extend it.

---

## Your integration points with each teammate

| Teammate | What you give them | What you need from them |
|---|---|---|
| **AI 1 (DSP Lead)** | `AudioChunkMessage` stream via WebSocket, `StreamConfig` agreed up front | Confirmation their FFT window size matches your `window_sec`/`hop_sec` |
| **AI 2 (Detection Lead)** | Nothing direct — you feed AI1, AI1 feeds AI2 | Nothing direct, but ask them if they want your decoded payload text (`DecodedPayloadEvent`) attached to their classification events for richer alerts |
| **Backend 1 (Integration Lead)** | `DecodedPayloadEvent` for their event store/API, and your simulator API base URL for their gateway/reverse-proxy setup | Their event schema for alerts, so your decoded-payload data can be attached correctly |
| **Frontend 1 (Dashboard)** | Live audio stream (for waveform viz) via the same WebSocket AI1 uses | Nothing — they're a pure consumer of your stream |
| **Frontend 2 (Simulator UI)** | The three REST endpoints in `attack_simulator_api.py` — give them the OpenAPI docs at `/docs` once you deploy it | Their expected request/response shape confirmed against `GeneratePayloadRequest`/`TransmitRequest` |

---

## Your day-by-day checklist

**Day 0 / Setup:**
- [ ] Read and share `data_schemas.py` with the whole team.
- [ ] Confirm `StreamConfig` values with AI1 before either of you codes further.
- [ ] Run all three self-tests (`payload_codec.py`, `audio_streamer.py`, `attack_simulator_api.py`) locally, confirm all pass.

**Build phase:**
- [ ] Get `LiveMicSource` working on real hardware (not just `FileReplaySource`) — test on the actual laptop that will run the demo.
- [ ] Wire `/simulate/transmit` `mode="virtual"` to actually push into the same `AudioStreamer` instance AI1/Frontend1 are subscribed to (the code has a comment marking exactly where this hook goes).
- [ ] Give Frontend2 the running `/docs` Swagger UI early so they can build against it without waiting on you.
- [ ] Stress-test: generate and transmit 20+ payloads back-to-back, confirm no memory leak / slowdown in `_payloads` dict (consider adding simple cleanup/expiry if time allows).

**Pre-demo:**
- [ ] Confirm both `audio` and `virtual` transmit modes work on the actual demo laptop/speaker.
- [ ] Time the full loop: generate → transmit → AI1 features → AI2 classification → Backend1 event → Frontend alert. Know your real end-to-end latency number — judges will ask.
- [ ] Have `virtual` mode as your default demo path unless the room is confirmed quiet and the mic is confirmed good.

---

## Likely Judge Q&A for your part specifically

**Q: Why not just use a full FFT for the decode step too, since you already need FFT elsewhere?**
A: The decode step only needs to test two known frequencies per bit window, not resolve the whole spectrum — Goertzel gives us that answer in O(N) per frequency instead of paying for a full FFT's frequency resolution we'd discard anyway. It's a targeted optimization for a targeted sub-problem.

**Q: What happens if the venue's audio setup fails during your live demo?**
A: We built a "virtual transmission" mode specifically for this — it injects the encoded signal directly into the same pipeline the real audio would hit, so we can still show the full detection flow end-to-end without depending on unpredictable venue speakers/mics.

**Q: How do you keep the audio stream from becoming a bottleneck as more consumers (dashboard, DSP service) subscribe?**
A: We broadcast once over a WebSocket to all subscribers rather than having each consumer poll or open its own mic stream — this also avoids the classic problem of multiple processes fighting to lock the same microphone device.
