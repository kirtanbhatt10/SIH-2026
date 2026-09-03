#!/usr/bin/env python3
"""Direct pipeline isolation test — generated WAV vs sample.wav."""

import asyncio
import os
import sys
import wave

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.core.config import SAMPLE_RATE, STREAM_CONFIG
from backend.services.monitoring_pipeline import MonitoringPipeline
from backend.services.payload_service import encode_text_to_signal, save_signal_to_wav
from backend.services.threat_service import get_threats, add_threat  # noqa: F401 — side effect via pipeline


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return sig, sr


def analyze(sig: np.ndarray, label: str) -> dict:
    rms = float(np.sqrt(np.mean(sig * sig)))
    peak = float(np.max(np.abs(sig)))
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1 / SAMPLE_RATE)
    band = (freqs >= 17000) & (freqs <= 22000)
    peak_idx = np.argmax(fft[band]) if band.any() else 0
    dom = float(freqs[band][peak_idx]) if band.any() else 0.0
    return {
        "label": label,
        "samples": len(sig),
        "duration_sec": len(sig) / SAMPLE_RATE,
        "rms": rms,
        "peak": peak,
        "dominant_hz": dom,
    }


async def feed_pipeline(signal: np.ndarray, label: str) -> dict:
    pipe = MonitoringPipeline(sample_rate=SAMPLE_RATE)
    await pipe.start()
    win = int(SAMPLE_RATE * STREAM_CONFIG.window_sec)
    hop = int(SAMPLE_RATE * STREAM_CONFIG.hop_sec)
    for start in range(0, max(1, len(signal) - win), hop):
        chunk = signal[start : start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        await pipe.process_chunk(chunk.astype(np.float32))
    snap = pipe.last_dsp_snapshot
    ml = pipe._last_ml  # noqa: SLF001
    await pipe.stop()
    return {
        "label": label,
        "dsp_detected": snap.get("detected"),
        "dsp_pattern": snap.get("pattern"),
        "dsp_carriers": snap.get("carrier_freqs"),
        "ml_class": (ml or {}).get("predicted_class"),
        "ml_risk": (ml or {}).get("risk_level"),
        "threats_after": len(get_threats()),
    }


async def main():
    sample_path = os.path.join(_REPO, "samples", "backend2", "sample.wav")
    gen_sig = encode_text_to_signal("SIH2026", freq_0=18500, freq_1=20500)
    gen_path = os.path.join(_REPO, "generated_payloads", "_debug_gen.wav")
    save_signal_to_wav(gen_path, gen_sig, SAMPLE_RATE)

    sample_sig, _ = load_wav(sample_path)
    print("=== WAV ANALYSIS ===")
    print(analyze(sample_sig, "sample.wav"))
    print(analyze(gen_sig, "generated"))

    print("=== DIRECT PIPELINE ===")
    r1 = await feed_pipeline(sample_sig, "sample.wav")
    print(r1)
    r2 = await feed_pipeline(gen_sig, "generated")
    print(r2)


if __name__ == "__main__":
    asyncio.run(main())
