"""
Shared utilities for PC1/PC2 acoustic test scripts.

Uses existing integration layer (integration/backend_client.py) and
DSPPipeline (ai/dsp) — no duplicate encoder or BFSK implementation.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import httpx
import numpy as np
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from dsp import DSPPipeline  # noqa: E402
from integration.backend_client import BackendThreatClient  # noqa: E402
from integration.config import BACKEND_API_URL  # noqa: E402

CHUNK_SIZE = 2048


def setup_paths() -> str:
    """Return repository root (paths already inserted at import)."""
    return _REPO_ROOT


def load_wav_mono_float(path: str) -> tuple[np.ndarray, int]:
    """Load WAV as mono float64 in [-1, 1]."""
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    elif data.dtype == np.uint8:
        audio = (audio - 128.0) / 128.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def run_dsp_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = False,
) -> tuple[DSPPipeline, dict]:
    """
    Feed sequential chunks into DSPPipeline and return pipeline + ThreatEvent.
    """
    pipeline = DSPPipeline(sample_rate=sample_rate)
    chunk_num = 0
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            break
        pipeline.process(chunk.astype(np.float64))
        chunk_num += 1
        if verbose:
            print(f"[PC2] chunk={chunk_num} samples={len(chunk)}")
    event = pipeline.to_threat_event()
    return pipeline, event


def print_ai_analysis(event: dict, pipeline: Optional[DSPPipeline] = None) -> None:
    """Print AI/DSP analysis block from ThreatEvent."""
    verdict = pipeline.get_verdict() if pipeline is not None else {}
    ultrasonic = verdict.get("ultrasonic_activity_detected", event.get("detected"))
    carriers = event.get("carrier_freqs") or []
    carrier_str = ", ".join(f"{c:.1f}" for c in carriers) if carriers else "none"

    print("========================================")
    print("AI/DSP ANALYSIS")
    print("========================================")
    print(f"Ultrasonic activity : {ultrasonic}")
    print(f"Carrier frequency   : {carrier_str}")
    print(f"Pattern             : {event.get('pattern')}")
    print(f"SNR                 : {event.get('snr')}")
    print(f"Suspicion score     : {event.get('suspicion_score')}")
    print(f"Confidence          : {event.get('confidence')}")
    print(f"Detected            : {event.get('detected')}")
    print(f"Risk                : {event.get('risk')}")
    print(f"Chunks analyzed     : {event.get('chunks_analyzed')}")
    print("========================================")


def check_backend_health(base_url: str | None = None) -> bool:
    """Return True if /api/system-status returns HTTP 200."""
    url = (base_url or BACKEND_API_URL).rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{url}/api/system-status")
            return r.status_code == 200
    except httpx.RequestError:
        return False


def submit_and_validate_backend(
    event: dict,
    base_url: str | None = None,
) -> dict[str, Any]:
    """
    POST ThreatEvent, GET current + history. Returns pass/fail flags.
    Does not modify AI values.
    """
    client = BackendThreatClient(base_url=base_url)
    results: dict[str, Any] = {
        "post_analyze": "FAIL",
        "threat_stored": "FAIL",
        "current_threat": "FAIL",
        "threat_history": "FAIL",
        "submit_result": None,
        "current": None,
        "threats": None,
    }

    submit = client.submit_threat_event(event)
    results["submit_result"] = submit
    if submit.success:
        results["post_analyze"] = "PASS"
        results["threat_stored"] = "PASS"

    try:
        with httpx.Client(timeout=10.0) as http:
            cur_r = http.get(f"{client.base_url}/api/threats/current")
            if cur_r.status_code == 200:
                body = cur_r.json()
                current = body.get("current")
                results["current"] = current
                if current is not None and current == event:
                    results["current_threat"] = "PASS"

            hist_r = http.get(f"{client.base_url}/api/threats")
            if hist_r.status_code == 200:
                threats = hist_r.json().get("threats", [])
                results["threats"] = threats
                if any(t == event for t in threats):
                    results["threat_history"] = "PASS"
    except httpx.RequestError:
        pass

    return results


def print_backend_result(results: dict[str, Any]) -> None:
    print("========================================")
    print("BACKEND RESULT")
    print("========================================")
    print(f"POST /api/analyze : {results.get('post_analyze', 'FAIL')}")
    print(f"Threat stored     : {results.get('threat_stored', 'FAIL')}")
    print(f"Current threat    : {results.get('current_threat', 'FAIL')}")
    print(f"Threat history    : {results.get('threat_history', 'FAIL')}")
    print("========================================")


def list_input_devices() -> list[dict]:
    """List microphone input devices via sounddevice."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default_in = sd.default.device[0]
        out = []
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                out.append(
                    {
                        "id": idx,
                        "name": d.get("name"),
                        "max_input_channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate"),
                        "is_default": idx == default_in,
                    }
                )
        return out
    except Exception as e:
        return [{"error": str(e)}]


def capture_microphone(
    duration_sec: float,
    sample_rate: int = 48000,
    device_id: Optional[int] = None,
) -> np.ndarray:
    """Capture raw mono float32 audio from microphone."""
    import sounddevice as sd

    frames = int(duration_sec * sample_rate)
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_id,
    )
    sd.wait()
    return audio[:, 0].copy()
