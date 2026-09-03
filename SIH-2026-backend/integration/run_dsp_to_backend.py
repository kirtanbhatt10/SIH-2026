"""
Run DSPPipeline on audio and POST the ThreatEvent to Backend 1.

Examples::

    # Synthetic FSK (no WAV file required)
    python -m integration.run_dsp_to_backend --synthetic fsk

    # Existing Backend 2 reference sample
    python -m integration.run_dsp_to_backend --wav samples/backend2/sample.wav

Environment::

    BACKEND_API_URL=http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_AI_ROOT = os.path.join(_REPO_ROOT, "ai")
if _AI_ROOT not in sys.path:
    sys.path.insert(0, _AI_ROOT)

from dsp import DSPPipeline  # noqa: E402
from integration.backend_client import BackendThreatClient  # noqa: E402

CHUNK = 2048


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _run_pipeline(audio: np.ndarray, sample_rate: int) -> dict:
    pipeline = DSPPipeline(sample_rate=sample_rate)
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pipeline.process(audio[i : i + CHUNK])
    return pipeline.to_threat_event()


def main() -> int:
    parser = argparse.ArgumentParser(description="DSP → Backend 1 integration runner")
    parser.add_argument("--wav", help="Path to mono WAV file")
    parser.add_argument(
        "--synthetic",
        choices=["fsk", "ook", "chirp", "tone"],
        help="Generate synthetic attack instead of loading WAV",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Synthetic signal length (s)")
    parser.add_argument("--backend-url", help="Override BACKEND_API_URL")
    args = parser.parse_args()

    if args.wav:
        audio, sr = _load_wav(args.wav)
    elif args.synthetic:
        pipeline = DSPPipeline()
        audio = pipeline.generate_attack_signal(args.synthetic, duration_sec=args.duration)
        sr = pipeline.sample_rate
    else:
        parser.error("Provide --wav or --synthetic")

    event = _run_pipeline(audio, sr)
    print("ThreatEvent from DSP:")
    print(json.dumps(event, indent=2))

    client = BackendThreatClient(base_url=args.backend_url)
    result = client.submit_threat_event(event)

    print(f"\nPOST {client.analyze_url} -> HTTP {result.status_code}")
    if result.success:
        print("Status: stored (backend acknowledged)")
        current, err = client.get_current_threat()
        if err:
            print(f"Warning: could not fetch current threat: {err}")
            return 1
        print("\nGET /api/threats/current:")
        print(json.dumps(current, indent=2))
        if current == event:
            print("\nVerification: current threat matches submitted payload exactly.")
        else:
            print("\nVerification FAILED: current threat differs from submitted payload.")
            return 1
        return 0

    print(f"Status: NOT stored — {result.error}")
    if result.response_body:
        print(json.dumps(result.response_body, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
