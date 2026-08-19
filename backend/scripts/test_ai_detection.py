"""Run Backend 2 sample.wav through the AI DSP pipeline (diagnostic)."""

import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "ai"))

from dsp import DSPPipeline  # noqa: E402
from backend.services.payload_service import encode_text_to_signal  # noqa: E402

CHUNK = 2048


def run_audio(audio: np.ndarray, sample_rate: int = 48000) -> dict:
    pipeline = DSPPipeline(sample_rate=sample_rate)
    for i in range(0, len(audio) - CHUNK, CHUNK):
        pipeline.process(audio[i : i + CHUNK])
    return pipeline.to_threat_event()


def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    audio = data.astype(np.float64)
    if data.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit-duration", type=float, default=None, help="Test alternate symbol duration")
    args = parser.parse_args()

    pipeline = DSPPipeline()

    if args.bit_duration:
        print(f"=== Encoded payload @ bit_duration={args.bit_duration} ===")
        sig = encode_text_to_signal("SIH 2026 BACKEND 2", bit_duration=args.bit_duration)
        print(json.dumps(run_audio(sig), indent=2))
        return

    print("=== Backend 2 sample ===")
    audio, sr = load_wav(os.path.join(_REPO_ROOT, "samples", "backend2", "sample.wav"))
    event = run_audio(audio, sr)
    print(json.dumps(event, indent=2))

    print("\n=== AI FSK 19000/20500 (training defaults) ===")
    sig = pipeline.generate_attack_signal(
        "fsk", duration_sec=3.0, freq_mark=19000, freq_space=20500, baud_rate=20
    )
    print(json.dumps(run_audio(sig), indent=2))

    print("\n=== Bit duration sweep on Backend 2 encoder ===")
    for bd in (0.05, 0.04, 0.0427):
        sig = encode_text_to_signal("SIH 2026 BACKEND 2", bit_duration=bd, pad_silence_sec=0.5)
        ev = run_audio(sig)
        print(
            f"  {bd}s: detected={ev['detected']} pattern={ev['pattern']} "
            f"carriers={ev['carrier_freqs']} score={ev['suspicion_score']}"
        )


if __name__ == "__main__":
    main()
