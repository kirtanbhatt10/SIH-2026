#!/usr/bin/env python3
"""
PC2 Acoustic Receiver — raw capture + DSPPipeline + Backend integration.

Microphone mode (does NOT use decode_signal_to_text for threat detection):
    python scripts/pc2_receiver_test.py --mode mic --duration 15

File replay mode:
    python scripts/pc2_receiver_test.py --mode file --wav generated_payloads/abc123.wav
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import FORENSIC_CAPTURE_DIR, SAMPLE_RATE
from backend.services.payload_service import save_signal_to_wav
from scripts.pipeline_common import (
    BACKEND_API_URL,
    capture_microphone,
    check_backend_health,
    list_input_devices,
    load_wav_mono_float,
    print_ai_analysis,
    print_backend_result,
    run_dsp_pipeline,
    setup_paths,
    submit_and_validate_backend,
)

setup_paths()


def _resolve_device(device_id: int | None, list_devices: bool) -> int | None:
    devices = list_input_devices()
    if list_devices:
        print("Available input devices:")
        for d in devices:
            if "error" in d:
                print(f"  ERROR: {d['error']}")
                continue
            mark = " (default)" if d.get("is_default") else ""
            print(f"  [{d['id']}] {d['name']}{mark}")
        if not devices or "error" in devices[0]:
            return None
    if device_id is not None:
        return device_id
    for d in devices:
        if d.get("is_default"):
            return d["id"]
    return devices[0]["id"] if devices and "error" not in devices[0] else None


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 acoustic receiver test")
    parser.add_argument("--mode", choices=["mic", "file"], required=True)
    parser.add_argument("--wav", help="WAV path for file mode")
    parser.add_argument("--duration", type=float, default=15.0, help="Mic capture duration (seconds)")
    parser.add_argument("--device", type=int, default=None, help="Input device ID")
    parser.add_argument("--list-devices", action="store_true", help="List microphones and exit")
    parser.add_argument(
        "--save-capture",
        default=None,
        help="Save captured/replayed audio to WAV (mic mode default: forensic_captures/)",
    )
    parser.add_argument("--backend-url", default=BACKEND_API_URL, help="Backend base URL")
    parser.add_argument("--no-submit", action="store_true", help="Skip POST /api/analyze")
    args = parser.parse_args()

    if args.list_devices:
        _resolve_device(None, True)
        return 0

    audio: np.ndarray | None = None
    sample_rate = SAMPLE_RATE
    device_name = "file"
    channels = 1

    if args.mode == "file":
        if not args.wav:
            print("ERROR: --wav is required for file mode")
            return 1
        if not os.path.isfile(args.wav):
            print(f"ERROR: WAV not found: {args.wav}")
            return 1
        audio, sample_rate = load_wav_mono_float(args.wav)
        print("========================================")
        print("PC2 ACOUSTIC RECEIVER")
        print("========================================")
        print("Mode           : FILE REPLAY")
        print(f"WAV            : {os.path.abspath(args.wav)}")
        print(f"Sample Rate    : {sample_rate}")
        print(f"Channels       : 1 (mono)")
        print(f"Samples        : {len(audio)}")
        print("Capture        : LOADED")
        print("========================================")
    else:
        device_id = _resolve_device(args.device, args.list_devices)
        devices = list_input_devices()
        if device_id is None:
            print("ERROR: No microphone available")
            return 1
        for d in devices:
            if d.get("id") == device_id:
                device_name = d.get("name", str(device_id))
                channels = d.get("max_input_channels", 1)

        print("========================================")
        print("PC2 ACOUSTIC RECEIVER")
        print("========================================")
        print("Mode           : MICROPHONE")
        print(f"Device         : [{device_id}] {device_name}")
        print(f"Sample Rate    : {sample_rate}")
        print(f"Channels       : {channels}")
        print(f"Duration (s)   : {args.duration}")
        print("Capture        : RUNNING")
        print("========================================")

        start_ts = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"[PC2] capture_start={start_ts}")

        try:
            audio = capture_microphone(args.duration, sample_rate=sample_rate, device_id=device_id)
        except Exception as e:
            print(f"ERROR: Microphone capture failed: {e}")
            return 1

        end_ts = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"[PC2] capture_end={end_ts}")
        print(f"[PC2] captured_samples={len(audio)}")

        save_path = args.save_capture
        if save_path is None:
            os.makedirs(FORENSIC_CAPTURE_DIR, exist_ok=True)
            save_path = os.path.join(
                FORENSIC_CAPTURE_DIR,
                f"pc2_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            )
        save_signal_to_wav(save_path, audio.astype(np.float32), sample_rate)
        print(f"[PC2] saved_capture={os.path.abspath(save_path)}")

    if audio is None or len(audio) < 2048:
        print("ERROR: Insufficient audio samples for DSP processing")
        return 1

    pipeline, event = run_dsp_pipeline(audio, sample_rate, verbose=True)
    print_ai_analysis(event, pipeline)

    if args.no_submit:
        print("[PC2] Skipping backend submission (--no-submit)")
        return 0

    if not check_backend_health(args.backend_url):
        print(f"ERROR: Backend not reachable at {args.backend_url}")
        print("Start: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        return 1

    backend_results = submit_and_validate_backend(event, base_url=args.backend_url)
    print_backend_result(backend_results)

    all_pass = all(
        backend_results.get(k) == "PASS"
        for k in ("post_analyze", "threat_stored", "current_threat", "threat_history")
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
