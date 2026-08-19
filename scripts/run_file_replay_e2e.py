#!/usr/bin/env python3
"""
File replay end-to-end test: Simulator WAV → DSPPipeline → POST /api/analyze → GET threats.

Example:
    python scripts/run_file_replay_e2e.py
    python scripts/run_file_replay_e2e.py --payload "SIH_PC1_PC2_TEST"
    python scripts/run_file_replay_e2e.py --wav samples/backend2/sample.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.core.config import BIT_DURATION, FREQ_0, FREQ_1, SAMPLE_RATE, STORAGE_DIR
from backend.services.payload_service import encode_text_to_signal, save_signal_to_wav
from scripts.pipeline_common import (
    BACKEND_API_URL,
    check_backend_health,
    load_wav_mono_float,
    print_ai_analysis,
    print_backend_result,
    run_dsp_pipeline,
    setup_paths,
    submit_and_validate_backend,
)

setup_paths()


def _status(ok: bool, detail: str = "") -> str:
    return "PASS" if ok else f"FAIL{(' (' + detail) if detail else ''}{(')' if detail else '')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="File replay E2E test")
    parser.add_argument("--payload", default="SIH_PC1_PC2_TEST", help="Generate WAV from this text")
    parser.add_argument("--wav", help="Use existing WAV instead of generating")
    parser.add_argument("--backend-url", default=BACKEND_API_URL)
    args = parser.parse_args()

    results: dict[str, str] = {}

    wav_path = args.wav
    if wav_path is None:
        try:
            signal = encode_text_to_signal(
                args.payload,
                freq_0=FREQ_0,
                freq_1=FREQ_1,
                bit_duration=BIT_DURATION,
                sample_rate=SAMPLE_RATE,
            )
            results["simulator_encoding"] = "PASS"
            results["bfsk_modulation"] = "PASS"
            os.makedirs(STORAGE_DIR, exist_ok=True)
            payload_id = str(uuid.uuid4())[:8]
            wav_path = os.path.join(STORAGE_DIR, f"{payload_id}.wav")
            save_signal_to_wav(wav_path, signal, SAMPLE_RATE)
            results["wav_generation"] = "PASS" if os.path.isfile(wav_path) else "FAIL"
        except Exception as e:
            results["simulator_encoding"] = _status(False, str(e))
            results["bfsk_modulation"] = "FAIL"
            results["wav_generation"] = "FAIL"
            wav_path = None
    else:
        results["simulator_encoding"] = "SKIP (using --wav)"
        results["bfsk_modulation"] = "SKIP (using --wav)"
        results["wav_generation"] = "SKIP (using --wav)"

    if not wav_path or not os.path.isfile(wav_path):
        print("========================================")
        print("FILE REPLAY E2E TEST")
        print("========================================")
        for k, v in results.items():
            print(f"{k.replace('_', ' ').title():22}: {v}")
        print("WAV loading            : FAIL (no file)")
        print("========================================")
        print("FINAL RESULT: FAIL")
        print("========================================")
        return 1

    try:
        audio, sr = load_wav_mono_float(wav_path)
        results["wav_loading"] = "PASS"
        nonzero = bool(abs(audio).max() > 1e-6)
        results["raw_audio_conversion"] = "PASS" if nonzero else "FAIL (silent)"
    except Exception as e:
        results["wav_loading"] = _status(False, str(e))
        results["raw_audio_conversion"] = "FAIL"
        audio, sr = None, SAMPLE_RATE

    event = None
    pipeline = None
    if audio is not None:
        try:
            pipeline, event = run_dsp_pipeline(audio, sr, verbose=False)
            results["dsp_processing"] = "PASS"
            results["threat_event_generation"] = "PASS" if event.get("schema_version") else "FAIL"
        except Exception as e:
            results["dsp_processing"] = _status(False, str(e))
            results["threat_event_generation"] = "FAIL"

    backend_ok = False
    if event is not None:
        if not check_backend_health(args.backend_url):
            results["post_api_analyze"] = "FAIL (backend not running)"
            results["backend_storage"] = "FAIL"
            results["threat_retrieval"] = "FAIL"
        else:
            br = submit_and_validate_backend(event, base_url=args.backend_url)
            results["post_api_analyze"] = br.get("post_analyze", "FAIL")
            results["backend_storage"] = br.get("threat_stored", "FAIL")
            threat_ok = br.get("current_threat") == "PASS" and br.get("threat_history") == "PASS"
            results["threat_retrieval"] = "PASS" if threat_ok else "FAIL"

    print("========================================")
    print("FILE REPLAY E2E TEST")
    print("========================================")
    print(f"WAV path               : {os.path.abspath(wav_path)}")
    order = [
        ("simulator_encoding", "Simulator encoding"),
        ("bfsk_modulation", "BFSK modulation"),
        ("wav_generation", "WAV generation"),
        ("wav_loading", "WAV loading"),
        ("raw_audio_conversion", "Raw audio conversion"),
        ("dsp_processing", "DSP processing"),
        ("threat_event_generation", "ThreatEvent generation"),
        ("post_api_analyze", "POST /api/analyze"),
        ("backend_storage", "Backend storage"),
        ("threat_retrieval", "Threat retrieval"),
    ]
    for key, label in order:
        if key in results:
            print(f"{label:22}: {results[key]}")

    if event is not None:
        print()
        print_ai_analysis(event, pipeline)

    required_keys = [
        "wav_loading",
        "raw_audio_conversion",
        "dsp_processing",
        "threat_event_generation",
        "post_api_analyze",
        "backend_storage",
        "threat_retrieval",
    ]
    if wav_path and args.wav is None:
        required_keys = [
            "simulator_encoding",
            "bfsk_modulation",
            "wav_generation",
        ] + required_keys

    all_pass = all(results.get(k) == "PASS" for k in required_keys if k in results)
    print("========================================")
    print(f"FINAL RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("========================================")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
