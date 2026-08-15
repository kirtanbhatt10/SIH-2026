import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
import hashlib
import re
import sys

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read

from dsp.recording_quality import check_recording


SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "float32"
DEFAULT_DURATION_SEC = 5.0


def select_input_device():
    """
    Choose an input device, preferring one whose NATIVE sample rate matches
    SAMPLE_RATE.

    Why native rate matters (incident 15 Aug 2026)
    ----------------------------------------------
    The previous version matched on device name only and returned the first
    hit. On this laptop that selected:

        [2] Microphone Array (Senary Audio)  MME  native 44100 Hz

    while we request 48000 Hz. Windows accepted the mismatched request, then
    the stream stalled 1-2 seconds in and delivered digital silence for the
    remainder. Every physical test recording was destroyed this way, and
    record_wav.py reported [SUCCESS] regardless.

    Measured with scripts/diagnose_capture.py:

        [18] Microphone Array (Senary Audio capture)  WDM-KS  native 48000
             -> 8.02s captured, audio present throughout (floor ~163-207 LSB,
                claps 11209-17844 LSB)                              WORKS
        [22] Microphone (Senary Audio capture)        WDM-KS  native 48000
             -> PaErrorCode -9996 Invalid device                    FAILS
        [0]/[1]/[2] MME, [5]-[7] DirectSound          native 44100
             -> flat 1 LSB, digital silence                         FAILS

    Selection order therefore is:
      1. native rate == SAMPLE_RATE  (avoids the resampling stall)
      2. host API preference: WASAPI > WDM-KS > DirectSound > MME
      3. known-good hardware name (senary / realtek / array)
      4. not a virtual/webcam mic

    Always verify a new machine with:
        python scripts/diagnose_capture.py --all
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Higher is better. WASAPI is the most reliable modern path on Windows;
    # MME is the legacy one that silently resamples and stalls.
    API_RANK = {"Windows WASAPI": 3, "Windows WDM-KS": 2,
                "Windows DirectSound": 1, "MME": 0}

    candidates = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"]
        low = name.lower()
        if "iriun" in low or "virtual" in low or "sound mapper" in low:
            continue

        api = hostapis[dev["hostapi"]]["name"]
        native = float(dev["default_samplerate"])
        candidates.append({
            "idx": idx,
            "name": name,
            "api": api,
            "native": native,
            "rate_ok": abs(native - SAMPLE_RATE) < 1.0,
            "api_rank": API_RANK.get(api, 0),
            "known_hw": any(k in low for k in ("senary", "realtek", "array")),
        })

    if not candidates:
        raise RuntimeError(
            "No usable input devices found.\n"
            "Run: python scripts/enum_devices.py"
        )

    candidates.sort(key=lambda c: (c["rate_ok"], c["api_rank"], c["known_hw"]),
                    reverse=True)
    best = candidates[0]

    if not best["rate_ok"]:
        print(f"\n[WARNING] No input device runs natively at {SAMPLE_RATE} Hz.")
        print(f"          Best available: [{best['idx']}] {best['name']} "
              f"({best['api']}, native {best['native']:.0f} Hz)")
        print("          Rate conversion can stall the stream mid-recording.")
        print("          Verify with: python scripts/diagnose_capture.py --all")

    print(f"\n[Device Selection] Considered {len(candidates)} input device(s):")
    for c in candidates[:6]:
        mark = "->" if c["idx"] == best["idx"] else "  "
        flag = "" if c["rate_ok"] else f"  (native {c['native']:.0f} Hz - MISMATCH)"
        print(f"  {mark} [{c['idx']:2d}] {c['name']}  ({c['api']}){flag}")

    return best["idx"], best["name"]


def calculate_file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def infer_expected_hz(label: str):
    """Infer the transmitted tone from a label like 'rx_18000' so we can
    verify the tone actually arrived. Returns None for baselines."""
    if any(k in label.lower() for k in ("baseline", "silence", "clap", "sil", "noise")):
        return None
    m = re.search(r"(\d{3,5})", label)
    if m:
        v = int(m.group(1))
        if 100 <= v <= 24000:
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description="Physical test audio recorder")
    ap.add_argument("--force", action="store_true",
                    help="save even if the quality gate rejects the capture")
    ap.add_argument("--expected", type=float, default=None,
                    help="transmitted tone in Hz (default: inferred from label)")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index; overrides auto-selection. "
                         "Verified working on this laptop: 18 "
                         "(Senary Audio capture, WDM-KS, native 48 kHz)")
    ap.add_argument("--list-devices", action="store_true",
                    help="list input devices and exit")
    cli = ap.parse_args()
    force = cli.force

    print("=" * 60)
    print("      PHYSICAL TEST AUDIO RECORDER (PROD HARDWARE)")
    print("=" * 60)

    if cli.list_devices:
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                api = hostapis[d["hostapi"]]["name"]
                print(f"[{i:2d}] {d['name']:<45s} {api:<22s} "
                      f"native {d['default_samplerate']:.0f} Hz")
        return 0

    if cli.device is not None:
        device_idx = cli.device
        try:
            device_name = sd.query_devices(device_idx, "input")["name"]
        except Exception as e:
            print(f"[ERROR] Device {device_idx} is not a usable input: {e}")
            print("        Run with --list-devices to see valid indices.")
            return 1
        print(f"\n[Device] Forced by --device: [{device_idx}] {device_name}")
    else:
        device_idx, device_name = select_input_device()

    print(f"\n[Selected Mic Device #{device_idx}]: {device_name}")
    
    # Verify device settings
    try:
        sd.check_input_settings(device=device_idx, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
        print(f"[Device Check] {SAMPLE_RATE} Hz, {CHANNELS}-channel {DTYPE} supported OK.")
    except Exception as err:
        print(f"[WARNING] Input setting warning for device #{device_idx}: {err}")

    label = input("\nEnter unique recording label (e.g. test_silence, test_clap, rx_18000): ").strip()
    if not label:
        raise ValueError("Recording label cannot be empty.")
    
    label = label.replace(" ", "_")
    output_filename = f"{label}.wav"
    output_abs_path = os.path.abspath(output_filename)

    if os.path.exists(output_abs_path):
        ans = input(f"\n[WARNING] {output_filename} already exists. Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("Recording cancelled.")
            return

    duration_str = input(f"Enter duration in seconds [default={DEFAULT_DURATION_SEC}]: ").strip()
    duration = float(duration_str) if duration_str else DEFAULT_DURATION_SEC

    expected_hz = cli.expected if cli.expected else infer_expected_hz(label)
    if expected_hz:
        print(f"  Expecting a {expected_hz:.0f} Hz tone (from label). "
              f"Use --expected to override.")

    print("\nRecording Configuration:")
    print(f"  Output Path  : {output_abs_path}")
    print(f"  Sample Rate  : {SAMPLE_RATE} Hz")
    print(f"  Channels     : {CHANNELS}")
    print(f"  Duration     : {duration} seconds")
    print(f"  Input Device : #{device_idx} ({device_name})")

    input("\nPress ENTER to start recording...")

    print("\n*** RECORDING NOW ***")
    print("Speak/clap or play ultrasonic signal...")

    num_samples = int(duration * SAMPLE_RATE)
    audio = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=device_idx,
    )
    sd.wait()

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))

    # ── QUALITY GATE ──────────────────────────────────────────────────────
    # Do NOT write a dead capture to disk. On 15 Aug 2026 an entire physical
    # test sweep was silently saved at 1-2 LSB (no audio at all) and was very
    # nearly presented as measured hardware evidence. Fail loudly instead.
    mono = audio[:, 0] if audio.ndim > 1 else audio
    report = check_recording(mono, SAMPLE_RATE, expected_hz=expected_hz)

    print("\n[Quality Check]")
    print(f"  Peak           : {report['peak']:.6e}  ({report['peak_lsb']} LSB)")
    print(f"  RMS            : {report['rms']:.6e}  ({report['rms_dbfs']} dBFS)")
    print(f"  Unique values  : {report['unique_values']}")
    if report["tone"]:
        t = report["tone"]
        print(f"  Tone @ {t['expected_hz']:.0f} Hz : peak {t['measured_peak_hz']} Hz, "
              f"SNR {t['tone_snr_db']} dB, present={t['tone_present']}")

    for w in report["warnings"]:
        print(f"  [WARN] {w}")

    if not report["ok"]:
        print("\n" + "!" * 62)
        print("  RECORDING REJECTED — NOT SAVED")
        print("!" * 62)
        for p in report["problems"]:
            print(f"  - {p}")
        print("\n  Troubleshooting:")
        print("    1. python scripts/enum_devices.py   (is the right mic selected?)")
        print("    2. Check OS microphone permissions and the input level slider")
        print("    3. Clap during a test recording — a working mic gives >1000 LSB")
        print("    4. Confirm the transmitter is actually playing the tone")
        if force:
            print("\n  --force given: saving anyway. DO NOT use this file as evidence.")
        else:
            print("\n  Re-run once fixed. (Use --force to save regardless.)")
            return 1

    # Save to disk as 32-bit float WAV
    write(output_abs_path, SAMPLE_RATE, audio)

    # Compute SHA256
    file_sha256 = calculate_file_sha256(output_abs_path)

    # Verification: reload written file immediately
    read_sr, read_data = read(output_abs_path)
    assert read_sr == SAMPLE_RATE, f"Sample rate mismatch on reload! Expected {SAMPLE_RATE}, got {read_sr}"
    assert len(read_data) == num_samples, f"Sample length mismatch on reload! Expected {num_samples}, got {len(read_data)}"

    print("\n[SUCCESS] Recording Completed & Verified:")
    print(f"  Saved File     : {output_abs_path}")
    print(f"  Sample Count   : {len(read_data)}")
    print(f"  Peak Amplitude : {peak:.6f}")
    print(f"  RMS Amplitude  : {rms:.6f}")
    print(f"  SHA256 Hash    : {file_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)