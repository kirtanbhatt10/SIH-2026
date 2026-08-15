"""
diagnose_capture.py — Find out why the capture stream dies mid-recording
=========================================================================
AI 1 (DSP / Signal Processing Lead)

WHY
---
Analysis of the 15 Aug 2026 test set showed the microphone stream stops
delivering samples roughly 1-2 seconds into every recording:

    rx_1000.wav      last activity @ 1.6s of 10s
    rx_1000_v2.wav   last activity @ 2.0s of 10s
    rx_5000_v2.wav   last activity @ 1.5s of 10s
    rx_18000.wav     no activity at all

Ambient room noise disappears too, which a real microphone never does. So this
is a stream failure, not a quiet room and not a frequency limitation.

`sd.rec()` pre-allocates its buffer and returns normally even when the stream
stalls, so the failure is silent. This script uses `sd.InputStream` with an
explicit status callback, which surfaces overflow/underflow errors that
`sd.rec()` swallows.

USAGE
-----
    python scripts/diagnose_capture.py                 # test default device
    python scripts/diagnose_capture.py --device 3      # test a specific device
    python scripts/diagnose_capture.py --all           # sweep every input device
    python scripts/diagnose_capture.py --rate 44100    # try the native rate

Make noise (talk, clap continuously) for the whole test window.
"""

import argparse
import sys

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed:  pip install sounddevice")
    sys.exit(1)

LSB = 1.0 / 32768.0
ALIVE_LSB = 20.0          # a block above this contains real audio


def test_device(device, samplerate, duration, blocksize=1024):
    """
    Record with InputStream and report per-second liveness plus driver errors.
    Returns True if audio was present through to the end.
    """
    chunks, statuses = [], []

    def callback(indata, frames, time_info, status):
        if status:
            statuses.append(str(status))
        chunks.append(indata[:, 0].copy())

    try:
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
        name = info["name"]
        native = info["default_samplerate"]
    except Exception as e:
        print(f"  cannot query device: {e}")
        return False

    print(f"\n  Device : {name}")
    print(f"  Native rate: {native:.0f} Hz   Requested: {samplerate} Hz", end="")
    if abs(native - samplerate) > 1:
        print("   <-- MISMATCH (suspect)")
    else:
        print()

    try:
        with sd.InputStream(device=device, channels=1, samplerate=samplerate,
                            blocksize=blocksize, dtype="float32", callback=callback):
            print(f"  Recording {duration}s — MAKE NOISE NOW (talk/clap continuously)")
            sd.sleep(int(duration * 1000))
    except Exception as e:
        print(f"  STREAM ERROR: {type(e).__name__}: {e}")
        return False

    if not chunks:
        print("  FAIL: callback never fired — no data at all.")
        return False

    audio = np.concatenate(chunks)
    got = len(audio) / samplerate
    print(f"  Captured {len(audio)} samples ({got:.2f}s of {duration}s requested)")

    if statuses:
        print(f"  DRIVER STATUS FLAGS ({len(statuses)} events): {set(statuses)}")
        print("    ^ overflow/underflow here explains the dropout")

    # Per-second liveness
    print("\n  Per-second peak (LSB units, >20 = audio present):")
    alive = []
    for s in range(int(np.ceil(got))):
        seg = audio[s * samplerate:(s + 1) * samplerate]
        if not len(seg):
            break
        pk = float(np.max(np.abs(seg))) / LSB
        alive.append(pk > ALIVE_LSB)
        bar = "#" * min(40, int(np.log10(max(pk, 1)) * 10))
        print(f"    {s:3d}s  {pk:9.0f}  {bar}")

    if not any(alive):
        print("\n  RESULT: DEAD — no audio captured at any point.")
        return False
    if not all(alive):
        first_dead = alive.index(False) if False in alive else None
        print(f"\n  RESULT: STREAM DIED at ~{first_dead}s (this is the bug).")
        return False

    print("\n  RESULT: OK — audio present for the whole duration.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--blocksize", type=int, default=1024)
    ap.add_argument("--all", action="store_true", help="sweep every input device")
    args = ap.parse_args()

    print("=" * 70)
    print("  CAPTURE STREAM DIAGNOSTIC")
    print("=" * 70)

    hostapis = sd.query_hostapis()
    devices = sd.query_devices()

    if args.all:
        inputs = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        print(f"\nTesting {len(inputs)} input devices at {args.rate} Hz, "
              f"{args.duration}s each.\nMake continuous noise throughout.\n")
        good = []
        for idx in inputs:
            api = hostapis[devices[idx]["hostapi"]]["name"]
            print("-" * 70)
            print(f"[{idx}] {devices[idx]['name']}  (HostAPI: {api})")
            if test_device(idx, args.rate, args.duration, args.blocksize):
                good.append((idx, devices[idx]["name"], api))
        print("\n" + "=" * 70)
        if good:
            print("  DEVICES THAT WORKED:")
            for idx, nm, api in good:
                print(f"    [{idx}] {nm}  ({api})")
            print(f"\n  Use one of these in record_wav.py.")
        else:
            print("  NO DEVICE PRODUCED CONTINUOUS AUDIO.")
            print("  Next: try --rate 44100, and check OS mic privacy settings.")
        return 0 if good else 1

    ok = test_device(args.device, args.rate, args.duration, args.blocksize)

    if not ok:
        print("\n  Try, in order:")
        print("    1. python scripts/diagnose_capture.py --all")
        print("    2. python scripts/diagnose_capture.py --rate 44100")
        print("       (if the device's native rate is 44100, forcing 48000 can stall it)")
        print("    3. python scripts/diagnose_capture.py --blocksize 4096")
        print("    4. Prefer a WASAPI device over MME/WDM-KS on Windows")
        print("    5. Close any app that may hold the mic (Teams, Zoom, browser)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
