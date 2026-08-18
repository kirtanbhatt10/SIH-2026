"""
record_dataset.py — Record the real-world training dataset
===========================================================
AI 1 (DSP / Signal Processing Lead)

WHY THE SAME MICROPHONE FOR EVERY CLASS
---------------------------------------
It is tempting to take the benign class from a public dataset (DEMAND,
FreeSound) and record only the attack classes yourself. Do not.

Measured: two sets of pure noise containing NO attack signal, differing only
in simulated microphone frequency response, were separated by a RandomForest
with accuracy 1.000. Every microphone has its own response curve, self-noise
and anti-alias rolloff. If class 0 comes from one recording chain and class 1
from another, the model learns the microphone fingerprint and never looks at
the signal. It scores perfectly in testing and fails completely in the demo,
where every class arrives through the same microphone.

This script records every class through one microphone in one session, so the
only thing that differs between classes is the transmitted signal.

USAGE
-----
    # full dataset, one laptop plays and records
    python scripts/record_dataset.py --device 11

    # two laptops: this one records only
    python scripts/record_dataset.py --device 11 --mode receiver

    # one class at a time
    python scripts/record_dataset.py --device 11 --classes benign fsk

    # note the distance in the filename
    python scripts/record_dataset.py --device 11 --distance 30cm

OUTPUT
------
    real_data/
      benign/  fsk/  ook/  chirp/  tone/

Hand that folder to AI 2. It loads with --real-data-dir.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed:  pip install sounddevice")
    sys.exit(1)

from scipy.io.wavfile import write as wav_write

from dsp import DSPPipeline
from dsp.recording_quality import check_recording

SAMPLE_RATE = 48000

# files per class, and how the transmit signal is built
CLASS_PLAN = {
    "benign": dict(n=10, kind=None,
                   note="No transmission. Vary the room naturally: typing, "
                        "fan, door, conversation, silence."),
    "fsk":    dict(n=10, kind="fsk",
                   kwargs=dict(freq_mark=18500, freq_space=20500, baud_rate=25)),
    "ook":    dict(n=8,  kind="ook",
                   kwargs=dict(carrier_freq=19000, baud_rate=50)),
    "chirp":  dict(n=8,  kind="chirp",
                   kwargs=dict(freq_start=18000, freq_end=21000)),
    "tone":   dict(n=8,  kind="tone",
                   kwargs=dict(frequency=19500)),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--mode", choices=["loopback", "receiver"], default="loopback")
    ap.add_argument("--classes", nargs="+", default=list(CLASS_PLAN),
                    choices=list(CLASS_PLAN))
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--distance", default="30cm",
                    help="recorded into the filename, e.g. 10cm / 30cm / 100cm")
    ap.add_argument("--out-dir", default="real_data")
    ap.add_argument("--count", type=int, default=None,
                    help="override files per class")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pipeline = DSPPipeline()

    print("=" * 72)
    print("  REAL DATASET RECORDER")
    print("=" * 72)
    try:
        info = sd.query_devices(args.device, "input")
        print(f"  Device   : [{args.device}] {info['name']}")
        if abs(info["default_samplerate"] - SAMPLE_RATE) > 1:
            print(f"  WARNING  : native rate {info['default_samplerate']:.0f} Hz, "
                  f"not 48000. This caused the 15 Aug failure.")
    except Exception as e:
        print(f"  Cannot query device: {e}")
        return 1

    print(f"  Mode     : {args.mode}")
    print(f"  Distance : {args.distance}")
    print(f"  Duration : {args.duration}s per file")
    print(f"  Output   : {os.path.abspath(args.out_dir)}")
    print()
    print("  IMPORTANT: do not move either laptop, change the volume, or")
    print("  switch microphones between classes. If the recording chain")
    print("  changes, the classifier learns the chain instead of the signal.")

    total = sum(args.count or CLASS_PLAN[c]["n"] for c in args.classes)
    print(f"\n  {total} files, ~{total * args.duration / 60:.0f} minutes of audio")

    if args.dry_run:
        for c in args.classes:
            plan = CLASS_PLAN[c]
            print(f"    {c:8s} {args.count or plan['n']} files")
        print("\n  DRY RUN — nothing recorded.")
        return 0

    saved, rejected = 0, 0

    for cls in args.classes:
        plan = CLASS_PLAN[cls]
        n = args.count or plan["n"]
        out = os.path.join(args.out_dir, cls)
        os.makedirs(out, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"  CLASS: {cls}   ({n} files)")
        if plan.get("note"):
            print(f"  {plan['note']}")
        print("=" * 72)

        signal = None
        if plan["kind"]:
            signal = pipeline.generate_test_signal(
                plan["kind"], duration_sec=args.duration + 1.0, **plan["kwargs"])
            signal = (signal / max(1e-9, np.max(np.abs(signal))) * 0.8).astype(np.float32)

        for i in range(1, n + 1):
            name = f"{cls}_{args.distance}_{i:02d}.wav"
            path = os.path.join(out, name)

            if cls == "benign":
                print(f"\n  [{i}/{n}] {name} — make normal room noise, "
                      f"NO transmission")
            elif args.mode == "receiver":
                print(f"\n  [{i}/{n}] {name} — start the {cls} transmission now")
            else:
                print(f"\n  [{i}/{n}] {name} — playing and recording")

            if cls == "benign" or args.mode == "receiver":
                input("      ENTER to record...")
                rec = sd.rec(int(SAMPLE_RATE * args.duration),
                             samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32", device=args.device)
                sd.wait()
            else:
                rec = sd.playrec(signal[:int(SAMPLE_RATE * args.duration)].reshape(-1, 1),
                                 samplerate=SAMPLE_RATE, channels=1,
                                 device=(args.device, sd.default.device[1]),
                                 dtype="float32")
                sd.wait()

            audio = rec[:, 0]
            expected = None
            if cls == "fsk":
                expected = 18500
            elif cls in ("ook", "tone"):
                expected = plan["kwargs"].get("carrier_freq") or plan["kwargs"].get("frequency")

            rep = check_recording(audio, SAMPLE_RATE, expected_hz=expected)

            if rep["ok"]:
                wav_write(path, SAMPLE_RATE, audio)
                saved += 1
                snr = rep.get("tone", {}).get("tone_snr_db")
                extra = f", tone SNR {snr} dB" if snr is not None else ""
                print(f"      saved  peak {rep['peak_lsb']} LSB{extra}")
            else:
                rejected += 1
                print(f"      REJECTED  peak {rep['peak_lsb']} LSB")
                for p in rep["problems"]:
                    print(f"        - {p}")
                print("      Not saved. Re-record this one.")

    print("\n" + "=" * 72)
    print(f"  {saved} saved, {rejected} rejected")
    print("=" * 72)
    if rejected:
        print("  Re-record the rejected files before handing the folder over.")
    print("\n  Next:")
    print(f"    python -m dsp.recording_quality {args.out_dir}/*/*.wav")
    print(f"    then give {args.out_dir}/ to AI 2 (--real-data-dir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
