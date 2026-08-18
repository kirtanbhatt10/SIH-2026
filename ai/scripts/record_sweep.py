"""
record_sweep.py — Automated hardware sweep: play a tone and record it
======================================================================
AI 1 (DSP / Signal Processing Lead)

WHY THIS EXISTS
---------------
The hardware sweep needs ~10 tones recorded. Doing that by hand means running
record_wav.py ten times, starting playback on a second machine each time, and
hoping the timing lines up. This automates it: for each frequency it starts
the recording, plays the tone through the speaker, stops, and validates the
capture — in one pass.

ONE LAPTOP OR TWO?
------------------
Both work, and they measure different things.

  --mode loopback   (default, ONE laptop)
      This laptop plays the tone AND records it. Measures the combined
      speaker + air + microphone response of a single machine. Simplest,
      no coordination, good enough for the feasibility report.

  --mode receiver   (TWO laptops)
      This laptop only records; a second machine plays the tones. Matches
      the real covert-channel scenario (data leaves machine A by speaker,
      machine B listens) and lets you vary distance properly.

Start with loopback. Move to two machines once you know the band works.

USAGE
-----
    # one laptop, full sweep
    python scripts/record_sweep.py --device 11

    # two laptops: this one records only
    python scripts/record_sweep.py --device 11 --mode receiver

    # custom frequencies / duration
    python scripts/record_sweep.py --device 11 --freqs 15000 17000 18000 --duration 8

    # see what it would do
    python scripts/record_sweep.py --device 11 --dry-run
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

from dsp.recording_quality import check_recording

SAMPLE_RATE = 48000
DEFAULT_FREQS = [1000, 5000, 10000, 15000, 17000, 18000, 19000, 20000, 21000]


def make_tone(freq: float, duration: float, sample_rate: int = SAMPLE_RATE,
              amplitude: float = 0.8, fade_ms: float = 50.0) -> np.ndarray:
    """Sine tone with short fades, so the speaker does not click on start/stop.

    A hard edge produces a broadband transient that shows up as spurious
    energy across the spectrum and can be mistaken for signal.
    """
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * freq * t)

    fade = int(sample_rate * fade_ms / 1000.0)
    if fade > 0 and 2 * fade < n:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return tone.astype(np.float32)


def record_one(freq, duration, device, mode, out_dir, lead_in=0.5):
    """Record one frequency. Returns (filename, report) or (None, report)."""
    label = "baseline_quiet" if freq is None else f"rx_{int(freq)}"
    path = os.path.join(out_dir, f"{label}.wav")

    total = duration + lead_in * 2
    n_frames = int(SAMPLE_RATE * total)

    if mode == "loopback" and freq is not None:
        # Pad the tone with silence so the recording captures the room
        # before and after — needed for a local noise-floor estimate.
        pad = np.zeros(int(SAMPLE_RATE * lead_in), dtype=np.float32)
        playback = np.concatenate([pad, make_tone(freq, duration), pad])
        captured = sd.playrec(
            playback.reshape(-1, 1),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=(device, sd.default.device[1]),
            dtype="float32",
        )
        sd.wait()
    else:
        # receiver mode, or a silent baseline: record only
        captured = sd.rec(n_frames, samplerate=SAMPLE_RATE, channels=1,
                          dtype="float32", device=device)
        sd.wait()

    audio = captured[:, 0]
    report = check_recording(audio, SAMPLE_RATE, expected_hz=freq)

    if report["ok"]:
        wav_write(path, SAMPLE_RATE, audio)
        return path, report
    return None, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, required=True,
                    help="input device index (from scripts/enum_devices.py)")
    ap.add_argument("--mode", choices=["loopback", "receiver"], default="loopback",
                    help="loopback = this laptop plays and records (default); "
                         "receiver = record only, another machine plays")
    ap.add_argument("--freqs", type=int, nargs="+", default=DEFAULT_FREQS)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("  HARDWARE SWEEP")
    print("=" * 72)
    try:
        info = sd.query_devices(args.device, "input")
        print(f"  Input device : [{args.device}] {info['name']}")
        print(f"  Native rate  : {info['default_samplerate']:.0f} Hz")
        if abs(info["default_samplerate"] - SAMPLE_RATE) > 1:
            print("  WARNING: native rate is not 48000 Hz. This is the exact")
            print("           condition that destroyed the 15 Aug sweep.")
    except Exception as e:
        print(f"  Cannot query device {args.device}: {e}")
        return 1

    print(f"  Mode         : {args.mode}")
    print(f"  Frequencies  : {', '.join(str(f) for f in args.freqs)} Hz")
    print(f"  Duration     : {args.duration}s each")
    print(f"  Output       : {os.path.abspath(args.out_dir)}")

    if args.dry_run:
        print("\n  DRY RUN — nothing recorded.")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    # ── Baseline first: the noise floor every SNR is measured against ──
    if not args.skip_baseline:
        print("\n" + "-" * 72)
        print("  BASELINE — stay silent, do not play anything")
        input("  Press ENTER when the room is quiet...")
        path, rep = record_one(None, args.duration, args.device, "receiver",
                               args.out_dir)
        status = "saved" if path else "REJECTED"
        print(f"  baseline_quiet.wav  {status}  peak {rep['peak_lsb']} LSB")
        if not path:
            for p in rep["problems"]:
                print(f"    - {p}")
        results.append(("baseline", None, rep, path is not None))

    # ── Sweep ──
    for freq in args.freqs:
        print("\n" + "-" * 72)
        print(f"  {freq} Hz")
        if args.mode == "receiver":
            print(f"  Start playing the {freq} Hz tone on the transmitter now.")
            input("  Press ENTER to begin recording...")
        else:
            print("  Playing and recording...")

        path, rep = record_one(freq, args.duration, args.device, args.mode,
                               args.out_dir)

        tone = rep.get("tone", {})
        snr = tone.get("tone_snr_db")
        if path:
            print(f"  saved rx_{freq}.wav   peak {rep['peak_lsb']} LSB   "
                  f"SNR {snr} dB   tone_present={tone.get('tone_present')}")
        else:
            print(f"  REJECTED   peak {rep['peak_lsb']} LSB   SNR {snr} dB")
            for p in rep["problems"]:
                print(f"    - {p}")
        results.append((freq, snr, rep, path is not None))

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'freq':>8s}  {'SNR dB':>8s}  {'peak LSB':>9s}  result")
    print("  " + "-" * 46)

    ceiling = None
    for freq, snr, rep, saved in results:
        name = "baseline" if freq is None or freq == "baseline" else f"{freq} Hz"
        snr_s = f"{snr:8.2f}" if snr is not None else "       -"
        verdict = "OK" if saved else "REJECTED"
        print(f"  {name:>8s}  {snr_s}  {rep['peak_lsb']:9.1f}  {verdict}")
        if saved and isinstance(freq, int) and snr is not None and snr >= 15.0:
            ceiling = freq

    print()
    if ceiling:
        print(f"  Measured usable ceiling: {ceiling} Hz  (>= 15 dB SNR)")
        if ceiling < 18000:
            print("  This is BELOW the 18 kHz detection band. That is a valid")
            print("  finding — tell Backend 2 before they choose a carrier.")
        else:
            print("  This reaches the 18-21 kHz detection band.")
    else:
        print("  No frequency cleared 15 dB SNR. Check volume and distance,")
        print("  then re-run. If it persists, that is your hardware limit.")

    print("\n  Next:")
    print("    python -m dsp.recording_quality *.wav")
    print("    python scripts/make_evidence.py --curated --input-dir .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
