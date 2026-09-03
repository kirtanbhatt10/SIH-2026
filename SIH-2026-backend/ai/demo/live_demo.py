"""
Live Demo Script — Stage Presentation
========================================
Run this on stage to demonstrate the full detection pipeline.

What it does:
    1. Generates a silent ultrasonic attack signal
    2. Processes it chunk-by-chunk through the DSP pipeline
    3. Prints a live terminal-based spectrogram visualization
    4. Shows detection results and threat classification in real time
    5. Displays final verdict

Run:
    python demo/live_demo.py

    Options:
    python demo/live_demo.py --attack fsk    (default)
    python demo/live_demo.py --attack ook
    python demo/live_demo.py --attack chirp
    python demo/live_demo.py --attack clean  (no attack, just noise)
    python demo/live_demo.py --duration 8    (seconds)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dsp.dsp_api import DSPPipeline


# ── ANSI color codes for terminal ──
def _enable_ansi():
    """
    Turn on ANSI escape handling.

    Windows consoles do not process ANSI sequences unless
    ENABLE_VIRTUAL_TERMINAL_PROCESSING is set. Without it, cursor-movement
    codes are printed literally and the live display shreds itself.
    Returns False if ANSI is unavailable, in which case we fall back to
    plain scrolling output instead of in-place redraw.
    """
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


USE_ANSI = _enable_ansi()


def _force_utf8_stdout():
    """
    The box-drawing and block characters used by the display are not
    representable in Windows' legacy cp1252 codepage. If stdout is using a
    non-UTF-8 encoding, reconfigure it; otherwise printing raises
    UnicodeEncodeError mid-demo.
    """
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" not in enc:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdout()

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"

if not USE_ANSI:
    RED = GREEN = YELLOW = CYAN = WHITE = DIM = BOLD = RESET = BG_RED = BG_GREEN = ""


def render_bar(value: float, width: int = 40, char: str = "█") -> str:
    """Render a horizontal bar chart."""
    filled = int(value * width)
    empty = width - filled

    if value > 0.7:
        color = RED
    elif value > 0.3:
        color = YELLOW
    else:
        color = GREEN

    return f"{color}{char * filled}{DIM}{'░' * empty}{RESET}"


def render_spectrum(magnitudes_norm: list, freq_axis: list) -> str:
    """Render a compact frequency spectrum in the terminal."""
    # Downsample to ~20 bars for terminal display
    n_bars = 20
    step = max(1, len(magnitudes_norm) // n_bars)
    bars = magnitudes_norm[::step][:n_bars]

    lines = []
    max_height = 10  # rows

    for row in range(max_height, 0, -1):
        threshold = row / max_height
        line = "  │"
        for val in bars:
            if val >= threshold:
                if val > 0.7:
                    line += f"{RED}██{RESET}"
                elif val > 0.3:
                    line += f"{YELLOW}██{RESET}"
                else:
                    line += f"{GREEN}██{RESET}"
            else:
                line += "  "
        lines.append(line)

    # Axis
    axis = "  └" + "──" * len(bars)
    lines.append(axis)

    # Labels
    if len(freq_axis) > 0:
        labels = f"   {freq_axis[0]/1000:.0f}kHz" + " " * (len(bars) * 2 - 14) + f"{freq_axis[-1]/1000:.0f}kHz"
        lines.append(labels)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Acoustic Cybersecurity System — Live Demo")
    parser.add_argument("--attack", choices=["fsk", "ook", "chirp", "clean"], default="fsk")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--snr", type=float, default=15.0)
    args = parser.parse_args()

    pipeline = DSPPipeline(sample_rate=48000, n_fft=2048)
    chunk_size = 2048
    sample_rate = 48000
    chunk_duration = chunk_size / sample_rate  # ~42.7 ms

    # ── Generate the signal ──
    if args.attack == "clean":
        from dsp.utils import SignalGenerator
        gen = SignalGenerator(sample_rate=sample_rate)
        signal = gen.generate_ambient_noise(duration_sec=args.duration, amplitude=0.3)
        attack_label = "NONE (Clean Environment)"
    else:
        attack_configs = {
            "fsk": {"freq_mark": 19000, "freq_space": 20500, "baud_rate": 25},
            "ook": {"carrier_freq": 19500, "baud_rate": 30},
            "chirp": {"freq_start": 18500, "freq_end": 21500, "num_sweeps": 6},
        }
        signal = pipeline.generate_attack_signal(
            args.attack,
            duration_sec=args.duration,
            snr_db=args.snr,
            **attack_configs[args.attack],
        )
        attack_label = f"{args.attack.upper()} Ultrasonic Exfiltration"

    # ── Header ──
    os.system("cls" if os.name == "nt" else "clear")
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║           ACOUSTIC CYBERSECURITY SYSTEM — LIVE DETECTION          ║
║              Ultrasonic Data Exfiltration Detector             ║
╚══════════════════════════════════════════════════════════════╝{RESET}

  {DIM}Attack Type  :{RESET} {BOLD}{attack_label}{RESET}
  {DIM}Duration     :{RESET} {args.duration}s
  {DIM}Sample Rate  :{RESET} {sample_rate} Hz
  {DIM}FFT Size     :{RESET} {chunk_size}
  {DIM}Detection Band:{RESET} 18 kHz – 21 kHz

{YELLOW}  ▶ Starting real-time analysis...{RESET}
""")
    time.sleep(1.5)

    # ── Process chunk by chunk ──
    n_chunks = (len(signal) - chunk_size) // chunk_size
    alert_count = 0
    proc_times = []          # DSP time per chunk, excluding display pacing
    prev_height = 0          # rendered height of the previous frame

    for i in range(n_chunks):
        start_idx = i * chunk_size
        chunk = signal[start_idx : start_idx + chunk_size]
        _t0 = time.perf_counter()
        result = pipeline.process(chunk)
        proc_times.append(time.perf_counter() - _t0)

        elapsed = (i + 1) * chunk_duration
        score = result["suspicion_score"]
        is_sus = result["is_suspicious"]
        mod_type = result["analysis"]["modulation_type"]
        confidence = result["analysis"]["confidence"]
        peak_freq = result["spectrogram"]["peak_freq_hz"]
        mags = result["spectrogram"]["magnitudes_norm"]
        freqs = result["spectrogram"]["freq_axis"]

        if is_sus:
            alert_count += 1

        # ── Render frame ──
        if is_sus:
            status = f"{BG_RED}{WHITE}{BOLD}  ** THREAT DETECTED **  {RESET}"
        else:
            status = f"{BG_GREEN}{WHITE}  OK  CLEAR  {RESET}"

        frame = f"""
  {DIM}Time:{RESET} {elapsed:6.2f}s / {args.duration:.1f}s    {status}

  {BOLD}Ultrasonic Spectrum (18-21 kHz):{RESET}
{render_spectrum(mags, freqs)}

  {DIM}Peak Frequency :{RESET} {BOLD}{peak_freq:,.0f} Hz{RESET}
  {DIM}Suspicion Score:{RESET} {render_bar(score)} {BOLD}{score:.3f}{RESET}
  {DIM}Modulation     :{RESET} {BOLD}{mod_type.upper()}{RESET} (confidence: {confidence:.2f})
  {DIM}Alerts         :{RESET} {RED if alert_count > 0 else GREEN}{alert_count}{RESET}
"""

        # Rewind by the frame's ACTUAL height. A hardcoded cursor-up (\033[18A)
        # did not match the rendered height, so each frame overwrote the wrong
        # lines and the display shredded itself on Windows Terminal.
        if i > 0 and prev_height and USE_ANSI:
            sys.stdout.write(f"\033[{prev_height}A")
            # Clear each line before redrawing, so a shorter frame cannot leave
            # fragments of the previous one behind.
            for _ in range(prev_height):
                sys.stdout.write("\033[2K\033[1B")
            sys.stdout.write(f"\033[{prev_height}A")

        sys.stdout.write(frame)
        sys.stdout.flush()
        prev_height = frame.count("\n")

        # Cosmetic pacing only — EXCLUDED from the reported processing time.
        # (Reporting wall-clock/chunk here would have shown ~75 ms/chunk for a
        #  pipeline that actually runs in ~1.2 ms.)
        time.sleep(max(0.05, chunk_duration * 0.5))

    # ── Final verdict ──
    verdict = pipeline.get_verdict()
    stats = pipeline.get_stats()

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║                     ANALYSIS COMPLETE                        ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

    if verdict["is_threat"]:
        print(f"""  {BG_RED}{WHITE}{BOLD}  ⚠ COVERT ULTRASONIC TRANSMISSION DETECTED ⚠  {RESET}

  {DIM}Average Suspicion  :{RESET} {RED}{verdict['average_suspicion']:.4f}{RESET}
  {DIM}Max Suspicion      :{RESET} {RED}{verdict['max_suspicion']:.4f}{RESET}
  {DIM}Carrier Frequencies:{RESET} {BOLD}{verdict['consistent_carriers']}{RESET} Hz
  {DIM}Threat Duration    :{RESET} {verdict['threat_duration_sec']:.3f} s
  {DIM}Chunks Analyzed    :{RESET} {verdict['chunks_analyzed']}
  {DIM}Total Alerts       :{RESET} {RED}{stats['total_alerts']}{RESET}

  {YELLOW}→ This signal shows structure consistent with covert acoustic signalling.
  → It carries no network traffic, so network-layer monitoring would not see it.
  → Detected here from a synthetic signal; see docs/KNOWN_ISSUES.md for limits.{RESET}
""")
    else:
        print(f"""  {BG_GREEN}{WHITE}{BOLD}  ✓ NO THREAT DETECTED  {RESET}

  {DIM}Average Suspicion :{RESET} {GREEN}{verdict['average_suspicion']:.4f}{RESET}
  {DIM}Chunks Analyzed   :{RESET} {verdict['chunks_analyzed']}
  {DIM}Environment       :{RESET} Clean

  {GREEN}→ No structured ultrasonic transmissions detected.{RESET}
""")

    # Measured DSP time only — the display sleep above is not counted.
    if proc_times:
        avg_ms = sum(proc_times) / len(proc_times) * 1000
        max_ms = max(proc_times) * 1000
        realtime_factor = chunk_duration / (sum(proc_times) / len(proc_times))
        print(f"  {DIM}DSP processing: avg {avg_ms:.2f} ms/chunk, "
              f"max {max_ms:.2f} ms  ({realtime_factor:.0f}x faster than real time){RESET}")
        print(f"  {DIM}(display pacing excluded; chunk = {chunk_duration*1000:.1f} ms of audio){RESET}")
    print()


if __name__ == "__main__":
    main()
