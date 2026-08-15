"""
Demo Asset Generator
=====================
Creates all visual assets needed for the stage demo and presentation.

Run:
    python demo/generate_demo_assets.py

Outputs (in demo/assets/):
    ├── spectrogram_fsk.png        — FSK attack spectrogram
    ├── spectrogram_ook.png        — OOK attack spectrogram
    ├── spectrogram_chirp.png      — Chirp attack spectrogram
    ├── spectrogram_clean.png      — Clean (no attack) spectrogram
    ├── spectrogram_comparison.png — All 4 side-by-side
    └── demo_attack_fsk.wav        — WAV file for live playback

Who uses this:
    PPT Lead   — screenshots for presentation slides
    Frontend 2 — reference images for demo mode UI
    You        — verify spectrograms look correct
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dsp.dsp_api import DSPPipeline


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = DSPPipeline(sample_rate=48000, n_fft=2048)
    chunk_size = 2048
    sample_rate = 48000

    scenarios = {
        "fsk": {
            "type": "fsk",
            "kwargs": {"freq_mark": 19000, "freq_space": 20500, "baud_rate": 25},
            "title": "FSK Attack (19 kHz / 20.5 kHz)",
        },
        "ook": {
            "type": "ook",
            "kwargs": {"carrier_freq": 19500, "baud_rate": 30},
            "title": "OOK Attack (19.5 kHz carrier)",
        },
        "chirp": {
            "type": "chirp",
            "kwargs": {"freq_start": 18500, "freq_end": 21500, "num_sweeps": 6},
            "title": "Chirp Sweep (18.5 – 21.5 kHz)",
        },
    }

    print("=" * 60)
    print("  Acoustic Cybersecurity System — Demo Asset Generator")
    print("=" * 60)

    # ── Generate attack spectrograms ──
    for name, config in scenarios.items():
        print(f"\n  Generating {name.upper()} spectrogram...")
        pipeline.reset()

        signal = pipeline.generate_test_signal(
            config["type"],
            duration_sec=5.0,
            snr_db=15,
            **config["kwargs"],
        )

        # Process through pipeline chunk by chunk
        for i in range(0, len(signal) - chunk_size, chunk_size):
            chunk = signal[i : i + chunk_size]
            pipeline.process(chunk)

        filepath = os.path.join(output_dir, f"spectrogram_{name}.png")
        pipeline.save_spectrogram_image(filepath)

    # ── Generate clean (no attack) spectrogram ──
    print("\n  Generating CLEAN spectrogram...")
    pipeline.reset()
    from dsp.utils import SignalGenerator

    gen = SignalGenerator(sample_rate=sample_rate)
    noise = gen.generate_ambient_noise(duration_sec=5.0, amplitude=0.3)
    for i in range(0, len(noise) - chunk_size, chunk_size):
        chunk = noise[i : i + chunk_size]
        pipeline.process(chunk)

    filepath = os.path.join(output_dir, "spectrogram_clean.png")
    pipeline.save_spectrogram_image(filepath)

    # ── Generate comparison figure ──
    print("\n  Generating comparison figure...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(
            "Acoustic Cybersecurity System — Ultrasonic Attack Detection Spectrograms",
            fontsize=16,
            fontweight="bold",
        )

        all_names = ["clean", "fsk", "ook", "chirp"]
        titles = [
            "Clean Environment (No Attack)",
            "FSK Attack (19 kHz / 20.5 kHz)",
            "OOK Attack (19.5 kHz carrier)",
            "Chirp Sweep (18.5 – 21.5 kHz)",
        ]

        for idx, (sname, title) in enumerate(zip(all_names, titles)):
            pipeline.reset()

            if sname == "clean":
                sig = gen.generate_ambient_noise(duration_sec=5.0, amplitude=0.3)
            else:
                cfg = scenarios[sname]
                sig = pipeline.generate_test_signal(
                    cfg["type"], duration_sec=5.0, snr_db=15, **cfg["kwargs"]
                )

            for i in range(0, len(sig) - chunk_size, chunk_size):
                pipeline.process(sig[i : i + chunk_size])

            data = pipeline.get_spectrogram_snapshot()
            ax = axes[idx // 2][idx % 2]

            if data["spectrogram_db"].size > 0:
                ax.imshow(
                    data["spectrogram_db"],
                    aspect="auto",
                    origin="lower",
                    cmap="inferno",
                    extent=[
                        0, data["duration_sec"],
                        18.0, 22.0,
                    ],
                    vmin=-100,
                    vmax=0,
                )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (kHz)")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        comparison_path = os.path.join(output_dir, "spectrogram_comparison.png")
        fig.savefig(comparison_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {comparison_path}")
    except ImportError:
        print("  (matplotlib not available — skipping comparison figure)")

    # ── Generate demo WAV file for live playback ──
    print("\n  Generating demo attack WAV...")
    demo_signal = pipeline.generate_test_signal(
        "fsk",
        duration_sec=10.0,
        snr_db=20,
        freq_mark=19000,
        freq_space=20000,
        baud_rate=20,
    )
    wav_path = os.path.join(output_dir, "demo_attack_fsk.wav")
    pipeline.save_wav(demo_signal, wav_path)
    print(f"  Saved: {wav_path}")

    print("\n" + "=" * 60)
    print(f"  All assets saved to: {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
