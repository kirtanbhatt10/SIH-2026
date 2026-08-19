"""
Dataset V2 inspection tool (Task 15 + Task 16).

Does NOT train a model. Two things only:

1. Feature distribution sanity check: for the built dataset on disk,
   report min/max/mean/std/NaN/Inf per feature, and flag features that look
   constant, broken, or numerically unstable.

2. Sample visualizations: regenerate a small, fresh set of example samples
   (same generator, small dedicated seed) and plot waveform + spectrum for
   one example per class, saving PNGs (already covered by ai/.gitignore).

    python inspect_dataset.py --dataset-dir ../../training_data_v2 --plots-dir ./inspection_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AI_DIR = _HERE.parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

try:
    from . import config_v2 as cfg
    from .generator import generate_sample, make_shared_dsp_components, CLASS_NAMES
except ImportError:  # pragma: no cover
    import config_v2 as cfg  # type: ignore
    from generator import generate_sample, make_shared_dsp_components, CLASS_NAMES  # type: ignore

from dsp import SAMPLE_RATE  # noqa: E402


def feature_distribution_report(dataset_dir: Path) -> list[dict]:
    X = np.load(dataset_dir / "features.npy")
    y = np.load(dataset_dir / "labels.npy")
    with open(dataset_dir / "dataset_info.json") as f:
        info = json.load(f)
    names = info["feature_names"]

    rows = []
    for i, name in enumerate(names):
        col = X[:, i]
        finite = col[np.isfinite(col)]
        row = {
            "feature": name,
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "std": float(np.std(finite)) if len(finite) else float("nan"),
            "nan_count": int(np.isnan(col).sum()),
            "inf_count": int(np.isinf(col).sum()),
        }
        row["suspicious_constant"] = row["std"] < 1e-9
        rows.append(row)

    print("=" * 100)
    print("  Feature distribution report")
    print("=" * 100)
    header = f"{'feature':<26}{'min':>12}{'max':>12}{'mean':>12}{'std':>12}{'nan':>6}{'inf':>6}  flag"
    print(header)
    for row in rows:
        flag = "CONSTANT?" if row["suspicious_constant"] else ""
        print(
            f"{row['feature']:<26}{row['min']:>12.4f}{row['max']:>12.4f}"
            f"{row['mean']:>12.4f}{row['std']:>12.4f}{row['nan_count']:>6}{row['inf_count']:>6}  {flag}"
        )

    suspicious = [r["feature"] for r in rows if r["suspicious_constant"]]
    print("-" * 100)
    if suspicious:
        print(f"  SUSPICIOUS (near-constant across the whole dataset): {suspicious}")
    else:
        print("  No near-constant features detected.")

    print("\n  Per-class feature means (first 8 features, for a quick sanity look):")
    for class_idx, class_name in sorted(CLASS_NAMES.items()):
        mask = y == class_idx
        means = X[mask, :8].mean(axis=0)
        print(f"    {class_name:<8}: " + " ".join(f"{m:8.3f}" for m in means))

    return rows


def make_example_plots(plots_dir: Path, seed: int = 999):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    preprocessor, extractor = make_shared_dsp_components()

    for class_idx, class_name in sorted(CLASS_NAMES.items()):
        sample = generate_sample(class_idx, 0, seed, preprocessor, extractor)
        audio = sample.audio.astype(np.float64)
        meta = sample.metadata

        n = len(audio)
        t = np.arange(n) / SAMPLE_RATE * 1000.0  # ms
        windowed = audio * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        peak_freq = float(freqs[np.argmax(spectrum)]) if len(spectrum) else float("nan")

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].plot(t, audio, linewidth=0.7)
        axes[0].set_title(f"{class_name} -- waveform (label={class_idx})")
        axes[0].set_xlabel("time (ms)")
        axes[0].set_ylabel("amplitude")

        axes[1].plot(freqs / 1000.0, spectrum, linewidth=0.7)
        axes[1].axvspan(18.0, 21.0, color="orange", alpha=0.15, label="detection band")
        axes[1].set_xlim(0, 24)
        axes[1].set_title(f"spectrum -- dominant freq ~{peak_freq/1000:.2f} kHz, snr={meta['snr']}")
        axes[1].set_xlabel("frequency (kHz)")
        axes[1].set_ylabel("magnitude")
        axes[1].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        out_path = plots_dir / f"{class_name}_example.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  wrote {out_path}  (peak_freq~{peak_freq:.0f} Hz, snr={meta['snr']}, "
              f"noise_type={meta['noise_type']}, variant={meta.get('benign_variant')})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(_AI_DIR / "training_data_v2"))
    ap.add_argument("--plots-dir", default=str(_HERE / "inspection_plots"))
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    feature_distribution_report(Path(args.dataset_dir))

    if not args.skip_plots:
        print()
        make_example_plots(Path(args.plots_dir))


if __name__ == "__main__":
    main()
