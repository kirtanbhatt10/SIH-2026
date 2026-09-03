"""Generate distribution plots for realism audit (saved under experiment plots/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, FEATURE_NAMES, PLOTS_DIR
from .data import AuditData


def _plot_feature_by_class(
    data: AuditData,
    feature: str,
    out_path: Path,
) -> None:
    fi = FEATURE_NAMES.index(feature)
    X = data.dev_features()
    y = data.dev_labels()

    fig, ax = plt.subplots(figsize=(8, 4))
    for cid, cname in CLASS_NAMES.items():
        vals = X[y == cid, fi]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.45, label=cname, density=True)
    ax.set_title(f"Development distribution: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pair_scatter(
    data: AuditData,
    feat_x: str,
    feat_y: str,
    class_a: str,
    class_b: str,
    out_path: Path,
) -> None:
    from .config import CLASS_ID

    ia, ib = CLASS_ID[class_a], CLASS_ID[class_b]
    ix, iy = FEATURE_NAMES.index(feat_x), FEATURE_NAMES.index(feat_y)
    X = data.dev_features()
    y = data.dev_labels()
    mask = (y == ia) | (y == ib)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cid, cname, color in ((ia, class_a, "C0"), (ib, class_b, "C1")):
        sub = X[mask & (y == cid)]
        ax.scatter(sub[:, ix], sub[:, iy], s=8, alpha=0.35, label=cname, c=color)
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"{class_a} vs {class_b}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_plots(data: AuditData) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    key_features = [
        "tonal_prominence",
        "harmonic_ratio",
        "spectral_flatness",
        "duty_cycle",
        "bit_rate_estimate",
        "spectral_centroid",
        "peak_frequency",
        "ultrasonic_energy",
    ]
    for feat in key_features:
        path = PLOTS_DIR / f"dist_{feat}.png"
        _plot_feature_by_class(data, feat, path)
        saved.append(str(path))

    pairs = [
        ("fsk", "tone", "tonal_prominence", "peak_frequency"),
        ("ook", "benign", "duty_cycle", "spectral_flatness"),
        ("ook", "tone", "bit_rate_estimate", "tonal_prominence"),
    ]
    for a, b, fx, fy in pairs:
        path = PLOTS_DIR / f"pair_{a}_vs_{b}_{fx}_{fy}.png"
        _plot_pair_scatter(data, fx, fy, a, b, path)
        saved.append(str(path))

    return saved
