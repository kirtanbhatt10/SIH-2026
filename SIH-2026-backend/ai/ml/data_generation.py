"""Training-data acquisition for the AI 2 ML subsystem.

PROVENANCE — read this before quoting any number
------------------------------------------------
There are three tiers of data, and they are NOT interchangeable:

  1. "dsp_synthetic"  synthetic AUDIO pushed through the real
                      dsp.FeatureExtractor.  This is the default and what
                      every reported baseline should use.
  2. "fabricated"     feature vectors invented in this file from hand-written
                      prototype arrays.  No audio, no FFT, no DSP layer.
                      FALLBACK ONLY, for when `dsp` cannot be imported.
  3. "dsp_real"       real over-the-air recordings through the real
                      FeatureExtractor.  Blocked on AI 1's physical sweep.

An earlier version of this module used tier 2 exclusively and labelled the
output "synthetic".  That conflation produced a wrong conclusion: Phase 2
reported RF/SVM/LogReg as a statistical tie with SVM marginally ahead.  On
tier-1 data SVM collapses to ~0.46 accuracy because the real features have
wildly different scales (energy_db is negative, peak_frequency is in the
thousands) while the fabricated vectors were all roughly 0-1.  The fabricated
data could not expose the problem.

`load_dataset()` therefore tries tier 1 first and only falls back to tier 2
with a loud warning.  The tier used is recorded in metadata.json under
"source" and must appear next to any metric in RESULTS.md.

USAGE
-----
    from data_generation import load_dataset
    X, y, source = load_dataset(samples_per_class=500)

CLI
---
    python data_generation.py --samples_per_class 500
    python data_generation.py --force-fabricated     # tier 2 on purpose
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from config import (
    CLASS_NAMES,
    N_FEATURES,
    RANDOM_STATE,
    SOURCE_DSP_SYNTHETIC,
    SOURCE_FABRICATED,
)


# --------------------------------------------------------------------------- #
# Tier 1 — real DSP pipeline (preferred)
# --------------------------------------------------------------------------- #

def _add_dsp_to_path() -> None:
    """Make `import dsp` work whether ml/ is run from ai/ or from repo root."""
    here = Path(__file__).resolve().parent          # ai/ml
    for candidate in (here.parent, here.parent.parent):   # ai/ , repo root
        if (candidate / "dsp" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


def dsp_available() -> bool:
    _add_dsp_to_path()
    try:
        import dsp  # noqa: F401
        return True
    except Exception:
        return False


def generate_from_dsp(output_dir: str = "training_data",
                      samples_per_class: int = 500,
                      seed: int = RANDOM_STATE):
    """Tier 1: synthetic audio -> real dsp.FeatureExtractor.

    Delegates to DSPPipeline.generate_training_dataset(), which pushes every
    sample through the SAME AudioPreprocessor the live pipeline uses.  That is
    what prevents train/serve skew, and AI 1 enforces it with
    tests/test_training_serving_consistency.py.  Do not reimplement it here.
    """
    _add_dsp_to_path()
    from dsp import DSPPipeline

    np.random.seed(seed)
    pipeline = DSPPipeline()
    pipeline.generate_training_dataset(
        output_dir=output_dir,
        samples_per_class=samples_per_class,
    )

    out = Path(output_dir)
    X = np.load(out / "features.npy")
    y = np.load(out / "labels.npy")

    _write_metadata(out, X, y, samples_per_class, seed, SOURCE_DSP_SYNTHETIC)
    return X, y


# --------------------------------------------------------------------------- #
# Tier 2 — fabricated vectors (fallback only)
# --------------------------------------------------------------------------- #
# Kept so the pipeline is runnable without the dsp package, e.g. in a bare CI
# container.  Numbers produced from this tier are NOT measurements of the
# detector and must never appear in a slide.

FSK_CORRUPTION = 0.50
OOK_CORRUPTION = 0.15
FSK_DELTA = {13: 0.90, 14: 0.35, 25: 0.40}
OOK_DELTA = {18: 1.20, 21: 0.16, 24: -0.18}


def _prototypes():
    noise = np.full(N_FEATURES, 0.10)
    noise[:4] = 0.16
    noise[4:10] = 0.12
    noise[10:17] = 0.10
    noise[17:21] = 0.08
    noise[21:27] = 0.14
    noise[27:32] = 0.16

    benign = np.array([
        0.05, 0.10, 0.12, -30.0,
        0.30, 0.55, 0.90, 0.45, 0.60, 3.50,
        0.18, 0.10, 0.03, 1.20, 0.05, 0.20,
        0.05, 0.35, 3.20, 0.45,
        0.10, 0.06, 0.02, 0.85, 0.50, 0.02,
        0.10, 0.08, 0.45, 0.00, 0.45, 0.85,
    ], dtype=float)

    chirp = np.array([
        0.55, 0.60, 0.52, -9.0,
        0.65, 0.62, 0.55, 0.85, 0.30, 2.00,
        0.50, 0.40, 0.18, 1.50, 0.30, 0.35,
        0.30, 0.60, 3.00, 0.30,
        0.58, 0.20, 0.35, 0.50, 0.55, 0.85,
        0.58, 0.20, 0.92, 0.10, 0.82, 0.35,
    ], dtype=float)

    tone = np.array([
        0.55, 0.55, 0.48, -11.0,
        0.48, 0.18, 0.06, 0.40, 0.80, 1.40,
        0.75, 0.95, 0.90, 1.00, 0.02, 0.02,
        0.90, 0.95, 1.80, 0.10,
        0.55, 0.05, 0.10, 0.30, 0.98, 0.05,
        0.55, 0.05, 0.75, 0.30, 0.45, 0.10,
    ], dtype=float)

    return {0: (benign, noise), 3: (chirp, noise), 4: (tone, noise)}


def _fsk_ook():
    shared = np.array([
        0.58, 0.61, 0.53, -9.0,
        0.53, 0.36, 0.37, 0.66, 0.45, 2.60,
        0.71, 0.75, 0.50, 1.50, 0.25, 0.16,
        0.53, 0.75, 3.10, 0.21,
        0.57, 0.20, 0.29, 0.50, 0.42, 0.70,
        0.57, 0.24, 0.92, 0.10, 0.82, 0.42,
    ], dtype=float)
    fsk, ook = shared.copy(), shared.copy()
    for i, v in FSK_DELTA.items():
        fsk[i] += v
    for i, v in OOK_DELTA.items():
        ook[i] += v
    return shared, fsk, ook


def generate_fabricated(output_dir: str = "training_data",
                        samples_per_class: int = 500,
                        seed: int = RANDOM_STATE):
    """Tier 2 fallback. Invents feature vectors; touches no audio."""
    warnings.warn(
        "Using FABRICATED feature vectors -- the dsp package was not "
        "importable. These are NOT measurements of the detector. Any number "
        "derived from them must be labelled source='fabricated' and must not "
        "appear in the report or PPT.",
        RuntimeWarning, stacklevel=2,
    )

    rng = np.random.default_rng(seed)
    easy = _prototypes()
    noise = easy[0][1]
    _, fsk_mean, ook_mean = _fsk_ook()

    X_parts, y_parts = [], []
    for label in sorted(CLASS_NAMES):
        if label == 1:
            mean = fsk_mean
        elif label == 2:
            mean = ook_mean
        else:
            mean = easy[label][0]
        X = mean[None, :] + rng.normal(0.0, noise[None, :],
                                       size=(samples_per_class, N_FEATURES))
        X_parts.append(np.clip(X, 0.0, None))
        y_parts.append(np.full(samples_per_class, label, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    fskidx = np.where(y == 1)[0]
    ookidx = np.where(y == 2)[0]
    nf = int(FSK_CORRUPTION * len(fskidx))
    no = int(OOK_CORRUPTION * len(ookidx))

    for i, v in FSK_DELTA.items():
        X[fskidx[:nf], i] -= v
    for i, v in OOK_DELTA.items():
        X[fskidx[:nf], i] += v
    for i, v in OOK_DELTA.items():
        X[ookidx[:no], i] -= v
    for i, v in FSK_DELTA.items():
        X[ookidx[:no], i] += v
    X = np.clip(X, 0.0, None)

    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    assert X.shape == (len(CLASS_NAMES) * samples_per_class, N_FEATURES)
    assert not np.isnan(X).any() and not np.isinf(X).any()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", X)
    np.save(out / "labels.npy", y)
    with open(out / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)
    _write_metadata(out, X, y, samples_per_class, seed, SOURCE_FABRICATED)
    return X, y


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #

def _write_metadata(out: Path, X, y, samples_per_class, seed, source):
    label_map = {str(k): v for k, v in CLASS_NAMES.items()}
    notes = {
        SOURCE_DSP_SYNTHETIC: (
            "Synthetic AUDIO through the real dsp.FeatureExtractor. Valid as a "
            "detector baseline. NOT over-the-air performance."
        ),
        SOURCE_FABRICATED: (
            "FABRICATED feature vectors. No audio, no FFT, no DSP layer. "
            "Structural testing only. Never report as a measurement."
        ),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "source": source,
            "source_note": notes.get(source, ""),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": len(CLASS_NAMES),
            "samples_per_class": samples_per_class,
            "seed": seed,
            "label_map": label_map,
            "generator": "ai.ml.data_generation",
        }, f, indent=2)


def load_dataset(samples_per_class: int = 500,
                 seed: int = RANDOM_STATE,
                 output_dir: str | None = None,
                 force_fabricated: bool = False,
                 reuse_existing: bool = True):
    """Return (X, y, source).

    Order of preference:
      1. an existing dataset on disk (source read from metadata.json)
      2. tier 1, generated via the real DSP pipeline
      3. tier 2, fabricated -- only if `dsp` cannot be imported
    """
    import config
    out = Path(output_dir) if output_dir else config.DATA_DIR

    if reuse_existing and (out / "features.npy").exists() \
            and (out / "labels.npy").exists():
        X = np.load(out / "features.npy")
        y = np.load(out / "labels.npy")
        source = "unknown"
        meta = out / "metadata.json"
        if meta.exists():
            try:
                source = json.load(open(meta)).get("source", "unknown")
            except Exception:
                pass
        if source == "unknown":
            warnings.warn(
                f"{out}/metadata.json missing or unreadable -- cannot confirm "
                "whether this data came from the real DSP layer. Delete the "
                "folder and regenerate before reporting any number.",
                RuntimeWarning, stacklevel=2,
            )
        return X, y, source

    if not force_fabricated and dsp_available():
        X, y = generate_from_dsp(str(out), samples_per_class, seed)
        return X, y, SOURCE_DSP_SYNTHETIC

    X, y = generate_fabricated(str(out), samples_per_class, seed)
    return X, y, SOURCE_FABRICATED


# Backwards-compatible shim for older phase scripts.
def generate_training_dataset(output_dir="training_data",
                              samples_per_class=500,
                              seed=RANDOM_STATE,
                              label_source=None):
    X, y, _ = load_dataset(samples_per_class=samples_per_class, seed=seed,
                           output_dir=output_dir, reuse_existing=False)
    return {"features": X, "labels": y}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", default="training_data")
    ap.add_argument("--samples_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--force-fabricated", action="store_true",
                    dest="force_fabricated",
                    help="use tier 2 even if the dsp package is available")
    args = ap.parse_args()

    print("dsp package importable:", dsp_available())
    X, y, source = load_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        output_dir=args.output_dir,
        force_fabricated=args.force_fabricated,
        reuse_existing=False,
    )
    print(f"source : {source}")
    print(f"wrote  : {X.shape[0]}x{X.shape[1]} -> {args.output_dir}/")
    print(f"labels : {np.bincount(y).tolist()}")
    if source == SOURCE_FABRICATED:
        print("\n  WARNING: fabricated data. Do not report these numbers.")
