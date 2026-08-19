"""
Dataset V2 reproducibility test (Task 20).

1. Generate twice with the same seed/config -> verify identical.
2. Generate with a different seed -> verify meaningfully different.

Run:
    python test_reproducibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AI_DIR = _HERE.parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

try:
    from .build_dataset import build_dataset
except ImportError:
    from build_dataset import build_dataset  # type: ignore


def test_same_seed():
    """Same seed + same config -> identical features, labels, metadata."""
    dir_a = Path("/tmp/dsv2_repro_a")
    dir_b = Path("/tmp/dsv2_repro_b")

    for d in (dir_a, dir_b):
        if d.exists():
            import shutil
            shutil.rmtree(d)

    print("  Run A (seed=42, 10/class) ...")
    build_dataset(dir_a, samples_per_class=10, seed=42, save_audio=False)
    print("  Run B (seed=42, 10/class) ...")
    build_dataset(dir_b, samples_per_class=10, seed=42, save_audio=False)

    Xa, ya = np.load(dir_a / "features.npy"), np.load(dir_a / "labels.npy")
    Xb, yb = np.load(dir_b / "features.npy"), np.load(dir_b / "labels.npy")

    assert np.array_equal(ya, yb), "labels differ"
    assert np.array_equal(Xa, Xb), "feature values differ"

    import csv
    with open(dir_a / "metadata.csv") as f:
        meta_a = list(csv.DictReader(f))
    with open(dir_b / "metadata.csv") as f:
        meta_b = list(csv.DictReader(f))
    assert len(meta_a) == len(meta_b)
    for ra, rb in zip(meta_a, meta_b):
        assert ra == rb, f"metadata row differs: {ra} vs {rb}"

    print("  [PASS] same seed -> identical dataset")

    # cleanup
    import shutil
    shutil.rmtree(dir_a)
    shutil.rmtree(dir_b)


def test_different_seed():
    """Different seed -> meaningfully different features."""
    dir_a = Path("/tmp/dsv2_repro_s1")
    dir_b = Path("/tmp/dsv2_repro_s2")

    for d in (dir_a, dir_b):
        if d.exists():
            import shutil
            shutil.rmtree(d)

    print("  Run C (seed=42, 10/class) ...")
    build_dataset(dir_a, samples_per_class=10, seed=42, save_audio=False)
    print("  Run D (seed=99, 10/class) ...")
    build_dataset(dir_b, samples_per_class=10, seed=99, save_audio=False)

    Xa = np.load(dir_a / "features.npy")
    Xb = np.load(dir_b / "features.npy")

    assert not np.array_equal(Xa, Xb), "features unexpectedly identical with different seeds"

    # Check that the per-class mean differs meaningfully (not just noise-level)
    ya = np.load(dir_a / "labels.npy")
    yb = np.load(dir_b / "labels.npy")
    for c in range(5):
        ma = Xa[ya == c].mean(axis=0)
        mb = Xb[yb == c].mean(axis=0)
        diff = np.abs(ma - mb)
        assert diff.max() > 1e-3, f"class {c}: means suspiciously similar across seeds"

    print("  [PASS] different seed -> meaningfully different dataset")

    import shutil
    shutil.rmtree(dir_a)
    shutil.rmtree(dir_b)


if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset V2 Reproducibility Tests")
    print("=" * 60)

    test_same_seed()
    test_different_seed()

    print("-" * 60)
    print("  ALL REPRODUCIBILITY TESTS PASSED")
