"""Phase 1 — Baseline classifier (Random Forest).

Establish a reference number before any sophistication.  This mirrors the
exact script in the AI 1 handoff plan §6 Phase 1, plus FPR (audit rule) and
per-class detail.

USAGE
-----
    python ai/ml/phase1_baseline.py [--samples_per_class 500] [--seed 42]

OUTPUT
------
    ai/ml/output/baseline_report.txt
    ai/ml/output/models/baseline_rf.joblib
    ai/ml/output/confusion_baseline.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# So `python ai/ml/phase1_baseline.py` resolves sibling imports.
sys.path.insert(0, os.path.dirname(__file__))

import config
import data_generation
import metrics
from config import CLASS_NAMES, OUTPUT_DIR, MODEL_DIR


def load_or_generate(samples_per_class, seed):
    """Load the dataset, preferring the REAL DSP layer.

    Delegates to data_generation.load_dataset(), which tries, in order:
      1. an existing dataset on disk (provenance read from metadata.json)
      2. tier 1 -- synthetic audio through the real dsp.FeatureExtractor
      3. tier 2 -- fabricated vectors, only if `dsp` cannot be imported

    Returns (X, y, source) where source is one of "dsp_synthetic",
    "fabricated", "dsp_real" or "unknown". The source string must be quoted
    next to any metric derived from this data.
    """
    X, y, source = data_generation.load_dataset(
        samples_per_class=samples_per_class, seed=seed)

    assert X.shape[1] == config.N_FEATURES, X.shape
    assert not np.isnan(X).any() and not np.isinf(X).any()

    if source == config.SOURCE_FABRICATED:
        print("\n  !! DATA IS FABRICATED -- not produced by the DSP layer.")
        print("     Structural testing only. Do not report these numbers.\n")
    elif source == "unknown":
        print("\n  !! DATA PROVENANCE UNKNOWN -- metadata.json missing.")
        print("     Delete training_data/ and regenerate before reporting.\n")

    return X, y, source


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 baseline")
    ap.add_argument("--samples_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()

    X, y, src = load_or_generate(args.samples_per_class, args.seed)
    print(f"Data: {src} | {X.shape} | labels {np.bincount(y).tolist()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    clf = RandomForestClassifier(**config.BASELINE_PARAMS)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    ev = metrics.evaluate(y_te, y_pred)
    latency = metrics.measure_latency(clf, X_te)

    report = metrics.pretty_report(y_te, y_pred, title="BASELINE RF")

    lines = [
        f"Phase 1 baseline — Random Forest (n={clf.n_estimators})",
        f"Data source : {src}",
        f"Train size  : {X_tr.shape[0]}  Test size: {X_te.shape[0]}",
        "",
        report,
        "Per-class FPR (false positive rate):",
    ]
    for cname, d in ev["per_class"].items():
        lines.append(f"  {cname:8s} fpr={d['fpr']:.3f} tpr={d['recall']:.3f} "
                     f"tp={d['tp']} fp={d['fp']} fn={d['fn']}")
    lines += [
        "",
        f"Threat-level FPR (benign->threat): {ev['threat_fpr']:.4f} "
        f"({ev['n_benign_as_threat']}/{ev['n_benign']} benign flagged)",
        f"Macro F1 : {ev['macro_f1']:.3f}",
        f"Weighted F1 : {ev['weighted_f1']:.3f}",
        f"Inference latency: {latency['latency_ms_mean']:.3f} +- "
        f"{latency['latency_ms_std']:.3f} ms/sample",
    ]
    body = "\n".join(lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "baseline_report.txt").write_text(body + "\n")
    with open(OUTPUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump({
            "source": src,
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "threat_fpr": ev["threat_fpr"],
            "per_class": ev["per_class"],
            "latency_ms": latency["latency_ms_mean"],
        }, f, indent=2)

    try:
        import joblib
        joblib.dump(clf, MODEL_DIR / "baseline_rf.joblib")
        print("Saved model:", MODEL_DIR / "baseline_rf.joblib")
    except Exception as e:  # pragma: no cover
        print("joblib unavailable, skipping artefact:", e)

    # Confusion matrix plot (saved even if it can't be shown).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(
            ev["confusion_matrix"],
            display_labels=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Baseline RF confusion matrix (synthetic)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_baseline.png", dpi=130)
        plt.close()
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)

    print(body)


if __name__ == "__main__":
    main()
