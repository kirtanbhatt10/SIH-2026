"""Phase 2 — Model comparison (RF vs SVM vs Logistic Regression).

Selection is by *measured* cross-validated result, not preference.  Judges
will ask "why this model?" — this script produces the table that answers it,
plus the feature-importance ranking (audit: report what matters, and say so
if a feature contributes nothing).

WHY THE SCALED PIPELINES MATTER
-------------------------------
The first version of this script fed raw features straight into SVC and
LogisticRegression.  On fabricated data (all features roughly 0-1) that looked
fine and the three models tied.  On REAL dsp features it is badly misleading:

    model                 raw features    with StandardScaler
    RandomForest              0.837            0.837
    LogisticRegression        0.687             ~0.80
    SVM_rbf                   0.463             ~0.83

Real features span very different ranges — energy_db is negative,
peak_frequency is in the thousands, spectral_flatness is 0-1.  RBF SVM uses
Euclidean distance and LogisticRegression uses a shared regularisation
strength, so both are dominated by whichever feature has the largest
magnitude.  Random Forest splits per feature and is scale-invariant, which is
why it was unaffected.

Comparing an unscaled SVM against a Random Forest is not a fair comparison,
and "SVM performs poorly" would have been a wrong finding about the model
rather than a true statement about the preprocessing.  Both scale-sensitive
models are therefore wrapped in a Pipeline with StandardScaler; the scaler is
fitted inside each CV fold, so no test-fold statistics leak into training.

USAGE
-----
    python ai/ml/phase2_model_comparison.py [--samples_per_class 500]
                                           [--cv 5]

OUTPUT
------
    ai/ml/output/model_comparison.txt / .csv
    ai/ml/output/feature_importance.txt / .png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config
import metrics
from config import CLASS_NAMES, FEATURE_NAMES, OUTPUT_DIR
from data_generation import load_dataset


def _build_models(rs: int) -> dict:
    """Model factory. Scale-sensitive learners get a StandardScaler."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        # Tree ensemble: scale-invariant, no scaler needed.
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=rs),

        # RBF kernel is distance-based -> MUST be scaled.
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=rs)),
        ]),

        # Regularisation is shared across coefficients -> MUST be scaled.
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=rs)),
        ]),

        # Kept unscaled on purpose: demonstrates the size of the effect and
        # justifies the scaler in the report.
        "SVM_rbf_UNSCALED": SVC(kernel="rbf", C=1.0, gamma="scale",
                                probability=True, random_state=rs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 model comparison")
    ap.add_argument("--samples_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    X, y, src = load_dataset(args.samples_per_class, args.seed)

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    models = _build_models(args.seed)
    rows = []
    print(f"Data source: {src} | {X.shape} | comparing {len(models)} models "
          f"({args.cv}-fold CV)")
    for name, clf in models.items():
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        rows.append({
            "model": name,
            "scaled": "no" if name in ("RandomForest", "SVM_rbf_UNSCALED") else "yes",
            "cv_acc": round(float(acc.mean()), 4),
            "cv_acc_std": round(float(acc.std()), 4),
            "cv_f1_macro": round(float(f1.mean()), 4),
        })
        print(f"  {name:20s} acc={acc.mean():.4f}+-{acc.std():.4f} "
              f"f1_macro={f1.mean():.4f}")

    # Diagnostic row is excluded from selection.
    ranked = sorted([r for r in rows if r["model"] != "SVM_rbf_UNSCALED"],
                    key=lambda r: -r["cv_acc"])
    best = ranked[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = ["Phase 2 - model comparison",
           f"Data source: {src}",
           f"CV: {args.cv}-fold stratified, seed={args.seed}",
           ""]
    header = (f"{'model':22s} {'scaled':>6s} {'cv_acc':>8s} {'std':>7s} "
              f"{'f1_macro':>9s}")
    txt.append(header)
    txt.append("-" * len(header))
    for r in sorted(rows, key=lambda r: -r["cv_acc"]):
        txt.append(f"{r['model']:22s} {r['scaled']:>6s} {r['cv_acc']:8.4f} "
                   f"{r['cv_acc_std']:7.4f} {r['cv_f1_macro']:9.4f}")

    unscaled = next((r for r in rows if r["model"] == "SVM_rbf_UNSCALED"), None)
    scaled_svm = next((r for r in rows if r["model"] == "SVM_rbf"), None)

    txt += ["", f"Selected: {best['model']} "
                f"(cv_acc {best['cv_acc']:.4f} +- {best['cv_acc_std']:.4f})"]

    if unscaled and scaled_svm:
        delta = scaled_svm["cv_acc"] - unscaled["cv_acc"]
        txt += ["",
                "Scaling effect (why the pipelines are there):",
                f"  SVM_rbf unscaled : {unscaled['cv_acc']:.4f}",
                f"  SVM_rbf scaled   : {scaled_svm['cv_acc']:.4f}"
                f"   ({delta:+.4f})",
                "  Real features have very different ranges (energy_db is",
                "  negative, peak_frequency is in the thousands). RBF SVM is",
                "  distance-based, so without StandardScaler the largest-",
                "  magnitude feature dominates. Random Forest splits per",
                "  feature and is unaffected."]

    txt += ["",
            "Selection justification:",
            "  Random Forest is recommended for production even where another",
            "  model edges it on CV accuracy: native feature importances (a",
            "  required deliverable), no scaling dependency at serve time,",
            "  robust to mixed feature scales, explainable to judges, trains",
            "  in seconds.",
            "",
            f"PROVENANCE: source='{src}'.",
            "  dsp_synthetic = synthetic audio through the real DSP layer.",
            "  fabricated    = invented vectors; NOT a measurement.",
            "  dsp_real      = real over-the-air recordings.",
            "Quote the source next to any number taken from this table."]

    (OUTPUT_DIR / "model_comparison.txt").write_text("\n".join(txt) + "\n")

    with open(OUTPUT_DIR / "model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Feature importance (Random Forest) ------------------------------ #
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(**config.BASELINE_PARAMS)
    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]

    f_lines = [f"Phase 2 - feature importance (RandomForest, source={src})", ""]
    f_lines.append(f"{'rank':>4s} {'feature':<26s} {'importance':>10s}")
    f_lines.append("-" * 44)
    for i, idx in enumerate(order, 1):
        f_lines.append(f"{i:4d} {FEATURE_NAMES[idx]:<26s} {imp[idx]:10.4f}")

    dead = [FEATURE_NAMES[i] for i in order if imp[i] < 0.005]
    if dead:
        f_lines += ["", "Contributing < 0.005 (report these - a feature that "
                        "does nothing is a real finding):",
                    "  " + ", ".join(dead)]
    (OUTPUT_DIR / "feature_importance.txt").write_text("\n".join(f_lines) + "\n")
    print("\n".join(f_lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 8))
        plt.barh([FEATURE_NAMES[i] for i in order][::-1],
                 [imp[i] for i in order][::-1], color="#4C72B0")
        plt.xlabel("importance")
        plt.title(f"Feature importance - RandomForest ({src})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=130)
        plt.close()
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)

    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
