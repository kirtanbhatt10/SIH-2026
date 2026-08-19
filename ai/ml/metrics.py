"""Shared metric helpers for the AI 2 ML subsystem.

Centralises:
  * per-class precision / recall / F1 (macro, weighted)
  * false-positive rate (FPR) — the metric the charter singles out
  * confusion matrix + which-classes-get-confused summary
  * detection latency (mean inference time per sample)
  * calibrated probability wrapper (sigmoid / isotonic)

These make the "no unmeasured accuracy claims" audit rule easy to honour:
every number a slide quotes should come from a function in here.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from config import CLASS_NAMES, THREAT_CLASSES


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             labels: Optional[list[int]] = None) -> dict[str, Any]:
    """Return a full metric bundle for hard predictions.

    ``labels`` must be the full set of class ids (defaults to sorted
    CLASS_NAMES).  FPR is computed per class (positive = that class) and
    also as the *threat* FPR (positive = any threat class, i.e. a benign
    sample wrongly flagged as a threat).
    """
    if labels is None:
        labels = sorted(CLASS_NAMES)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    tn, fp, fn, tp = cm.sum(axis=1), cm.sum(axis=0), None, None
    for i, lab in enumerate(labels):
        tp_ = cm[i, i]
        fp_ = cm[:, i].sum() - tp_
        fn_ = cm[i, :].sum() - tp_
        tn_ = cm.sum() - (tp_ + fp_ + fn_)
        fpr = fp_ / (fp_ + tn_) if (fp_ + tn_) else 0.0
        tpr = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        per_class[CLASS_NAMES.get(lab, str(lab))] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "fpr": float(fpr),
            "tpr": float(tpr),
            "tp": int(tp_), "fp": int(fp_), "fn": int(fn_), "tn": int(tn_),
        }

    # Threat-level FPR: benign sample promoted to any threat class.
    benign_mask = y_true == 0
    threat_pred = np.isin(y_pred, list(THREAT_CLASSES))
    n_benign = int(benign_mask.sum())
    n_benign_as_threat = int((benign_mask & threat_pred).sum())
    threat_fpr = n_benign_as_threat / n_benign if n_benign else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in per_class])),
        "weighted_f1": float(np.average(
            [per_class[c]["f1"] for c in per_class],
            weights=[per_class[c]["support"] for c in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm,
        "threat_fpr": float(threat_fpr),
        "n_benign": int(n_benign),
        "n_benign_as_threat": int(n_benign_as_threat),
    }


def calibration_curve(y_true, probs, n_bins=10):
    """Return (mean_pred, frac_pos) bins for plotting calibration."""
    order = np.argsort(probs)
    probs_s = probs[order]
    y_s = y_true[order]
    edges = np.linspace(0, len(probs_s), n_bins + 1).astype(int)
    mean_pred, frac_pos = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            mean_pred.append(probs_s[lo:hi].mean())
            frac_pos.append(y_s[lo:hi].mean())
    return np.array(mean_pred), np.array(frac_pos)


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: mean |predicted prob - empirical freq|, weighted by bin size."""
    order = np.argsort(probs)
    probs_s = probs[order]
    y_s = y_true[order]
    edges = np.linspace(0, len(probs_s), n_bins + 1).astype(int)
    n = len(probs_s)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            conf = probs_s[lo:hi].mean()
            acc = y_s[lo:hi].mean()
            ece += (hi - lo) / n * abs(acc - conf)
    return float(ece)


def fit_calibrated(clf, X_tr, y_tr, X_te, y_te, method="sigmoid", cv=3):
    """Fit a calibrated classifier and return eval + ECE on the test set.

    ``clf`` should already be trained on X_tr/y_tr.  Returns the fitted
    CalibratedClassifierCV (with the original estimator as base), the
    calibrated test probabilities, and calibration diagnostics.
    """
    cal = CalibratedClassifierCV(clf, method=method, cv=cv)
    cal.fit(X_tr, y_tr)
    t0 = time.perf_counter()
    proba = cal.predict_proba(X_te)
    latency_ms = (time.perf_counter() - t0) / max(1, len(X_te)) * 1000.0
    # ECE on P(benign) correctness, i.e. is predicted prob of benign accurate?
    benign_col = list(cal.classes_).index(0) if 0 in cal.classes_ else 0
    p_benign = proba[:, benign_col]
    ece = expected_calibration_error(1 - y_te.astype(bool).astype(int), p_benign)
    return cal, proba, {"ece": ece, "latency_ms": latency_ms, "method": method}


def measure_latency(clf, X, n_repeat=50):
    """Mean + std inference time (ms) per sample on the fitted clf."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        clf.predict(X)
        times.append((time.perf_counter() - t0) / max(1, len(X)) * 1000.0)
    return {"latency_ms_mean": float(np.mean(times)),
            "latency_ms_std": float(np.std(times))}


def pretty_report(y_true, y_pred, labels=None, title=""):
    s = classification_report(
        y_true, y_pred, labels=labels or sorted(CLASS_NAMES),
        target_names=[CLASS_NAMES[i] for i in (labels or sorted(CLASS_NAMES))],
        zero_division=0)
    return (f"--- {title} ---\n" + s if title else s)
