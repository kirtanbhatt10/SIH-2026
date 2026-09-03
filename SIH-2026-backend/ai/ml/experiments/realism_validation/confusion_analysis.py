"""Confusion boundary analysis using frozen baseline on development data."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    CLASS_ID,
    CLASS_NAMES,
    CONFUSION_PAIRS,
    FROZEN_MODEL_PATH,
    HISTORICAL_DEV_ERRORS,
    METADATA_NUMERIC,
)
from .data import AuditData


def _pair_key(true_name: str, pred_name: str) -> str:
    return f"{true_name}_to_{pred_name}"


def confusion_boundary_analysis(data: AuditData) -> dict[str, Any]:
    model = joblib.load(FROZEN_MODEL_PATH)
    X = data.dev_features()
    y = data.dev_labels()
    meta = data.dev_metadata()
    y_pred = model.predict(X)

    pair_counts: dict[str, int] = {}
    for true_i, true_name in CLASS_NAMES.items():
        for pred_i, pred_name in CLASS_NAMES.items():
            if true_i == pred_i:
                continue
            count = int(((y == true_i) & (y_pred == pred_i)).sum())
            if count:
                pair_counts[_pair_key(true_name, pred_name)] = count

    verified = {}
    for true_name, pred_name in CONFUSION_PAIRS:
        key = _pair_key(true_name, pred_name)
        verified[key] = {
            "count": pair_counts.get(key, 0),
            "historical_count": HISTORICAL_DEV_ERRORS.get(key),
            "match_historical": pair_counts.get(key, 0) == HISTORICAL_DEV_ERRORS.get(key),
        }

    err_mask = y != y_pred
    err_meta = meta.loc[err_mask].copy()
    err_meta["y_true"] = y[err_mask]
    err_meta["y_pred"] = y_pred[err_mask]

    pair_profiles: dict[str, Any] = {}
    for true_name, pred_name in CONFUSION_PAIRS:
        key = _pair_key(true_name, pred_name)
        ti, pi = CLASS_ID[true_name], CLASS_ID[pred_name]
        sub = err_meta[(err_meta["y_true"] == ti) & (err_meta["y_pred"] == pi)]
        if sub.empty:
            pair_profiles[key] = {"count": 0}
            continue
        correct_same_true = meta.loc[(y == ti) & ~err_mask]
        profile = {"count": int(len(sub)), "metadata": {}, "vs_correct_same_class": {}}
        for col in METADATA_NUMERIC:
            if col not in sub.columns:
                continue
            ev = pd.to_numeric(sub[col], errors="coerce").dropna()
            cv = pd.to_numeric(correct_same_true[col], errors="coerce").dropna()
            if ev.empty:
                continue
            profile["metadata"][col] = {
                "error_mean": float(ev.mean()),
                "error_median": float(ev.median()),
            }
            if not cv.empty:
                profile["vs_correct_same_class"][col] = {
                    "correct_mean": float(cv.mean()),
                    "delta_mean": float(ev.mean() - cv.mean()),
                }
        pair_profiles[key] = profile

    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "model": str(FROZEN_MODEL_PATH),
        "partition": "clean_development",
        "locked_test_used": False,
        "total_errors": int(err_mask.sum()),
        "total_samples": int(len(y)),
        "error_rate": float(err_mask.mean()),
        "pair_counts": pair_counts,
        "top_confusion_pairs": [{"pair": k, "count": v} for k, v in top_pairs],
        "verified_known_pairs": verified,
        "pair_error_profiles": pair_profiles,
    }
