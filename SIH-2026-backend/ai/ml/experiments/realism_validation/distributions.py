"""Per-class feature and parameter distribution statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, FEATURE_NAMES, METADATA_NUMERIC
from .data import AuditData


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    qs = [5, 25, 50, 75, 95]
    vals = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)}


def feature_distributions(data: AuditData) -> dict[str, Any]:
    X = data.dev_features()
    y = data.dev_labels()
    out: dict[str, Any] = {"features": {}, "summary": {}}

    for fi, fname in enumerate(FEATURE_NAMES):
        per_class = {}
        for cid, cname in CLASS_NAMES.items():
            mask = y == cid
            vals = X[mask, fi]
            finite = vals[np.isfinite(vals)]
            per_class[cname] = {
                "count": int(mask.sum()),
                "mean": float(np.mean(finite)) if len(finite) else None,
                "median": float(np.median(finite)) if len(finite) else None,
                "std": float(np.std(finite, ddof=0)) if len(finite) else None,
                "min": float(np.min(finite)) if len(finite) else None,
                "max": float(np.max(finite)) if len(finite) else None,
                "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25))
                if len(finite)
                else None,
                "percentiles": _percentiles(finite) if len(finite) else {},
                "nan_count": int(np.isnan(vals).sum()),
                "inf_count": int(np.isinf(vals).sum()),
            }
        out["features"][fname] = per_class

    out["summary"]["global_nan"] = int(np.isnan(X).sum())
    out["summary"]["global_inf"] = int(np.isinf(X).sum())
    return out


def parameter_distributions(data: AuditData) -> dict[str, Any]:
    meta = data.dev_metadata()
    out: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for col in METADATA_NUMERIC:
        if col not in meta.columns:
            continue
        per_class = {}
        for cname in CLASS_NAMES.values():
            sub = pd.to_numeric(meta.loc[meta["signal_type"] == cname, col], errors="coerce")
            sub = sub.dropna()
            if sub.empty and cname != "benign":
                sub = pd.to_numeric(meta.loc[data.dev_labels() == {"benign": 0, "fsk": 1, "ook": 2, "chirp": 3, "tone": 4}[cname], col], errors="coerce").dropna()
            if sub.empty:
                per_class[cname] = {"count": 0}
                continue
            per_class[cname] = {
                "count": int(len(sub)),
                "mean": float(sub.mean()),
                "median": float(sub.median()),
                "std": float(sub.std(ddof=0)),
                "min": float(sub.min()),
                "max": float(sub.max()),
                "iqr": float(sub.quantile(0.75) - sub.quantile(0.25)),
                "percentiles": _percentiles(sub.to_numpy()),
            }
        out["numeric"][col] = per_class

    for col in ["noise_type", "benign_variant"]:
        if col not in meta.columns:
            continue
        per_class = {}
        for cname in CLASS_NAMES.values():
            sub = meta.loc[meta["signal_type"] == cname, col].dropna().astype(str)
            per_class[cname] = sub.value_counts().to_dict()
        out["categorical"][col] = {k: {str(a): int(b) for a, b in v.items()} for k, v in per_class.items()}

    return out
