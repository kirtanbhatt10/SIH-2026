"""Dataset diversity, synthetic shortcuts, signal-level and DSP audits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CLASS_ID,
    CLASS_NAMES,
    DIAGNOSTIC_DIR,
    FEATURE_NAMES,
    METADATA_CATEGORICAL,
    METADATA_NUMERIC,
)
from .data import AuditData


def _feature_key(row: np.ndarray, decimals: int = 8) -> tuple:
    return tuple(np.round(row, decimals))


def dataset_diversity_audit(data: AuditData) -> dict[str, Any]:
    X = data.dev_features()
    y = data.dev_labels()
    meta = data.dev_metadata()

    per_class: dict[str, Any] = {}
    for cname, cid in CLASS_ID.items():
        mask = y == cid
        Xc = X[mask]
        mc = meta.iloc[mask]

        groups = defaultdict(list)
        for i, idx in enumerate(np.where(mask)[0]):
            groups[_feature_key(X[idx])].append(int(idx))
        dup_rate = sum(len(v) - 1 for v in groups.values() if len(v) > 1) / max(int(mask.sum()), 1)

        nn = NearestNeighbors(n_neighbors=min(6, int(mask.sum())))
        nn.fit(Xc)
        dists, _ = nn.kneighbors(Xc)
        mean_nn = float(np.mean(dists[:, 1]))

        param_spread = {}
        for col in METADATA_NUMERIC:
            if col not in mc.columns:
                continue
            vals = pd.to_numeric(mc[col], errors="coerce").dropna()
            if len(vals) > 1:
                param_spread[col] = float(vals.std(ddof=0))

        per_class[cname] = {
            "n_samples": int(mask.sum()),
            "feature_std_mean": float(np.mean(np.std(Xc, axis=0, ddof=0))),
            "feature_std_max": float(np.max(np.std(Xc, axis=0, ddof=0))),
            "mean_nearest_neighbor_distance": mean_nn,
            "duplicate_feature_rate": float(dup_rate),
            "unique_generation_groups": int(mc["generation_group"].nunique()),
            "parameter_std": param_spread,
            "snr_std": param_spread.get("snr"),
            "frequency_std": param_spread.get("frequency"),
            "amplitude_std": param_spread.get("amplitude"),
        }

    return {
        "per_class": per_class,
        "notes": [
            "Duplicate feature rate counts exact 32-D vector duplicates within development.",
            "Low NN distance + low parameter spread => narrow class manifold.",
        ],
    }


def synthetic_shortcut_audit(data: AuditData) -> dict[str, Any]:
    meta = data.dev_metadata()
    y = data.dev_labels()
    shortcuts: list[dict[str, Any]] = []

    # Metadata-only class predictors (diagnostic trees)
    for col in METADATA_NUMERIC:
        if col not in meta.columns:
            continue
        vals = pd.to_numeric(meta[col], errors="coerce").to_numpy()
        mask = np.isfinite(vals)
        if mask.sum() < 50:
            continue
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(vals[mask].reshape(-1, 1), y[mask])
        acc = float(dt.score(vals[mask].reshape(-1, 1), y[mask]))
        if acc > 0.55:
            shortcuts.append(
                {
                    "feature_or_parameter": col,
                    "affected_classes": "all",
                    "evidence": f"Depth-3 decision tree on metadata alone achieves {acc:.3f} accuracy on dev.",
                    "why_unrealistic": "Real OTA classifiers should not rely on generation metadata.",
                    "severity": "HIGH" if acc > 0.75 else "MEDIUM",
                    "recommended_fix": "Ensure parameter ranges overlap across classes; avoid class-unique SNR/amplitude bands.",
                }
            )

    # Chirp uniqueness via tonal features
    X = data.dev_features()
    chirp_mask = y == CLASS_ID["chirp"]
    other_mask = ~chirp_mask
    chirp_centroid = X[chirp_mask, FEATURE_NAMES.index("spectral_centroid")]
    other_centroid = X[other_mask, FEATURE_NAMES.index("spectral_centroid")]
    if float(np.mean(chirp_centroid)) > float(np.percentile(other_centroid, 95)):
        shortcuts.append(
            {
                "feature_or_parameter": "spectral_centroid + chirp sweep metadata",
                "affected_classes": "chirp vs others",
                "evidence": "Chirp spectral centroid distribution sits above most non-chirp samples.",
                "why_unrealistic": "Chirp may be unrealistically separable via sweep energy alone.",
                "severity": "MEDIUM",
                "recommended_fix": "Add chirp-like benign interference and partial sweeps in benign class.",
            }
        )

    # Benign variant -> class shortcut
    if "benign_variant" in meta.columns:
        for variant in meta["benign_variant"].dropna().unique():
            n = int((meta["benign_variant"] == variant).sum())
            if n > 20:
                shortcuts.append(
                    {
                        "feature_or_parameter": f"benign_variant={variant}",
                        "affected_classes": "benign internal",
                        "evidence": f"{n} samples share variant label (internal diversity check).",
                        "why_unrealistic": "Not a classifier shortcut unless variants leak to other classes.",
                        "severity": "LOW",
                        "recommended_fix": "Monitor feature overlap across benign variants.",
                    }
                )

    # Frequency range shortcut for tone vs fsk
    fsk_freq = pd.to_numeric(meta.loc[y == CLASS_ID["fsk"], "frequency"], errors="coerce")
    tone_freq = pd.to_numeric(meta.loc[y == CLASS_ID["tone"], "frequency"], errors="coerce")
    overlap_frac = float(
        ((tone_freq >= fsk_freq.min()) & (tone_freq <= fsk_freq.max())).mean()
    ) if len(fsk_freq) and len(tone_freq) else 0.0
    if overlap_frac > 0.8:
        shortcuts.append(
            {
                "feature_or_parameter": "carrier frequency",
                "affected_classes": "fsk vs tone",
                "evidence": f"{overlap_frac:.1%} of tone carrier frequencies fall inside FSK mark range.",
                "why_unrealistic": "Frequency metadata alone cannot separate classes; feature overlap expected.",
                "severity": "INFO",
                "recommended_fix": "Improve modulation-sensitive features, not frequency metadata separation.",
            }
        )

    return {
        "shortcuts": shortcuts,
        "n_flagged": len([s for s in shortcuts if s["severity"] in {"HIGH", "MEDIUM"}]),
    }


def dsp_preprocessing_audit(data: AuditData) -> dict[str, Any]:
    X = data.dev_features()
    y = data.dev_labels()
    meta = data.dev_metadata()

    rms_idx = FEATURE_NAMES.index("rms_amplitude")
    ultra_idx = FEATURE_NAMES.index("ultrasonic_energy")
    squelch_mask = (X[:, rms_idx] < 1e-6) & (X[:, ultra_idx] < 1e-9)
    per_class_squelch = {
        CLASS_NAMES[int(c)]: int(squelch_mask[y == c].sum()) for c in np.unique(y)
    }

    # Benign high duty cycle overlap with OOK
    duty_idx = FEATURE_NAMES.index("duty_cycle")
    benign_duty = X[y == CLASS_ID["benign"], duty_idx]
    ook_duty = X[y == CLASS_ID["ook"], duty_idx]

    findings = [
        {
            "stage": "bandpass + peak normalize + squelch (AudioPreprocessor)",
            "finding": "Squelch floor 0.05 zeroes quiet chunks; collapsed vectors share fixed fingerprint.",
            "may_explain": "Low-SNR benign/OOK confusion when envelope statistics survive squelch differently.",
            "severity": "MEDIUM",
        },
        {
            "stage": "fixed 2048-sample FFT window (FeatureExtractor)",
            "finding": "Single ~42.7 ms window aggregates FSK symbol transitions and tone steady-state similarly.",
            "may_explain": "FSK -> Tone when deviation is small or tone is stable within window.",
            "severity": "HIGH",
        },
        {
            "stage": "temporal features (duty_cycle, bit_rate_estimate)",
            "finding": (
                f"Benign mean duty_cycle={float(benign_duty.mean()):.3f}, "
                f"OOK mean duty_cycle={float(ook_duty.mean()):.3f}."
            ),
            "may_explain": "OOK -> Benign and Benign -> OOK when ambient envelope mimics on/off stats.",
            "severity": "HIGH",
        },
        {
            "stage": "ultrasonic band energy features",
            "finding": "Energy ratio features compress amplitude differences after normalization.",
            "may_explain": "Amplitude/SNR diversity partially lost before classifier.",
            "severity": "MEDIUM",
        },
    ]

    return {
        "squelch_collapsed_by_class_dev": per_class_squelch,
        "benign_vs_ook_duty_cycle": {
            "benign_mean": float(benign_duty.mean()),
            "ook_mean": float(ook_duty.mean()),
            "overlap_bhattacharyya_proxy": float(
                min(benign_duty.max(), ook_duty.max()) - max(benign_duty.min(), ook_duty.min())
            )
            / max(benign_duty.max() - benign_duty.min(), 1e-9),
        },
        "findings": findings,
        "dsp_code_references": [
            "ai/dsp/audio_preprocessing.py — bandpass 17500-21500, squelch 0.05",
            "ai/dsp/feature_extraction.py — 32 features from single chunk",
            "ai/ml/dataset_v2/generator.py — crop/regen for squelch avoidance",
        ],
    }


def generation_parameter_analysis(data: AuditData, confusion: dict[str, Any]) -> dict[str, Any]:
    meta = data.dev_metadata()
    y = data.dev_labels()
    out: dict[str, Any] = {"pair_parameter_overlap": {}, "error_concentration": {}}

    pairs = [("fsk", "tone"), ("ook", "benign"), ("ook", "tone")]
    for a, b in pairs:
        ia, ib = CLASS_ID[a], CLASS_ID[b]
        overlap_report = {}
        for col in METADATA_NUMERIC:
            if col not in meta.columns:
                continue
            va = pd.to_numeric(meta.loc[y == ia, col], errors="coerce").dropna()
            vb = pd.to_numeric(meta.loc[y == ib, col], errors="coerce").dropna()
            if va.empty or vb.empty:
                continue
            lo = max(float(va.min()), float(vb.min()))
            hi = min(float(va.max()), float(vb.max()))
            overlap = max(0.0, hi - lo)
            span = max(float(va.max()), float(vb.max())) - min(float(va.min()), float(vb.min()))
            overlap_report[col] = {
                "range_a": [float(va.min()), float(va.max())],
                "range_b": [float(vb.min()), float(vb.max())],
                "overlap_fraction_of_union": float(overlap / max(span, 1e-9)),
            }
        out["pair_parameter_overlap"][f"{a}_vs_{b}"] = overlap_report

    for key, profile in confusion.get("pair_error_profiles", {}).items():
        if profile.get("count", 0) == 0:
            continue
        out["error_concentration"][key] = profile.get("vs_correct_same_class", {})

    return out


def signal_level_audit(*, seed: int = 42, samples_per_class: int = 20) -> dict[str, Any]:
    """Regenerate small diagnostic set in experiment dir — does not touch production data."""
    import sys

    diag_dir = DIAGNOSTIC_DIR
    diag_dir.mkdir(parents=True, exist_ok=True)

    _AI = Path(__file__).resolve().parents[3]
    if str(_AI) not in sys.path:
        sys.path.insert(0, str(_AI))

    from ml.dataset_v2.generator import generate_sample, make_shared_dsp_components, CLASS_NAMES as GEN_CLASSES

    preprocessor, extractor = make_shared_dsp_components()
    records = []
    for class_idx in sorted(GEN_CLASSES):
        for i in range(samples_per_class):
            sample = generate_sample(class_idx, i, seed, preprocessor, extractor)
            audio = sample.audio
            records.append(
                {
                    "class": GEN_CLASSES[class_idx],
                    "sample_id": sample.metadata["sample_id"],
                    "peak_amplitude": float(np.max(np.abs(audio))),
                    "rms": float(np.sqrt(np.mean(audio ** 2))),
                    "zero_fraction": float(np.mean(np.abs(audio) < 1e-6)),
                    "spectral_flatness": float(sample.features[FEATURE_NAMES.index("spectral_flatness")]),
                    "tonal_prominence": float(sample.features[FEATURE_NAMES.index("tonal_prominence")]),
                    "duty_cycle": float(sample.features[FEATURE_NAMES.index("duty_cycle")]),
                    "frequency": sample.metadata.get("frequency"),
                    "frequency_deviation": sample.metadata.get("frequency_deviation"),
                    "snr": sample.metadata.get("snr"),
                }
            )

    out_path = diag_dir / "diagnostic_signal_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    by_class: dict[str, list] = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)

    class_summaries = {}
    for cname, rows in by_class.items():
        class_summaries[cname] = {
            "n": len(rows),
            "rms_mean": float(np.mean([r["rms"] for r in rows])),
            "zero_fraction_mean": float(np.mean([r["zero_fraction"] for r in rows])),
            "spectral_flatness_mean": float(np.mean([r["spectral_flatness"] for r in rows])),
            "tonal_prominence_mean": float(np.mean([r["tonal_prominence"] for r in rows])),
        }

    realism_notes = []
    fsk = class_summaries.get("fsk", {})
    tone = class_summaries.get("tone", {})
    if fsk and tone:
        if abs(fsk.get("tonal_prominence_mean", 0) - tone.get("tonal_prominence_mean", 0)) < 0.05:
            realism_notes.append(
                "FSK and tone diagnostic batches have similar tonal_prominence — "
                "steady FSK windows resemble tones at feature level."
            )
    benign = class_summaries.get("benign", {})
    ook = class_summaries.get("ook", {})
    if benign and ook:
        if abs(benign.get("spectral_flatness_mean", 0) - ook.get("spectral_flatness_mean", 0)) < 0.02:
            realism_notes.append(
                "Benign and OOK diagnostic spectral_flatness overlap — "
                "noise-like benign can mimic gated carrier envelopes."
            )

    return {
        "diagnostic_dir": str(diag_dir),
        "samples_per_class": samples_per_class,
        "class_summaries": class_summaries,
        "realism_notes": realism_notes,
        "production_generator_modified": False,
    }


def realism_scorecard(
    diversity: dict[str, Any],
    overlap: dict[str, Any],
    shortcuts: dict[str, Any],
    dsp: dict[str, Any],
    separability: dict[str, Any],
) -> list[dict[str, str]]:
    def _diversity_status(cname: str) -> str:
        pc = diversity["per_class"].get(cname, {})
        if pc.get("duplicate_feature_rate", 0) > 0.05:
            return "WARN"
        if pc.get("snr_std", 0) and pc["snr_std"] < 2.0:
            return "WARN"
        return "PASS"

    def _sep_status(pair_key: str) -> str:
        p = overlap["pairs"].get(pair_key, {})
        if p.get("classes_overlapping_in_feature_space"):
            return "FAIL"
        return "PASS"

    high_shortcuts = shortcuts.get("n_flagged", 0)
    return [
        {"area": "Benign diversity", "status": _diversity_status("benign"), "evidence": "SNR/frequency/amplitude spread + duplicate rate"},
        {"area": "OOK diversity", "status": _diversity_status("ook"), "evidence": "Parameter std + NN distance"},
        {"area": "FSK diversity", "status": _diversity_status("fsk"), "evidence": "Frequency deviation spread"},
        {"area": "Tone diversity", "status": _diversity_status("tone"), "evidence": "Carrier frequency spread"},
        {"area": "FSK/Tone separation", "status": _sep_status("fsk_vs_tone"), "evidence": "Centroid distance + kNN purity"},
        {"area": "OOK/Benign separation", "status": _sep_status("ook_vs_benign"), "evidence": "Duty cycle + envelope overlap"},
        {"area": "OOK/Tone separation", "status": _sep_status("ook_vs_tone"), "evidence": "Pairwise Mahalanobis + overlap"},
        {"area": "DSP preservation", "status": "WARN", "evidence": "Single-window aggregation + squelch noted in dsp audit"},
        {"area": "Synthetic shortcuts", "status": "WARN" if high_shortcuts else "PASS", "evidence": f"{high_shortcuts} medium/high shortcuts flagged"},
        {"area": "Feature sufficiency", "status": "FAIL", "evidence": f"Top FSK/tone features: {separability.get('top_5_fsk_tone', [])[:3]}"},
        {"area": "Dataset realism", "status": "WARN", "evidence": "Classes overlap in 32-D space despite clean validation"},
    ]
