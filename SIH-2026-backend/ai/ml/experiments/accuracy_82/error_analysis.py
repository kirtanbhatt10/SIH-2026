"""Error analysis v2 on clean development data using the frozen baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.phase5.config import PHASE3_CLEAN_MODEL_PATH

from .config import (
    BASE_FEATURE_NAMES,
    BENIGN_CLASS,
    CLASS_NAMES,
    DOMINANT_CONFUSION_PAIRS,
    THREAT_CLASSES,
)
from .data import ExperimentData, load_clean_development_data
from .evaluation import evaluate_candidate, confusion_pairs


METADATA_CORRELATION_FIELDS = [
    "snr",
    "amplitude",
    "frequency",
    "frequency_deviation",
    "duty_cycle",
    "noise_level",
    "bit_rate",
    "peak_amplitude",
    "source_duration_sec",
    "crop_attempts",
    "benign_variant",
    "signal_type",
]

FEATURE_CORRELATION_FIELDS = [
    "harmonic_ratio",
    "spectral_flatness",
    "temporal_flatness",
    "amplitude_envelope_std",
    "duty_cycle",
    "tonal_prominence",
    "bit_rate_estimate",
    "spectral_centroid",
    "ultrasonic_energy",
]


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _compare_numeric(
    correct: pd.Series,
    incorrect: pd.Series,
) -> dict[str, Any]:
    c = _numeric_summary(correct)
    i = _numeric_summary(incorrect)
    if not c or not i:
        return {"correct": c, "incorrect": i, "delta_mean": None}
    return {
        "correct": c,
        "incorrect": i,
        "delta_mean": i["mean"] - c["mean"],
    }


def _pair_error_profile(
    meta: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    true_name: str,
    pred_name: str,
) -> dict[str, Any]:
    true_label = next(k for k, v in CLASS_NAMES.items() if v == true_name)
    pred_label = next(k for k, v in CLASS_NAMES.items() if v == pred_name)
    mask = (meta["y_true"] == true_label) & (meta["y_pred"] == pred_label)
    subset = meta.loc[mask]
    if subset.empty:
        return {"count": 0}

    feat_idx = {name: i for i, name in enumerate(feature_names)}
    profile: dict[str, Any] = {
        "count": int(mask.sum()),
        "metadata": {},
        "features": {},
    }
    for field in METADATA_CORRELATION_FIELDS:
        if field not in subset.columns:
            continue
        if subset[field].dtype == object or field in {"signal_type", "benign_variant"}:
            counts = subset[field].value_counts().to_dict()
            profile["metadata"][field] = {str(k): int(v) for k, v in counts.items()}
        else:
            profile["metadata"][field] = _numeric_summary(subset[field])

    for field in FEATURE_CORRELATION_FIELDS:
        if field not in feat_idx:
            continue
        values = features[subset.index.to_numpy(), feat_idx[field]]
        profile["features"][field] = _numeric_summary(pd.Series(values))

    return profile


def run_error_analysis_v2(
    *,
    baseline_model: Any | None = None,
    data: ExperimentData | None = None,
) -> dict[str, Any]:
    import joblib

    data = data or load_clean_development_data()
    model = baseline_model or joblib.load(PHASE3_CLEAN_MODEL_PATH)
    dev_idx = data.development_idx

    X = data.features[dev_idx]
    y = data.labels[dev_idx]
    meta = data.metadata.iloc[dev_idx].copy()
    y_pred = model.predict(X)

    metrics = evaluate_candidate(
        model, X, y, partition="clean_development", model_name="linear_svm_balanced"
    )
    cm = confusion_matrix(y, y_pred, labels=list(CLASS_NAMES.keys()))
    pairs = confusion_pairs(y, y_pred)

    meta["y_true"] = y
    meta["y_pred"] = y_pred
    meta["correct"] = y == y_pred
    meta["benign_false_alarm"] = (meta["y_true"] == BENIGN_CLASS) & (
        np.isin(meta["y_pred"], list(THREAT_CLASSES))
    )
    meta["threat_miss"] = np.isin(meta["y_true"], list(THREAT_CLASSES)) & (
        meta["y_pred"] == BENIGN_CLASS
    )

    correct_meta = meta[meta["correct"]]
    error_meta = meta[~meta["correct"]]

    metadata_correlates: dict[str, Any] = {}
    for field in METADATA_CORRELATION_FIELDS:
        if field not in meta.columns:
            continue
        if field in {"signal_type", "benign_variant"}:
            grouped = meta.groupby(field)["correct"].agg(["mean", "count"])
            metadata_correlates[field] = {
                str(k): {"accuracy": float(v["mean"]), "count": int(v["count"])}
                for k, v in grouped.iterrows()
            }
        else:
            metadata_correlates[field] = _compare_numeric(
                correct_meta[field], error_meta[field]
            )

    feature_correlates: dict[str, Any] = {}
    feat_idx = {name: i for i, name in enumerate(data.feature_names)}
    for field in FEATURE_CORRELATION_FIELDS:
        if field not in feat_idx:
            continue
        correct_vals = data.features[correct_meta.index.to_numpy(), feat_idx[field]]
        error_vals = data.features[error_meta.index.to_numpy(), feat_idx[field]]
        feature_correlates[field] = _compare_numeric(
            pd.Series(correct_vals), pd.Series(error_vals)
        )

    dominant_profiles = {}
    for true_name, pred_name in DOMINANT_CONFUSION_PAIRS:
        key = f"{true_name}_to_{pred_name}"
        dominant_profiles[key] = _pair_error_profile(
            meta, data.features, list(data.feature_names), true_name, pred_name
        )

    return {
        "analysis_scope": {
            "dataset": data.dataset_dir,
            "partition": "clean_development (model_train + cal_dev)",
            "locked_test_used": False,
            "baseline_model": str(PHASE3_CLEAN_MODEL_PATH),
            "n_samples": int(len(dev_idx)),
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[i] for i in CLASS_NAMES],
        "dominant_confusion_pairs": pairs[:15],
        "known_confusion_pair_profiles": dominant_profiles,
        "benign_false_positive_count": int(meta["benign_false_alarm"].sum()),
        "threat_false_negative_count": int(meta["threat_miss"].sum()),
        "metadata_error_correlates": metadata_correlates,
        "feature_error_correlates": feature_correlates,
        "notes": [
            "Analysis performed on frozen clean development data only.",
            "Dominant pairs: FSK->TONE, OOK->BENIGN, OOK->TONE, TONE->BENIGN.",
            "Features originate from the real 32-D DSP pipeline; no label leakage.",
        ],
    }


def markdown_error_analysis(report: dict[str, Any]) -> str:
    scope = report["analysis_scope"]
    lines = [
        "# Error Analysis v2",
        "",
        f"Dataset: `{scope['dataset']}`",
        f"Partition: `{scope['partition']}`",
        f"Samples: {scope['n_samples']}",
        "",
        "## Dominant confusion pairs",
        "",
        "| True | Predicted | Count | Rate within true |",
        "|------|-----------|-------|------------------|",
    ]
    for row in report["dominant_confusion_pairs"][:10]:
        lines.append(
            f"| {row['true_class']} | {row['predicted_class']} | {row['count']} | "
            f"{row['rate_within_true_class']:.3f} |"
        )

    lines.extend(["", "## Known pair profiles", ""])
    for key, profile in report["known_confusion_pair_profiles"].items():
        lines.append(f"### {key.replace('_', ' ')} (n={profile.get('count', 0)})")
        meta = profile.get("metadata", {})
        if "snr" in meta and isinstance(meta["snr"], dict) and "mean" in meta["snr"]:
            lines.append(f"- SNR mean: {meta['snr']['mean']:.2f} dB")
        if "frequency" in meta and isinstance(meta["frequency"], dict):
            lines.append(f"- Frequency mean: {meta['frequency']['mean']:.1f} Hz")
        if "frequency_deviation" in meta and isinstance(meta["frequency_deviation"], dict):
            if "mean" in meta["frequency_deviation"]:
                lines.append(
                    f"- Frequency deviation mean: {meta['frequency_deviation']['mean']:.1f} Hz"
                )
        if "duty_cycle" in meta and isinstance(meta["duty_cycle"], dict):
            if "mean" in meta["duty_cycle"]:
                lines.append(f"- Duty cycle mean: {meta['duty_cycle']['mean']:.3f}")
            else:
                lines.append(f"- Duty cycle distribution: {meta['duty_cycle']}")
        feat = profile.get("features", {})
        if "harmonic_ratio" in feat:
            lines.append(f"- Harmonic ratio mean: {feat['harmonic_ratio']['mean']:.4f}")
        if "spectral_flatness" in feat:
            lines.append(
                f"- Spectral flatness mean: {feat['spectral_flatness']['mean']:.4f}"
            )
        lines.append("")

    lines.extend([
        "## Metadata error correlates (incorrect vs correct)",
        "",
    ])
    for field, stats in report["metadata_error_correlates"].items():
        if isinstance(stats, dict) and stats.get("delta_mean") is not None:
            lines.append(f"- {field}: delta_mean={stats['delta_mean']:+.4f}")

    lines.extend([
        "",
        f"Benign false positives: {report['benign_false_positive_count']}",
        f"Threat false negatives: {report['threat_false_negative_count']}",
    ])
    return "\n".join(lines)
