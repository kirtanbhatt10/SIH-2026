"""Produce dataset_root_cause_report.json for old and new datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml.phase3.data import grouped_train_test_split, load_dataset_v2


def _feature_key(row: np.ndarray) -> tuple:
    return tuple(np.round(row, 8))


def analyze_dataset(dataset_dir: Path | None = None) -> dict:
    dataset = load_dataset_v2(dataset_dir)
    X, meta, y = dataset.features, dataset.metadata, dataset.labels
    split = grouped_train_test_split(meta, y, seed=42)

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(X)):
        groups[_feature_key(X[i])].append(i)

    dup_groups = [
        {
            "count": len(idxs),
            "sample_ids": meta.iloc[idxs]["sample_id"].tolist(),
            "labels": [int(y[i]) for i in idxs],
            "generation_groups": sorted(set(meta.iloc[idxs]["generation_group"])),
            "benign_variants": sorted(
                set(meta.iloc[idxs]["benign_variant"].dropna().astype(str))
            ),
            "peak_amplitudes": [float(meta.iloc[i]["peak_amplitude"]) for i in idxs[:5]],
        }
        for idxs in groups.values()
        if len(idxs) > 1
    ]
    dup_groups.sort(key=lambda g: g["count"], reverse=True)

    cross = []
    train_map = {_feature_key(X[i]): int(i) for i in split.train_idx}
    for ti in split.test_idx:
        key = _feature_key(X[ti])
        if key in train_map:
            tr = train_map[key]
            cross.append(
                {
                    "test_sample_id": meta.iloc[ti]["sample_id"],
                    "train_sample_id": meta.iloc[tr]["sample_id"],
                    "label": int(y[ti]),
                    "test_group": meta.iloc[ti]["generation_group"],
                    "train_group": meta.iloc[tr]["generation_group"],
                    "test_variant": meta.iloc[ti].get("benign_variant"),
                    "train_variant": meta.iloc[tr].get("benign_variant"),
                }
            )

    squelch = (X[:, 20] < 1e-6) & (X[:, 0] < 1e-9)
    per_class_squelch = {int(c): int(squelch[y == c].sum()) for c in np.unique(y)}

    mask_b, mask_o = y == 0, y == 2
    benign_ook = {
        "benign_duty_cycle_mean": float(X[mask_b, 24].mean()),
        "ook_duty_cycle_mean": float(X[mask_o, 24].mean()),
        "benign_bit_rate_mean": float(X[mask_b, 25].mean()),
        "ook_bit_rate_mean": float(X[mask_o, 25].mean()),
        "benign_tonal_prominence_mean": float(X[mask_b, 16].mean()),
        "ook_tonal_prominence_mean": float(X[mask_o, 16].mean()),
    }

    root_causes = []
    if dup_groups and dup_groups[0]["count"] >= 10:
        root_causes.append(
            {
                "category": "B",
                "name": "different_audio_collapsing_to_identical_features",
                "detail": (
                    "Multiple benign variants/groups share identical 32-D vectors "
                    "after bandpass squelch (peak < 0.05 -> zeros -> fixed fingerprint)."
                ),
            }
        )
    if cross:
        root_causes.append(
            {
                "category": "B/E",
                "name": "cross_split_exact_feature_leakage",
                "detail": (
                    "Grouped split prevents generation_group overlap but identical "
                    "collapsed feature vectors still appear in train and test."
                ),
            }
        )
    if benign_ook["benign_duty_cycle_mean"] > 0.75:
        root_causes.append(
            {
                "category": "F",
                "name": "benign_envelope_mimics_ook",
                "detail": (
                    "Benign ambient envelope features yield high duty_cycle (~0.82), "
                    "overlapping OOK modulation statistics and causing benign->ook confusion."
                ),
            }
        )

    return {
        "dataset_version": dataset.dataset_info.get("dataset_version"),
        "generator_version": dataset.dataset_info.get("generator_version"),
        "n_samples": dataset.n_samples,
        "duplicate_groups": dup_groups,
        "n_duplicate_groups": len(dup_groups),
        "cross_split_exact_matches": cross,
        "n_cross_split_exact_matches": len(cross),
        "collapsed_squelch_by_class": per_class_squelch,
        "benign_vs_ook_feature_separation": benign_ook,
        "chirp_perfect_separability_note": (
            "Chirp sweeps produce unique start/end/sweep feature combinations; "
            "near-perfect separability is expected on synthetic chirp class."
        ),
        "root_causes": root_causes,
        "recommended_generator_fixes": [
            "Reject/regenerate collapsed squelch fingerprints",
            "Add continuous in-band benign variants (steady_ultrasonic_hum, band_limited_hiss)",
            "Do not treat squelched ambient as legitimate benign diversity",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/investigation/dataset_root_cause_report.json"),
    )
    args = parser.parse_args()
    report = analyze_dataset(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"root_causes": report["root_causes"]}, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
