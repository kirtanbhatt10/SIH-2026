"""Investigate exact duplicates and near-duplicates across Dataset V2 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.phase3.data import grouped_train_test_split, load_dataset_v2


def _exact_duplicate_groups(X: np.ndarray, metadata: pd.DataFrame) -> list[dict]:
    rows = []
    seen: dict[tuple[float, ...], list[int]] = {}
    for i in range(X.shape[0]):
        key = tuple(np.round(X[i], 8))
        seen.setdefault(key, []).append(i)
    for key, indices in seen.items():
        if len(indices) > 1:
            rows.append(
                {
                    "count": len(indices),
                    "sample_ids": metadata.iloc[indices]["sample_id"].tolist(),
                    "labels": metadata.iloc[indices]["label"].tolist(),
                    "generation_groups": metadata.iloc[indices]["generation_group"].unique().tolist(),
                }
            )
    return sorted(rows, key=lambda r: r["count"], reverse=True)


def _cross_split_exact_matches(
    X: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> list[dict]:
    train_map: dict[tuple[float, ...], int] = {}
    for idx in train_idx:
        train_map[tuple(np.round(X[idx], 8))] = int(idx)

    matches = []
    for idx in test_idx:
        key = tuple(np.round(X[idx], 8))
        if key in train_map:
            matches.append(
                {
                    "test_sample_id": metadata.iloc[idx]["sample_id"],
                    "train_sample_id": metadata.iloc[train_map[key]]["sample_id"],
                    "label": int(metadata.iloc[idx]["label"]),
                    "test_group": metadata.iloc[idx]["generation_group"],
                    "train_group": metadata.iloc[train_map[key]]["generation_group"],
                }
            )
    return matches


def _near_duplicate_pairs(
    X: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    threshold: float = 0.999,
    max_pairs: int = 20,
) -> list[dict]:
    pairs = []
    for t_idx in test_idx:
        t_vec = X[t_idx]
        t_norm = np.linalg.norm(t_vec)
        if t_norm == 0:
            continue
        for tr_idx in train_idx:
            tr_vec = X[tr_idx]
            denom = t_norm * np.linalg.norm(tr_vec)
            if denom == 0:
                continue
            sim = float(np.dot(t_vec, tr_vec) / denom)
            if sim >= threshold:
                pairs.append(
                    {
                        "similarity": sim,
                        "test_sample_id": metadata.iloc[t_idx]["sample_id"],
                        "train_sample_id": metadata.iloc[tr_idx]["sample_id"],
                        "test_label": int(metadata.iloc[t_idx]["label"]),
                        "train_label": int(metadata.iloc[tr_idx]["label"]),
                        "test_group": metadata.iloc[t_idx]["generation_group"],
                        "train_group": metadata.iloc[tr_idx]["generation_group"],
                    }
                )
    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs[:max_pairs]


def run_audit(*, output_path: Path | None = None) -> dict:
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)

    exact_groups = _exact_duplicate_groups(dataset.features, dataset.metadata)
    cross_exact = _cross_split_exact_matches(
        dataset.features, dataset.metadata, split.train_idx, split.test_idx
    )
    near_pairs = _near_duplicate_pairs(
        dataset.features,
        dataset.metadata,
        split.train_idx,
        split.test_idx,
    )

    report = {
        "dataset_version": dataset.dataset_info.get("dataset_version", "unknown"),
        "generator_version": dataset.dataset_info.get("generator_version", "unknown"),
        "n_samples": dataset.n_samples,
        "split_seed": 42,
        "train_size": int(len(split.train_idx)),
        "test_size": int(len(split.test_idx)),
        "exact_duplicate_groups": exact_groups,
        "n_exact_duplicate_groups": len(exact_groups),
        "largest_exact_duplicate_group_size": exact_groups[0]["count"] if exact_groups else 0,
        "cross_split_exact_feature_matches": cross_exact,
        "n_cross_split_exact_matches": len(cross_exact),
        "near_duplicate_pairs_top": near_pairs,
        "assessment": _assess(exact_groups, cross_exact, near_pairs),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return report


def _assess(exact_groups, cross_exact, near_pairs) -> dict:
    if cross_exact:
        verdict = "B_actual_dataset_leakage_or_evaluation_contamination"
        detail = (
            "Exact feature vectors appear in both train and test partitions despite "
            "group-based splitting. This is not harmless preprocessing collision."
        )
    elif exact_groups and exact_groups[0]["count"] >= 10:
        verdict = "D_insufficient_diversity_with_duplicate_artifacts"
        detail = "Large within-dataset duplicate groups suggest synthetic generator artifacts."
    elif near_pairs:
        verdict = "C_synthetic_generator_artifacts_and_near_duplicate_risk"
        detail = "No exact cross-split matches, but very high cosine similarity exists across splits."
    else:
        verdict = "A_mostly_harmless_or_low_risk"
        detail = "No exact cross-split duplicates detected."

    return {"verdict_code": verdict, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset duplicate / leakage audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/investigation/duplicate_audit_report.json"),
    )
    args = parser.parse_args()
    report = run_audit(output_path=args.output)
    print(json.dumps(report["assessment"], indent=2))
    print(f"exact groups: {report['n_exact_duplicate_groups']}")
    print(f"cross-split exact matches: {report['n_cross_split_exact_matches']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
