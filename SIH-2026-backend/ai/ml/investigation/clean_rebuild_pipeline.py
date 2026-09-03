"""Build clean dataset, retrain models, compare calibration, write comparison reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from ml.dataset_v2 import config_v2 as cfg
from ml.dataset_v2.build_dataset import build_dataset
from ml.dataset_v2.validate_dataset import validate_dataset
from ml.investigation.dataset_root_cause import analyze_dataset
from ml.investigation.model_experiments import run_experiments
from ml.phase3.data import grouped_train_test_split, load_dataset_v2
from ml.phase3.evaluation import evaluate_model, select_candidate_model, build_comparison_table
from ml.phase3.models import create_baseline_models
from ml.phase5.artifacts import load_phase3_model


def _run_old_baseline(dataset_dir: Path) -> dict:
    dataset = load_dataset_v2(dataset_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    model = load_phase3_model()
    return evaluate_model(
        model,
        dataset.features[split.test_idx],
        dataset.labels[split.test_idx],
        model_name="logistic_regression_phase3_saved",
    )


def _calibration_compare(model, X_train, y_train, X_test, y_test) -> dict:
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    y_train_threat = (y_train != 0).astype(int)
    y_test_threat = (y_test != 0).astype(int)
    train_threat_p = train_proba[:, 1:].sum(axis=1)
    test_threat_p = test_proba[:, 1:].sum(axis=1)

    uncal = evaluate_model(model, X_test, y_test, model_name="uncalibrated")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_threat_p, y_train_threat)
    cal_scores = iso.predict(test_threat_p)
    cal_pred = np.where(cal_scores >= 0.5, 1, 0)
    benign_mask = y_test == 0
    cal_benign_fpr = float((benign_mask & (cal_pred == 1)).sum() / benign_mask.sum())
    cal_threat_recall = float(cal_pred[y_test_threat == 1].mean())

    return {
        "uncalibrated": {
            "benign_fpr": uncal["benign_false_positive_rate"],
            "benign_recall": uncal["benign_recall"],
            "threat_recall": uncal["threat_recall"],
            "macro_f1": uncal["macro_f1"],
            "accuracy": uncal["accuracy"],
        },
        "isotonic_threat_binary": {
            "benign_fpr": cal_benign_fpr,
            "threat_recall": cal_threat_recall,
            "brier": float(brier_score_loss(y_test_threat, np.clip(cal_scores, 0, 1))),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean dataset + model rebuild pipeline")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ai_dir = Path(__file__).resolve().parents[2]
    ml_dir = ai_dir / "ml"
    old_dir = ml_dir / "training_data_v2"
    clean_dir = ml_dir / cfg.CLEAN_OUTPUT_DIR_NAME
    inv = ml_dir / "investigation"

    print("=== OLD dataset root cause ===")
    old_root = analyze_dataset(old_dir)
    with (inv / "dataset_root_cause_report.json").open("w", encoding="utf-8") as fh:
        json.dump(old_root, fh, indent=2)

    if not args.skip_build:
        print("=== BUILD clean dataset ===")
        build_dataset(
            output_dir=clean_dir,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            save_audio=False,
            dataset_version=cfg.CLEAN_DATASET_VERSION,
        )

    print("=== VALIDATE clean dataset ===")
    checks = validate_dataset(clean_dir, strict_clean=True)
    failed = [c for c in checks if not c.passed]
    for c in checks:
        print(c)
    if failed:
        print("VALIDATION FAILED:", failed)
        return 1

    new_root = analyze_dataset(clean_dir)
    with (inv / "dataset_root_cause_report_clean.json").open("w", encoding="utf-8") as fh:
        json.dump(new_root, fh, indent=2)

    print("=== MODEL experiments (clean, held-out) ===")
    experiments_path = inv / "model_experiments_clean_report.json"

    dataset = load_dataset_v2(clean_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=args.seed)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]

    from ml.investigation.model_experiments import _make_models

    rows = []
    fitted = {}
    for name, model in _make_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        fitted[name] = model
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "benign_fpr": metrics["benign_false_positive_rate"],
                "benign_recall": metrics["benign_recall"],
                "threat_recall": metrics["threat_recall"],
                "latency_ms_per_sample": metrics["latency_ms_mean"],
            }
        )
    with experiments_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    comparison = build_comparison_table(
        [evaluate_model(fitted[n], X_test, y_test, model_name=n) for n in fitted]
    )
    selected_name, reason = select_candidate_model(comparison)

    print("=== CALIBRATION compare (clean, selected model) ===")
    cal = _calibration_compare(fitted[selected_name], X_train, y_train, X_test, y_test)

    print("=== OLD saved model baseline ===")
    old_metrics = _run_old_baseline(old_dir)
    new_metrics = evaluate_model(fitted[selected_name], X_test, y_test, model_name=selected_name)

    def _pick(m):
        return {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "benign_fpr": m["benign_false_positive_rate"],
            "benign_recall": m["benign_recall"],
            "threat_recall": m["threat_recall"],
        }

    comparison_json = {
        "old_dataset_saved_model": _pick(old_metrics),
        "clean_dataset_selected_model": {
            **_pick(new_metrics),
            "selected_model": selected_name,
            "selection_reason": reason,
        },
        "duplicate_stats": {
            "old": {
                "duplicate_groups": old_root["n_duplicate_groups"],
                "cross_split_exact": old_root["n_cross_split_exact_matches"],
                "collapsed_benign": old_root["collapsed_squelch_by_class"].get("0", 0),
            },
            "clean": {
                "duplicate_groups": new_root["n_duplicate_groups"],
                "cross_split_exact": new_root["n_cross_split_exact_matches"],
                "collapsed_benign": new_root["collapsed_squelch_by_class"].get("0", 0),
            },
        },
        "calibration_clean_dataset": cal,
        "model_experiments_clean": rows,
        "split": {
            "seed": args.seed,
            "train_size": int(len(split.train_idx)),
            "test_size": int(len(split.test_idx)),
            "policy": "stratified_grouped_by_generation_group",
        },
    }
    with (inv / "clean_dataset_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison_json, fh, indent=2)

    md_lines = [
        "# Clean Dataset Comparison",
        "",
        "| Metric | Old (saved model, old data) | New (clean data, " + selected_name + ") | Change |",
        "|--------|------------------------------|----------------------------------------|--------|",
    ]
    for key in ("accuracy", "macro_f1", "benign_fpr", "benign_recall", "threat_recall"):
        o = _pick(old_metrics)[key]
        n = _pick(new_metrics)[key]
        md_lines.append(f"| {key} | {o:.4f} | {n:.4f} | {n-o:+.4f} |")
    md_lines.extend(
        [
            "",
            "## Duplicate stats",
            f"- Old cross-split exact matches: {old_root['n_cross_split_exact_matches']}",
            f"- Clean cross-split exact matches: {new_root['n_cross_split_exact_matches']}",
            "",
            "## Backend status",
            "**BLOCKED** — synthetic validation only; verify FPR targets before integration.",
        ]
    )
    (inv / "clean_dataset_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(comparison_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
