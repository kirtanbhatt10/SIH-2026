"""Main orchestrator for the accuracy improvement experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from ml.phase5.config import PHASE3_CLEAN_MODEL_PATH

from .config import (
    ARTIFACTS_DIR,
    EXPERIMENT_ID,
    FROZEN_BASELINE,
    GENERATOR_VERSION,
    REPORTS_DIR,
    TARGETS,
)
from .data import load_experiment_data, split_report
from .evaluation import error_analysis, evaluate_candidate, qualifies, rank_candidates
from .features import analyze_features, augment_features
from .models import create_model_catalog
from .tuning import tune_model_family

HANDOFF_REPORT = Path(__file__).resolve().parents[3] / "docs" / "AI_ML_ACCURACY_IMPROVEMENT_REPORT.md"


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _markdown_error_analysis(report: dict[str, Any]) -> str:
    lines = [
        "# Error Analysis",
        "",
        f"Partition: `{report['partition']}`",
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
    lines.extend(
        [
            "",
            f"Benign false positives: {report['benign_false_positive_count']}",
            f"Threat false negatives: {report['threat_false_negative_count']}",
        ]
    )
    return "\n".join(lines)


def run_experiments(*, skip_locked_test: bool = False) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data()
    X32 = data.features
    y = data.labels

    baseline_model = joblib.load(PHASE3_CLEAN_MODEL_PATH)
    cal_err = error_analysis(
        baseline_model,
        X32,
        y,
        data.metadata,
        data.cal_dev_idx,
        partition="cal_dev_baseline_reproduction",
    )
    _write_json(REPORTS_DIR / "baseline_metrics.json", cal_err["metrics"])
    _write_json(REPORTS_DIR / "error_analysis.json", cal_err)
    (REPORTS_DIR / "error_analysis.md").write_text(
        _markdown_error_analysis(cal_err), encoding="utf-8"
    )

    dev_idx = data.development_idx
    feat_report = analyze_features(data, analysis_idx=dev_idx)
    _write_json(REPORTS_DIR / "feature_analysis.json", feat_report)

    tuning_results = []
    cal_dev_rows = []
    fitted_models: dict[str, Any] = {}

    for model_name in create_model_catalog():
        tuned = tune_model_family(
            model_name,
            X32[data.train_fit_idx],
            y[data.train_fit_idx],
            X32[data.inner_dev_idx],
            y[data.inner_dev_idx],
        )
        tuning_results.append(
            {
                "model_name": model_name,
                "best_params": tuned["best_params"],
                "best_inner_dev": tuned["best_inner_dev"],
                "n_trials": len(tuned["trials"]),
            }
        )
        best_model = tuned["best_model"]
        best_model.fit(X32[data.model_train_idx], y[data.model_train_idx])
        cal_metrics = evaluate_candidate(
            best_model,
            X32[data.cal_dev_idx],
            y[data.cal_dev_idx],
            partition="cal_dev",
            model_name=model_name,
        )
        row = {
            "model_name": model_name,
            "feature_set": "base_32",
            "best_params": tuned["best_params"],
            **{k: cal_metrics[k] for k in (
                "accuracy", "macro_f1", "weighted_f1", "benign_precision",
                "benign_recall", "benign_false_positive_rate", "threat_precision",
                "threat_recall", "brier_score_threat", "expected_calibration_error_threat",
                "latency_ms_per_sample",
            )},
        }
        cal_dev_rows.append(row)
        fitted_models[f"{model_name}|base_32"] = best_model

    _write_json(REPORTS_DIR / "hyperparameter_results.json", tuning_results)

    ablation_rows = []
    for use_engineered in (False, True):
        X_aug, names = augment_features(X32, use_engineered=use_engineered)
        tuned = tune_model_family(
            "random_forest_balanced",
            X_aug[data.train_fit_idx],
            y[data.train_fit_idx],
            X_aug[data.inner_dev_idx],
            y[data.inner_dev_idx],
        )
        model = tuned["best_model"]
        model.fit(X_aug[data.model_train_idx], y[data.model_train_idx])
        metrics = evaluate_candidate(
            model,
            X_aug[data.cal_dev_idx],
            y[data.cal_dev_idx],
            partition="cal_dev",
            model_name="random_forest_balanced",
        )
        ablation_rows.append(
            {
                "experiment": "feature_ablation_random_forest",
                "use_engineered_features": use_engineered,
                "n_features": len(names),
                "feature_names": names,
                "best_params": tuned["best_params"],
                **{k: metrics[k] for k in (
                    "accuracy", "macro_f1", "benign_false_positive_rate", "threat_recall"
                )},
            }
        )

    _write_json(REPORTS_DIR / "ablation_results.json", ablation_rows)
    _write_json(REPORTS_DIR / "model_comparison.json", cal_dev_rows)

    ranked = rank_candidates(cal_dev_rows)
    best_dev = ranked[0]
    best_key = f"{best_dev['model_name']}|base_32"
    final_model = fitted_models[best_key]
    final_config = {
        "model_name": best_dev["model_name"],
        "feature_set": "base_32",
        "feature_count": 32,
        "best_params": best_dev["best_params"],
        "calibration": "uncalibrated",
        "selection_partition": "cal_dev",
        "selection_ranking": "macro_f1, -benign_fpr, threat_recall, accuracy, -latency",
        "cal_dev_metrics": best_dev,
        "frozen_baseline_reference": FROZEN_BASELINE["locked_test"],
    }
    ok, reject_reasons = qualifies(best_dev)
    final_config["qualifies_for_targets_on_cal_dev"] = ok
    final_config["qualification_notes"] = reject_reasons
    _write_json(REPORTS_DIR / "final_candidate.json", final_config)

    joblib.dump(final_model, ARTIFACTS_DIR / "final_candidate_model.joblib")
    _write_json(ARTIFACTS_DIR / "final_candidate_config.json", final_config)

    locked_result = None
    base = FROZEN_BASELINE["locked_test"]
    if not skip_locked_test:
        locked_metrics = evaluate_candidate(
            final_model,
            X32[data.locked_test_idx],
            y[data.locked_test_idx],
            partition="locked_test_one_shot",
            model_name=best_dev["model_name"],
        )
        locked_ok, locked_reasons = qualifies(locked_metrics)
        locked_result = {
            "candidate": final_config,
            "locked_test_metrics": locked_metrics,
            "qualifies_on_locked_test": locked_ok,
            "qualification_notes": locked_reasons,
            "baseline_locked_test": FROZEN_BASELINE["locked_test"],
            "deltas_vs_baseline": {
                "accuracy": locked_metrics["accuracy"] - base["accuracy"],
                "macro_f1": locked_metrics["macro_f1"] - base["macro_f1"],
                "weighted_f1": locked_metrics["weighted_f1"] - base["weighted_f1"],
                "benign_precision": locked_metrics["benign_precision"] - base["benign_precision"],
                "benign_recall": locked_metrics["benign_recall"] - base["benign_recall"],
                "benign_fpr": locked_metrics["benign_false_positive_rate"] - base["benign_fpr"],
                "threat_precision": locked_metrics["threat_precision"] - base["threat_precision"],
                "threat_recall": locked_metrics["threat_recall"] - base["threat_recall"],
                "brier": locked_metrics["brier_score_threat"] - base["brier"],
                "ece": locked_metrics["expected_calibration_error_threat"] - base["ece"],
                "latency_ms_per_sample": locked_metrics["latency_ms_per_sample"]
                - base["latency_ms_per_sample"],
            },
            "warning": "Locked test evaluated exactly once after candidate freeze.",
        }
        _write_json(REPORTS_DIR / "final_candidate_locked_test.json", locked_result)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": FROZEN_BASELINE["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "split_manifest": str(Path(__file__).resolve().parents[2] / "output" / "clean_splits" / "frozen_split_manifest.json"),
        "split_report": split_report(data),
        "targets": TARGETS,
        "frozen_baseline": FROZEN_BASELINE,
        "best_cal_dev_candidate": best_dev,
        "ranked_cal_dev_candidates": ranked,
        "locked_test_evaluated": locked_result is not None,
        "reproducibility": {
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "git_commit": _git_hash(),
        },
        "promotion_rule": "Do not overwrite phase3_clean/phase4_clean/phase5_clean.",
    }
    _write_json(REPORTS_DIR / "experiment_manifest.json", manifest)
    _write_handoff_report(best_dev, ranked, locked_result, cal_err)

    return {
        "best_cal_dev": best_dev,
        "ranked": ranked,
        "locked_test": locked_result,
        "manifest": manifest,
    }


def _write_handoff_report(
    best_dev: dict[str, Any],
    ranked: list[dict[str, Any]],
    locked: dict[str, Any] | None,
    cal_err: dict[str, Any],
) -> None:
    base = FROZEN_BASELINE["locked_test"]
    achieved_85 = False
    security_ok = False
    locked_metrics = locked["locked_test_metrics"] if locked else None
    if locked_metrics:
        achieved_85 = locked_metrics["accuracy"] >= TARGETS["accuracy"]
        security_ok, _ = qualifies(locked_metrics)

    lines = [
        "# AI/ML Accuracy Improvement Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Objective",
        "",
        "Controlled experiment to reach ~85% accuracy without breaking security metrics.",
        "Frozen production-review artifacts were **not modified**.",
        "",
        "## Baseline (frozen, locked test)",
        "",
        f"- Accuracy: {base['accuracy']:.4f}",
        f"- Macro F1: {base['macro_f1']:.4f}",
        f"- Benign FPR: {base['benign_fpr']:.4f}",
        f"- Threat recall: {base['threat_recall']:.4f}",
        "",
        "## Error analysis (cal_dev baseline reproduction)",
        "",
    ]
    for row in cal_err["dominant_confusion_pairs"][:8]:
        lines.append(
            f"- {row['true_class']} -> {row['predicted_class']}: {row['count']} errors"
        )

    lines.extend(["", "## Best development candidate (cal_dev selection)", ""])
    lines.append(f"- Model: `{best_dev['model_name']}`")
    lines.append(f"- Accuracy: {best_dev['accuracy']:.4f}")
    lines.append(f"- Macro F1: {best_dev['macro_f1']:.4f}")
    lines.append(f"- Benign FPR: {best_dev['benign_false_positive_rate']:.4f}")
    lines.append(f"- Threat recall: {best_dev['threat_recall']:.4f}")

    if locked_metrics:
        lines.extend(["", "## Locked-test one-shot evaluation", ""])
        lines.append("| Metric | Baseline | Candidate | Delta |")
        lines.append("|--------|----------|-----------|-------|")
        mapping = {
            "accuracy": "accuracy",
            "macro_f1": "macro_f1",
            "weighted_f1": "weighted_f1",
            "benign_precision": "benign_precision",
            "benign_recall": "benign_recall",
            "benign_false_positive_rate": "benign_fpr",
            "threat_precision": "threat_precision",
            "threat_recall": "threat_recall",
            "brier_score_threat": "brier",
            "expected_calibration_error_threat": "ece",
            "latency_ms_per_sample": "latency_ms_per_sample",
        }
        for cand_key, base_key in mapping.items():
            b = base[base_key]
            c = locked_metrics[cand_key]
            lines.append(f"| {cand_key} | {b:.4f} | {c:.4f} | {c-b:+.4f} |")

    lines.extend(["", "## Conclusion", ""])
    if achieved_85 and security_ok:
        lines.append(
            "Candidate meets all stated targets on the one-shot locked test. "
            "Promotion still requires explicit team approval; frozen artifacts unchanged."
        )
    else:
        lines.append(
            "**85% target not achieved without violating security/evaluation constraints** "
            "on the one-shot locked test, and/or cal_dev selection did not transfer."
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ai/ml/experiments/accuracy_improvement/reports/`",
            "- `ai/ml/experiments/accuracy_improvement/artifacts/`",
            "",
            "## Promotion rule",
            "",
            "Do NOT replace `phase3_clean/`, `phase4_clean/`, or `phase5_clean/`.",
        ]
    )
    HANDOFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run accuracy improvement experiments")
    parser.add_argument("--skip-locked-test", action="store_true")
    args = parser.parse_args()
    result = run_experiments(skip_locked_test=args.skip_locked_test)
    print(json.dumps(
        {
            "best_cal_dev_accuracy": result["best_cal_dev"]["accuracy"],
            "best_cal_dev_macro_f1": result["best_cal_dev"]["macro_f1"],
            "best_model": result["best_cal_dev"]["model_name"],
            "locked_test_evaluated": result["locked_test"] is not None,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
