"""Main orchestrator for the 82% accuracy data-diversity experiment."""

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

from .build_experiment_dataset import build_experiment_dataset
from .config import (
    ARTIFACTS_DIR,
    EXPERIMENT_DATASET_DIR,
    EXPERIMENT_ID,
    FROZEN_BASELINE,
    GENERATOR_VERSION,
    REPORTS_DIR,
    TARGETS,
)
from .data import load_experiment_data, split_report
from .error_analysis import markdown_error_analysis, run_error_analysis_v2
from .evaluation import evaluate_candidate, qualifies, rank_candidates, security_healthy
from .features import analyze_features, augment_features
from .models import create_model_catalog
from .tuning import tune_model_family

HANDOFF_REPORT = Path(__file__).resolve().parents[3] / "docs" / "AI_ML_82_PERCENT_EXPERIMENT_REPORT.md"


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


def _metrics_row(
    model_name: str,
    feature_set: str,
    n_features: int,
    best_params: dict[str, Any],
    metrics: dict[str, Any],
    experiment: str,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model_name": model_name,
        "feature_set": feature_set,
        "n_features": n_features,
        "best_params": best_params,
        **{k: metrics[k] for k in (
            "accuracy", "macro_f1", "weighted_f1", "benign_precision",
            "benign_recall", "benign_false_positive_rate", "threat_precision",
            "threat_recall", "latency_ms_per_sample",
        )},
    }


def _train_and_eval(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    data: Any,
    *,
    experiment: str,
    feature_set: str,
) -> tuple[dict[str, Any], Any]:
    tuned = tune_model_family(
        model_name,
        X[data.train_fit_idx],
        y[data.train_fit_idx],
        X[data.inner_dev_idx],
        y[data.inner_dev_idx],
    )
    model = tuned["best_model"]
    model.fit(X[data.model_train_idx], y[data.model_train_idx])
    metrics = evaluate_candidate(
        model,
        X[data.cal_dev_idx],
        y[data.cal_dev_idx],
        partition="cal_dev",
        model_name=model_name,
    )
    row = _metrics_row(
        model_name,
        feature_set,
        X.shape[1],
        tuned["best_params"],
        metrics,
        experiment,
    )
    return row, model


def _experiment_dataset_ready() -> bool:
    info_path = EXPERIMENT_DATASET_DIR / "dataset_info.json"
    if not info_path.exists():
        return False
    with info_path.open(encoding="utf-8") as fh:
        info = json.load(fh)
    return (
        info.get("dataset_version") == "dataset_v2_experiment.0"
        and info.get("generator_version") == GENERATOR_VERSION
        and int(info.get("nan_count", 1)) == 0
        and int(info.get("inf_count", 1)) == 0
    )


def run_experiment(
    *,
    skip_dataset_build: bool = False,
    skip_locked_test: bool = False,
) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_dataset_build and not _experiment_dataset_ready():
        build_info = build_experiment_dataset()
        _write_json(REPORTS_DIR / "dataset_build.json", build_info)

    err = run_error_analysis_v2()
    _write_json(REPORTS_DIR / "error_analysis_v2.json", err)
    (REPORTS_DIR / "error_analysis_v2.md").write_text(
        markdown_error_analysis(err), encoding="utf-8"
    )

    data = load_experiment_data()
    X32 = data.features
    y = data.labels
    _write_json(REPORTS_DIR / "split_report.json", split_report(data))

    dev_idx = data.development_idx
    feat_report = analyze_features(data, analysis_idx=dev_idx)
    _write_json(REPORTS_DIR / "feature_analysis.json", feat_report)

    candidates: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, np.ndarray]] = {}

    # Experiment A — same model architecture on diversity dataset
    row_a, model_a = _train_and_eval(
        "linear_svm_balanced",
        X32,
        y,
        data,
        experiment="A_same_model",
        feature_set="base_32",
    )
    row_a["candidate_key"] = "A|linear_svm_balanced|base_32"
    candidates.append(row_a)
    fitted[row_a["candidate_key"]] = (model_a, X32)

    experiment_b_ran = False
    experiment_c_ran = False

    # Experiment B — feature extensions if A insufficient
    if row_a["accuracy"] < TARGETS["accuracy"] or not security_healthy(row_a):
        experiment_b_ran = True
        X40, names40 = augment_features(X32, use_engineered=True)
        data40 = load_experiment_data(feature_names=names40, features=X40)
        row_b, model_b = _train_and_eval(
            "linear_svm_balanced",
            X40,
            y,
            data40,
            experiment="B_feature_extension",
            feature_set="base_32_plus_engineered",
        )
        row_b["candidate_key"] = "B|linear_svm_balanced|base_40"
        candidates.append(row_b)
        fitted[row_b["candidate_key"]] = (model_b, X40)
        _write_json(
            REPORTS_DIR / "experiment_b_ablation.json",
            {
                "base_32": {
                    k: row_a[k]
                    for k in ("accuracy", "macro_f1", "benign_false_positive_rate", "threat_recall")
                },
                "base_40": {
                    k: row_b[k]
                    for k in ("accuracy", "macro_f1", "benign_false_positive_rate", "threat_recall")
                },
            },
        )

    best_so_far = rank_candidates(candidates)[0]
    if best_so_far["accuracy"] < TARGETS["accuracy"] or not security_healthy(best_so_far):
        experiment_c_ran = True
        use_engineered = best_so_far.get("feature_set") == "base_32_plus_engineered"
        X_use, feat_names = augment_features(X32, use_engineered=use_engineered)
        data_use = (
            load_experiment_data(feature_names=feat_names, features=X_use)
            if use_engineered
            else data
        )
        feature_tag = "base_40" if use_engineered else "base_32"
        for model_name in ("rbf_svm_balanced", "extra_trees_balanced", "hist_gradient_boosting"):
            row_c, model_c = _train_and_eval(
                model_name,
                X_use,
                y,
                data_use,
                experiment="C_secondary_model",
                feature_set="base_32_plus_engineered" if use_engineered else "base_32",
            )
            row_c["candidate_key"] = f"C|{model_name}|{feature_tag}"
            candidates.append(row_c)
            fitted[row_c["candidate_key"]] = (model_c, X_use)

    _write_json(REPORTS_DIR / "model_comparison.json", candidates)
    ranked = rank_candidates(candidates)
    best = ranked[0]
    final_model, X_final = fitted[best["candidate_key"]]

    ok, reject_reasons = qualifies(best)
    final_config = {
        "experiment_id": EXPERIMENT_ID,
        "selected_experiment": best["experiment"],
        "model_name": best["model_name"],
        "feature_set": best["feature_set"],
        "feature_count": best["n_features"],
        "best_params": best["best_params"],
        "calibration": "uncalibrated",
        "dataset": str(EXPERIMENT_DATASET_DIR),
        "dataset_version": "dataset_v2_experiment.0",
        "generator_version": GENERATOR_VERSION,
        "selection_partition": "cal_dev",
        "selection_ranking": "accuracy, macro_f1, -benign_fpr, threat_recall",
        "cal_dev_metrics": best,
        "qualifies_for_targets_on_cal_dev": ok,
        "qualification_notes": reject_reasons,
        "experiments_run": {
            "A_same_model": True,
            "B_feature_extension": experiment_b_ran,
            "C_secondary_models": experiment_c_ran,
        },
        "frozen_baseline_reference": FROZEN_BASELINE,
    }
    _write_json(REPORTS_DIR / "final_candidate_dev.json", final_config)
    joblib.dump(final_model, ARTIFACTS_DIR / "final_candidate_model.joblib")
    _write_json(ARTIFACTS_DIR / "final_candidate_config.json", final_config)

    locked_result = None
    base = FROZEN_BASELINE["locked_test"]
    if not skip_locked_test:
        locked_metrics = evaluate_candidate(
            final_model,
            X_final[data.locked_test_idx],
            y[data.locked_test_idx],
            partition="locked_test_one_shot",
            model_name=best["model_name"],
        )
        locked_ok, locked_reasons = qualifies(locked_metrics)
        locked_result = {
            "candidate": final_config,
            "locked_test_metrics": locked_metrics,
            "qualifies_on_locked_test": locked_ok,
            "qualification_notes": locked_reasons,
            "baseline_locked_test_clean_dataset": base,
            "comparison_note": (
                "Baseline metrics are from frozen clean locked_test (463 samples). "
                "Candidate metrics are from experiment dataset locked_test partition "
                "using the same split policy but different generated samples."
            ),
            "comparison_table": {
                "accuracy": {
                    "baseline": base["accuracy"],
                    "candidate": locked_metrics["accuracy"],
                    "delta_pp": (locked_metrics["accuracy"] - base["accuracy"]) * 100,
                },
                "macro_f1": {
                    "baseline": base["macro_f1"],
                    "candidate": locked_metrics["macro_f1"],
                    "delta_pp": (locked_metrics["macro_f1"] - base["macro_f1"]) * 100,
                },
                "weighted_f1": {
                    "baseline": base["weighted_f1"],
                    "candidate": locked_metrics["weighted_f1"],
                    "delta_pp": (locked_metrics["weighted_f1"] - base["weighted_f1"]) * 100,
                },
                "benign_precision": {
                    "baseline": base["benign_precision"],
                    "candidate": locked_metrics["benign_precision"],
                    "delta_pp": (locked_metrics["benign_precision"] - base["benign_precision"]) * 100,
                },
                "benign_recall": {
                    "baseline": base["benign_recall"],
                    "candidate": locked_metrics["benign_recall"],
                    "delta_pp": (locked_metrics["benign_recall"] - base["benign_recall"]) * 100,
                },
                "benign_fpr": {
                    "baseline": base["benign_fpr"],
                    "candidate": locked_metrics["benign_false_positive_rate"],
                    "delta_pp": (
                        locked_metrics["benign_false_positive_rate"] - base["benign_fpr"]
                    ) * 100,
                },
                "threat_precision": {
                    "baseline": base["threat_precision"],
                    "candidate": locked_metrics["threat_precision"],
                    "delta_pp": (locked_metrics["threat_precision"] - base["threat_precision"]) * 100,
                },
                "threat_recall": {
                    "baseline": base["threat_recall"],
                    "candidate": locked_metrics["threat_recall"],
                    "delta_pp": (locked_metrics["threat_recall"] - base["threat_recall"]) * 100,
                },
                "latency_ms_per_sample": {
                    "baseline": base["latency_ms_per_sample"],
                    "candidate": locked_metrics["latency_ms_per_sample"],
                    "delta_pp": (
                        locked_metrics["latency_ms_per_sample"] - base["latency_ms_per_sample"]
                    ) * 100,
                },
            },
            "confusion_matrix": locked_metrics["confusion_matrix"],
            "warning": "Locked test evaluated exactly once after candidate freeze.",
        }
        _write_json(REPORTS_DIR / "accuracy_82_final_candidate.json", locked_result)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "dataset_v2_experiment.0",
        "generator_version": GENERATOR_VERSION,
        "split_report": split_report(data),
        "targets": TARGETS,
        "frozen_baseline": FROZEN_BASELINE,
        "error_analysis_v2": str(REPORTS_DIR / "error_analysis_v2.json"),
        "best_cal_dev_candidate": best,
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
        "frozen_artifacts_untouched": True,
    }
    _write_json(REPORTS_DIR / "experiment_manifest.json", manifest)
    _write_handoff_report(best, ranked, locked_result, err, experiment_b_ran, experiment_c_ran)

    return {
        "best_cal_dev": best,
        "ranked": ranked,
        "locked_test": locked_result,
        "manifest": manifest,
    }


def _write_handoff_report(
    best: dict[str, Any],
    ranked: list[dict[str, Any]],
    locked: dict[str, Any] | None,
    err: dict[str, Any],
    experiment_b: bool,
    experiment_c: bool,
) -> None:
    base = FROZEN_BASELINE["locked_test"]
    achieved_82 = False
    security_ok = False
    locked_metrics = locked["locked_test_metrics"] if locked else None
    if locked_metrics:
        achieved_82 = locked_metrics["accuracy"] >= TARGETS["accuracy"]
        security_ok, _ = qualifies(locked_metrics)

    lines = [
        "# AI/ML 82% Accuracy Data-Diversity Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Objective",
        "",
        "Reach ~82% locked-test accuracy via realistic class diversity around decision "
        "boundaries, without modifying frozen production artifacts.",
        "",
        "## Baseline (frozen clean, locked test)",
        "",
        f"- Accuracy: {base['accuracy']:.4f} ({base['accuracy']*100:.2f}%)",
        f"- Macro F1: {base['macro_f1']:.4f}",
        f"- Benign FPR: {base['benign_fpr']:.4f} ({base['benign_fpr']*100:.2f}%)",
        f"- Threat recall: {base['threat_recall']:.4f}",
        "",
        "## Error analysis v2 (clean development, frozen baseline)",
        "",
    ]
    for row in err["dominant_confusion_pairs"][:8]:
        lines.append(
            f"- {row['true_class']} -> {row['predicted_class']}: {row['count']} errors"
        )

    lines.extend([
        "",
        "## Dataset changes",
        "",
        "- New dataset: `ai/ml/training_data_v2_experiment/` (`dataset_v2_experiment.0`)",
        "- FSK/OOK/tone/benign diversity expanded around observed confusion boundaries",
        "- Clean dataset `training_data_v2_clean/` **unchanged**",
        "",
        "## Experiments run",
        "",
        "- Experiment A (linear SVM, base 32): yes",
        f"- Experiment B (feature extensions): {'yes' if experiment_b else 'skipped — A sufficient'}",
        f"- Experiment C (secondary models): {'yes' if experiment_c else 'skipped'}",
        "",
        "## Best development candidate (cal_dev selection)",
        "",
        f"- Experiment: `{best['experiment']}`",
        f"- Model: `{best['model_name']}`",
        f"- Features: `{best['feature_set']}` ({best['n_features']} dims)",
        f"- Accuracy: {best['accuracy']:.4f}",
        f"- Macro F1: {best['macro_f1']:.4f}",
        f"- Benign FPR: {best['benign_false_positive_rate']:.4f}",
        f"- Threat recall: {best['threat_recall']:.4f}",
    ])

    if locked and locked_metrics:
        table = locked["comparison_table"]
        lines.extend(["", "## Locked-test one-shot evaluation", ""])
        lines.append("| Metric | Baseline | Candidate | Delta (pp) |")
        lines.append("|--------|----------|-----------|------------|")
        for key in (
            "accuracy", "macro_f1", "weighted_f1", "benign_precision", "benign_recall",
            "benign_fpr", "threat_precision", "threat_recall", "latency_ms_per_sample",
        ):
            row = table[key]
            lines.append(
                f"| {key} | {row['baseline']:.4f} | {row['candidate']:.4f} | "
                f"{row['delta_pp']:+.2f} |"
            )

    lines.extend(["", "## Conclusion", ""])
    if achieved_82 and security_ok:
        lines.append(
            "Candidate meets all stated targets on the one-shot experiment locked test. "
            "Promotion requires explicit team-lead approval; frozen artifacts unchanged."
        )
    else:
        lines.append("**82% target not achieved.**")
        if locked_metrics:
            lines.append(
                f"Locked-test accuracy: {locked_metrics['accuracy']*100:.2f}% "
                f"(target >= {TARGETS['accuracy']*100:.0f}%)."
            )

    lines.extend([
        "",
        "## Frozen artifact protection",
        "",
        "- `phase3_clean/`, `phase4_clean/`, `phase5_clean/` — untouched",
        "- `training_data_v2_clean/` — untouched",
        "- `frozen_split_manifest.json` — untouched",
        "",
        "## Files",
        "",
        "- `ai/ml/experiments/accuracy_82/reports/`",
        "- `ai/ml/experiments/accuracy_82/artifacts/`",
        "- `reports/accuracy_82_final_candidate.json`",
    ])
    HANDOFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run accuracy_82 data-diversity experiment")
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--skip-locked-test", action="store_true")
    args = parser.parse_args()
    result = run_experiment(
        skip_dataset_build=args.skip_dataset_build,
        skip_locked_test=args.skip_locked_test,
    )
    print(json.dumps(
        {
            "best_cal_dev_accuracy": result["best_cal_dev"]["accuracy"],
            "best_model": result["best_cal_dev"]["model_name"],
            "locked_test_evaluated": result["locked_test"] is not None,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
