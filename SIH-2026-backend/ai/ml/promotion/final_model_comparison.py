"""Final apples-to-apples locked-test comparison: uncalibrated vs isotonic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.investigation.model_experiments import _make_models
from ml.phase3.config import CLASS_NAMES, FEATURE_NAMES
from ml.phase3.data import load_dataset_v2
from ml.phase3.evaluation import evaluate_model
from ml.phase4.calibrator import RiskCalibrator
from ml.phase4.evaluation import evaluate as evaluate_calibrated_risk
from ml.phase5.evaluation import evaluate_service
from ml.phase5.service import InferenceService
from ml.promotion.config_clean import (
    CLEAN_DATASET_DIR,
    DATASET_VERSION,
    FROZEN_SPLIT_MANIFEST_PATH,
    GENERATOR_VERSION,
    PHASE5_CLEAN_OUTPUT_DIR,
    SELECTED_MODEL_NAME,
)
from ml.promotion.splits import load_frozen_manifest
from ml.promotion.uncalibrated_adapter import UncalibratedRiskAdapter

COMPARISON_JSON_PATH = PHASE5_CLEAN_OUTPUT_DIR / "final_model_comparison.json"
COMPARISON_MD_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "AI_ML_FINAL_MODEL_COMPARISON.md"
)

CANDIDATE_UNCALIBRATED = "uncalibrated"
CANDIDATE_ISOTONIC = "isotonic_calibrated"


def _create_model():
    models = _make_models()
    return models[SELECTED_MODEL_NAME]


def _phase_metadata(model_name: str, calibration_method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    phase3 = {
        "model_name": model_name,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "feature_count": 32,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
    }
    phase4 = {
        "phase3_model_name": model_name,
        "calibration_method": calibration_method,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "risk_thresholds": {},
    }
    return phase3, phase4


def _build_service(model: Any, calibrator: Any, calibration_method: str) -> InferenceService:
    phase3_meta, phase4_meta = _phase_metadata(SELECTED_MODEL_NAME, calibration_method)
    phase4_meta["risk_thresholds"] = calibrator.get_thresholds()
    return InferenceService(
        model=model,
        calibrator=calibrator,
        phase3_metadata=phase3_meta,
        phase4_metadata=phase4_meta,
    )


def _locked_test_metrics(
    service: InferenceService,
    calibrator: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_ids: list[str],
    *,
    candidate: str,
) -> dict[str, Any]:
    svc_metrics = evaluate_service(
        service,
        X_test,
        y_test,
        sample_ids=sample_ids,
        partition="locked_test",
    )
    cal_metrics = evaluate_calibrated_risk(
        calibrator,
        X_test,
        y_test,
        partition="locked_test",
    )

    return {
        "candidate": candidate,
        "partition": "locked_test",
        "n_samples": svc_metrics["n_samples"],
        "accuracy": svc_metrics["accuracy"],
        "macro_f1": svc_metrics["macro_f1"],
        "weighted_f1": svc_metrics["weighted_f1"],
        "benign_precision": cal_metrics["per_class"]["benign"]["precision"],
        "benign_recall": svc_metrics["benign_recall"],
        "benign_false_positive_rate": svc_metrics["benign_false_positive_rate"],
        "threat_precision": svc_metrics["threat_precision"],
        "threat_recall": svc_metrics["threat_recall"],
        "per_class": svc_metrics["per_class"],
        "confusion_matrix": svc_metrics["confusion_matrix"],
        "confusion_matrix_labels": svc_metrics["confusion_matrix_labels"],
        "brier_score_threat": cal_metrics["brier_score_threat"],
        "expected_calibration_error_threat": cal_metrics["expected_calibration_error_threat"],
        "risk_distribution": svc_metrics["risk_distribution"],
        "risk_thresholds": svc_metrics["risk_thresholds"],
        "latency_ms_mean": svc_metrics["latency_ms_mean"],
        "latency_ms_std": svc_metrics["latency_ms_std"],
        "latency_ms_per_sample": svc_metrics["latency_ms_per_sample"],
    }


def _decision_summary(
    uncal: dict[str, Any],
    cal: dict[str, Any],
) -> dict[str, Any]:
    lower_fpr = (
        CANDIDATE_UNCALIBRATED
        if uncal["benign_false_positive_rate"] < cal["benign_false_positive_rate"]
        else CANDIDATE_ISOTONIC
    )
    if uncal["benign_false_positive_rate"] == cal["benign_false_positive_rate"]:
        lower_fpr = "tie"

    higher_threat_recall = (
        CANDIDATE_UNCALIBRATED
        if uncal["threat_recall"] > cal["threat_recall"]
        else CANDIDATE_ISOTONIC
    )
    if uncal["threat_recall"] == cal["threat_recall"]:
        higher_threat_recall = "tie"

    better_macro_f1 = (
        CANDIDATE_UNCALIBRATED
        if uncal["macro_f1"] > cal["macro_f1"]
        else CANDIDATE_ISOTONIC
    )
    if uncal["macro_f1"] == cal["macro_f1"]:
        better_macro_f1 = "tie"

    uncal_cal_score = uncal["brier_score_threat"] + uncal["expected_calibration_error_threat"]
    cal_cal_score = cal["brier_score_threat"] + cal["expected_calibration_error_threat"]
    better_calibration = (
        CANDIDATE_UNCALIBRATED if uncal_cal_score < cal_cal_score else CANDIDATE_ISOTONIC
    )
    if np.isclose(uncal_cal_score, cal_cal_score):
        better_calibration = "tie"

    # Phase 5 recommendation: prioritize cybersecurity (lower benign FPR, then threat recall).
    if lower_fpr == CANDIDATE_UNCALIBRATED and higher_threat_recall != CANDIDATE_ISOTONIC:
        phase5_choice = CANDIDATE_UNCALIBRATED
        phase5_reason = (
            "Uncalibrated has lower locked-test benign FPR without sacrificing threat recall."
        )
    elif lower_fpr == CANDIDATE_ISOTONIC and higher_threat_recall == CANDIDATE_ISOTONIC:
        phase5_choice = CANDIDATE_ISOTONIC
        phase5_reason = (
            "Calibrated candidate wins on both benign FPR and threat recall on locked test."
        )
    elif lower_fpr == CANDIDATE_UNCALIBRATED:
        phase5_choice = CANDIDATE_UNCALIBRATED
        phase5_reason = (
            "Uncalibrated has materially lower benign FPR; threat recall tradeoff favors "
            "fewer false alarms for acoustic cybersecurity."
        )
    elif higher_threat_recall == CANDIDATE_ISOTONIC and better_calibration == CANDIDATE_ISOTONIC:
        phase5_choice = CANDIDATE_ISOTONIC
        phase5_reason = (
            "Calibrated candidate improves threat recall and probability quality despite "
            "higher benign FPR."
        )
    else:
        phase5_choice = "inconclusive"
        phase5_reason = (
            "Locked-test metrics split across benign FPR, threat recall, and calibration "
            "quality without a clear cybersecurity winner."
        )

    fpr_delta = cal["benign_false_positive_rate"] - uncal["benign_false_positive_rate"]
    recall_delta = cal["threat_recall"] - uncal["threat_recall"]
    security_tradeoff = (
        f"Isotonic calibration changes locked-test benign FPR by {fpr_delta:+.4f} "
        f"and threat recall by {recall_delta:+.4f}. "
    )
    if fpr_delta > 0 and recall_delta > 0:
        security_tradeoff += (
            "Calibration increases both false alarms on benign audio and threat detection — "
            "a mixed but FPR-hostile shift for deployment."
        )
    elif fpr_delta > 0 and recall_delta <= 0:
        security_tradeoff += (
            "Calibration raises benign FPR without improving threat recall — "
            "strictly worse for false-alarm-sensitive cybersecurity use."
        )
    elif fpr_delta <= 0 and recall_delta > 0:
        security_tradeoff += (
            "Calibration lowers benign FPR while improving threat recall — "
            "favorable security tradeoff on locked test."
        )
    else:
        security_tradeoff += (
            "Calibration lowers or preserves benign FPR at the cost of threat recall."
        )

    if phase5_choice == CANDIDATE_UNCALIBRATED:
        recommendation = "UNCALIBRATED"
    elif phase5_choice == CANDIDATE_ISOTONIC:
        recommendation = "CALIBRATED"
    else:
        recommendation = "INCONCLUSIVE"

    return {
        "1_lower_benign_fpr": lower_fpr,
        "2_higher_threat_recall": higher_threat_recall,
        "3_better_macro_f1": better_macro_f1,
        "4_better_probability_calibration": better_calibration,
        "5_phase5_candidate": phase5_choice,
        "5_phase5_reason": phase5_reason,
        "6_security_tradeoff": security_tradeoff,
        "recommendation": recommendation,
        "recommendation_explanation": phase5_reason,
    }


def _write_markdown(result: dict[str, Any]) -> None:
    uncal = result["candidates"][CANDIDATE_UNCALIBRATED]["locked_test_metrics"]
    cal = result["candidates"][CANDIDATE_ISOTONIC]["locked_test_metrics"]
    decision = result["decision"]

    lines = [
        "# AI/ML Final Model Comparison",
        "",
        f"Generated: {result['generated_at_utc']}",
        "",
        "## Methodology",
        "",
        "Apples-to-apples comparison of two **frozen** candidates on the same locked test:",
        "",
        "| Stage | Partition | Purpose |",
        "|-------|-----------|---------|",
        "| Model fit | `model_train` (1748) | Train `linear_svm_balanced` once |",
        "| Calibration fit | `cal_dev` (289) | Fit uncalibrated risk thresholds OR isotonic calibrator |",
        "| Final evaluation | `locked_test` (463) | **One** evaluation per candidate — no tuning |",
        "",
        f"- Frozen split manifest: `{result['frozen_split_manifest']}`",
        f"- Dataset: `{DATASET_VERSION}` / `{GENERATOR_VERSION}`",
        "- Locked test was **not** used for model fitting or calibration fitting.",
        "- **Not production-ready.** Backend remains blocked.",
        "",
        "## Locked-test metrics",
        "",
        "| Metric | Uncalibrated | Isotonic calibrated |",
        "|--------|--------------|---------------------|",
    ]

    rows = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Benign precision", "benign_precision"),
        ("Benign recall", "benign_recall"),
        ("Benign FPR", "benign_false_positive_rate"),
        ("Threat precision", "threat_precision"),
        ("Threat recall", "threat_recall"),
        ("Brier (threat)", "brier_score_threat"),
        ("ECE (threat)", "expected_calibration_error_threat"),
        ("Latency ms/sample", "latency_ms_per_sample"),
    ]
    for label, key in rows:
        lines.append(
            f"| {label} | {uncal[key]:.4f} | {cal[key]:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Risk distribution (locked test)",
            "",
            "| Level | Uncalibrated | Isotonic calibrated |",
            "|-------|--------------|---------------------|",
        ]
    )
    for level in ("LOW", "MEDIUM", "HIGH"):
        lines.append(
            f"| {level} | {uncal['risk_distribution'][level]} | "
            f"{cal['risk_distribution'][level]} |"
        )

    lines.extend(["", "### Per-class metrics (locked test)", ""])
    for name in ("benign", "fsk", "ook", "chirp", "tone"):
        u = uncal["per_class"][name]
        c = cal["per_class"][name]
        lines.append(f"**{name}**")
        lines.append("")
        lines.append(
            f"| | Precision | Recall | F1 |"
        )
        lines.append(f"|---|---|---|---|")
        lines.append(
            f"| Uncalibrated | {u['precision']:.4f} | {u['recall']:.4f} | {u['f1']:.4f} |"
        )
        lines.append(
            f"| Isotonic | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |"
        )
        lines.append("")

    lines.extend(
        [
            "## Decision questions",
            "",
            f"1. **Which candidate has lower benign FPR?** `{decision['1_lower_benign_fpr']}`",
            f"2. **Which has higher threat recall?** `{decision['2_higher_threat_recall']}`",
            f"3. **Which has better macro F1?** `{decision['3_better_macro_f1']}`",
            f"4. **Which has better probability calibration?** `{decision['4_better_probability_calibration']}` "
            "(lower Brier + ECE combined)",
            f"5. **Which candidate should Phase 5 use?** `{decision['5_phase5_candidate']}` — "
            f"{decision['5_phase5_reason']}",
            f"6. **Security tradeoff:** {decision['6_security_tradeoff']}",
            "",
            "## RECOMMENDATION",
            "",
            f"**{decision['recommendation']}**",
            "",
            decision["recommendation_explanation"],
            "",
            "This comparison does not establish production readiness. "
            "Performance thresholds require team/project sign-off before backend integration.",
        ]
    )

    COMPARISON_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_final_model_comparison() -> dict[str, Any]:
    manifest = load_frozen_manifest()
    if manifest.get("status") != "FROZEN":
        raise ValueError("Split manifest is not frozen")

    dataset = load_dataset_v2(CLEAN_DATASET_DIR)
    model_train_idx = np.array(manifest["model_train_idx"], dtype=int)
    cal_dev_idx = np.array(manifest["cal_dev_idx"], dtype=int)
    locked_test_idx = np.array(manifest["locked_test_idx"], dtype=int)

    X_model_train = dataset.features[model_train_idx]
    y_model_train = dataset.labels[model_train_idx]
    X_cal = dataset.features[cal_dev_idx]
    y_cal = dataset.labels[cal_dev_idx]
    X_locked = dataset.features[locked_test_idx]
    y_locked = dataset.labels[locked_test_idx]
    locked_ids = dataset.metadata.iloc[locked_test_idx]["sample_id"].tolist()

    # --- Freeze shared model on model_train only ---
    model = _create_model()
    model.fit(X_model_train, y_model_train)
    model_train_metrics = evaluate_model(
        model, X_model_train, y_model_train, model_name=SELECTED_MODEL_NAME
    )
    phase3_meta = {
        "model_name": SELECTED_MODEL_NAME,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "feature_count": 32,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "predicted_classes_order": model_train_metrics["predicted_classes_order"],
    }

    # --- Candidate A: uncalibrated (thresholds fit on cal_dev) ---
    uncal_adapter = UncalibratedRiskAdapter(model, phase3_meta)
    uncal_adapter.calibrate(X_cal, y_cal)
    uncal_service = _build_service(model, uncal_adapter, "none")

    # --- Candidate B: isotonic RiskCalibrator (fit on cal_dev) ---
    iso_calibrator = RiskCalibrator(model, phase3_meta)
    iso_calibrator.calibrate(X_cal, y_cal)
    iso_service = _build_service(model, iso_calibrator, "isotonic")

    # --- ONE locked-test evaluation per candidate (no further changes) ---
    uncal_locked = _locked_test_metrics(
        uncal_service,
        uncal_adapter,
        X_locked,
        y_locked,
        locked_ids,
        candidate=CANDIDATE_UNCALIBRATED,
    )
    iso_locked = _locked_test_metrics(
        iso_service,
        iso_calibrator,
        X_locked,
        y_locked,
        locked_ids,
        candidate=CANDIDATE_ISOTONIC,
    )

    decision = _decision_summary(uncal_locked, iso_locked)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "model_fit_partition": "model_train",
            "calibration_fit_partition": "cal_dev",
            "evaluation_partition": "locked_test",
            "locked_test_used_for_selection": False,
            "shared_model": SELECTED_MODEL_NAME,
            "candidates": [CANDIDATE_UNCALIBRATED, CANDIDATE_ISOTONIC],
        },
        "frozen_split_manifest": str(FROZEN_SPLIT_MANIFEST_PATH),
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "partition_sizes": manifest["partition_sizes"],
        "model_train_metrics": {
            k: model_train_metrics[k]
            for k in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "benign_false_positive_rate",
                "benign_recall",
                "threat_recall",
            )
        },
        "candidates": {
            CANDIDATE_UNCALIBRATED: {
                "calibration_method": "none",
                "locked_test_metrics": uncal_locked,
            },
            CANDIDATE_ISOTONIC: {
                "calibration_method": "isotonic",
                "locked_test_metrics": iso_locked,
            },
        },
        "decision": decision,
        "production_readiness": (
            "NOT PRODUCTION READY — synthetic clean dataset only; "
            "team sign-off required for performance thresholds."
        ),
    }

    COMPARISON_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _write_markdown(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Final locked-test comparison: uncalibrated vs isotonic calibrated"
    )
    args = parser.parse_args()
    result = run_final_model_comparison()
    print(
        json.dumps(
            {
                "recommendation": result["decision"]["recommendation"],
                "uncalibrated_fpr": result["candidates"][CANDIDATE_UNCALIBRATED][
                    "locked_test_metrics"
                ]["benign_false_positive_rate"],
                "calibrated_fpr": result["candidates"][CANDIDATE_ISOTONIC][
                    "locked_test_metrics"
                ]["benign_false_positive_rate"],
                "uncalibrated_threat_recall": result["candidates"][CANDIDATE_UNCALIBRATED][
                    "locked_test_metrics"
                ]["threat_recall"],
                "calibrated_threat_recall": result["candidates"][CANDIDATE_ISOTONIC][
                    "locked_test_metrics"
                ]["threat_recall"],
                "output_json": str(COMPARISON_JSON_PATH),
                "output_md": str(COMPARISON_MD_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
