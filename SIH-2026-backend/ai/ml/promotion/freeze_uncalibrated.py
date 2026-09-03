"""Freeze the final uncalibrated linear_svm_balanced AI/ML pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.investigation.model_experiments import _make_models
from ml.phase3.config import CLASS_NAMES, FEATURE_NAMES
from ml.phase3.data import load_dataset_v2
from ml.phase3.evaluation import evaluate_model
from ml.phase3.models import assert_probability_shape, get_model_classes
from ml.phase4.evaluation import evaluate as evaluate_risk
from ml.phase5.config import (
    ARTIFACT_VERSION,
    EXPECTED_CLEAN_MODEL,
    MODEL_CALIBRATION,
    PHASE3_CLEAN_METADATA_PATH,
    PHASE3_CLEAN_MODEL_PATH,
    PHASE3_CLEAN_OUTPUT_DIR,
    PHASE4_CLEAN_METADATA_PATH,
    PHASE4_CLEAN_OUTPUT_DIR,
    PHASE5_CLEAN_EVAL_PATH,
    PHASE5_CLEAN_METADATA_PATH,
    PHASE5_CLEAN_OUTPUT_DIR,
    PHASE5_CLEAN_RISK_POLICY_PATH,
    PHASE5_CLEAN_SERVICE_PATH,
)
from ml.phase5.evaluation import evaluate_service
from ml.phase5.service import InferenceService
from ml.promotion.config_clean import (
    CLEAN_DATASET_DIR,
    DATASET_VERSION,
    FROZEN_SPLIT_MANIFEST_PATH,
    GENERATOR_VERSION,
    LEGACY_PHASE3_DIR,
    LEGACY_PHASE4_DIR,
    LEGACY_PHASE5_DIR,
    LEGACY_MARKER,
    SELECTED_MODEL_NAME,
    SPLIT_SEED,
)
from ml.promotion.splits import load_frozen_manifest
from ml.promotion.uncalibrated_adapter import UncalibratedRiskAdapter

CALIBRATION_STATUS = "NOT_USED_IN_FINAL_INFERENCE"
CALIBRATION_STATUS_REASON = (
    "Uncalibrated Linear SVM achieved substantially lower benign FPR, higher macro F1, "
    "better Brier/ECE, better accuracy, and lower latency on the locked test. Isotonic "
    "improved threat recall by only 0.68 pp while increasing benign FPR by 12.9 pp."
)

RISK_LEVEL_SEMANTICS = (
    "risk_level is derived from raw model threat_score (= sum of predict_proba mass on "
    "threat classes fsk/ook/chirp/tone). Thresholds are fit on cal_dev benign samples "
    "only to target HIGH<=5% and MEDIUM+<=15% benign band FPR on that partition. "
    "This is NOT isotonic calibration and does NOT imply calibrated probability semantics."
)

HANDOFF_PATH = Path(__file__).resolve().parents[2] / "docs" / "AI_ML_FINAL_HANDOFF.md"


def _extract_model_params(model: Any) -> dict[str, Any]:
    if not hasattr(model, "get_params"):
        return {}
    raw = model.get_params(deep=False)
    return _json_safe(raw)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _verify_model(model: Any, X_sample: np.ndarray) -> dict[str, Any]:
    assert_probability_shape(model, X_sample)
    classes = get_model_classes(model)
    coef = model.named_steps["clf"].coef_
    checks = {
        "estimator_type": type(model).__name__,
        "is_fitted": hasattr(model.named_steps["clf"], "coef_"),
        "feature_count": int(coef.shape[1]),
        "class_count": len(classes),
        "classes": classes,
        "nan_in_coef": bool(np.isnan(coef).any()),
        "inf_in_coef": bool(np.isinf(coef).any()),
    }
    if checks["feature_count"] != 32 or checks["class_count"] != 5:
        raise ValueError(f"Model shape mismatch: {checks}")
    if checks["nan_in_coef"] or checks["inf_in_coef"]:
        raise ValueError("Model coefficients contain NaN/Inf")
    return checks


def _reproducibility(service: InferenceService, X: np.ndarray) -> dict[str, Any]:
    a = service.predict_batch(X)
    b = service.predict_batch(X)
    proba_a = service.predict_proba(X)
    proba_b = service.predict_proba(X)
    reloaded = InferenceService.load(PHASE5_CLEAN_SERVICE_PATH)
    c = reloaded.predict_batch(X)

    pred_match = a == b
    proba_match = np.allclose(proba_a, proba_b, atol=1e-12)
    reload_match = a == c
    return {
        "double_inference_predictions_match": pred_match,
        "double_inference_probabilities_match": proba_match,
        "artifact_reload_predictions_match": reload_match,
        "passed": pred_match and proba_match and reload_match,
    }


def run_freeze() -> dict[str, Any]:
    manifest = load_frozen_manifest()
    comparison_path = PHASE5_CLEAN_OUTPUT_DIR / "final_model_comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        locked_ref = comparison["candidates"]["uncalibrated"]["locked_test_metrics"]
    else:
        locked_ref = None

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

    models = _make_models()
    model = models[SELECTED_MODEL_NAME]
    model.fit(X_model_train, y_model_train)
    model_checks = _verify_model(model, X_model_train[:5])
    train_metrics = evaluate_model(
        model, X_model_train, y_model_train, model_name=SELECTED_MODEL_NAME
    )

    phase3_meta = {
        "artifact_lineage": "clean_uncalibrated_freeze_v1",
        "artifact_version": ARTIFACT_VERSION,
        "legacy_marker": LEGACY_MARKER,
        "model_name": EXPECTED_CLEAN_MODEL,
        "model_type": "sklearn.pipeline.Pipeline",
        "model_parameters": _extract_model_params(model),
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "random_seed": SPLIT_SEED,
        "split_policy": manifest["split_policy"],
        "feature_count": 32,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "training_sample_count": int(len(model_train_idx)),
        "cal_dev_sample_count": int(len(cal_dev_idx)),
        "test_sample_count": int(len(locked_test_idx)),
        "model_calibration": MODEL_CALIBRATION,
        "calibration_status": CALIBRATION_STATUS,
        "model_fit_partition": "model_train",
        "frozen_split_manifest": str(FROZEN_SPLIT_MANIFEST_PATH),
        "model_verification": model_checks,
        "train_partition_metrics": {
            k: train_metrics[k]
            for k in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "benign_false_positive_rate",
                "benign_recall",
                "threat_recall",
            )
        },
        "predicted_classes_order": train_metrics["predicted_classes_order"],
        "ai_model_status": "TECHNICALLY_FROZEN_READY_FOR_INTEGRATION_REVIEW",
        "production_validation": "NOT_OTA_VALIDATED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    PHASE3_CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PHASE3_CLEAN_MODEL_PATH)
    with PHASE3_CLEAN_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(phase3_meta, fh, indent=2)

    comparison_report = {
        "selected_model": SELECTED_MODEL_NAME,
        "model_calibration": MODEL_CALIBRATION,
        "artifact_version": ARTIFACT_VERSION,
        "reference_locked_test_metrics": locked_ref,
        "train_metrics": phase3_meta["train_partition_metrics"],
    }
    with (PHASE3_CLEAN_OUTPUT_DIR / "model_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(comparison_report, fh, indent=2)

    risk_policy = UncalibratedRiskAdapter(model, phase3_meta)
    risk_policy.calibrate(X_cal, y_cal)
    risk_policy.save(PHASE5_CLEAN_RISK_POLICY_PATH)

    PHASE4_CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase4_meta = {
        "artifact_lineage": "clean_uncalibrated_freeze_v1",
        "calibration_status": CALIBRATION_STATUS,
        "calibration_status_reason": CALIBRATION_STATUS_REASON,
        "model_calibration": MODEL_CALIBRATION,
        "isotonic_implementation_preserved_for_research": True,
        "isotonic_artifact_path_if_present": str(PHASE4_CLEAN_OUTPUT_DIR / "risk_calibrator.joblib"),
        "final_inference_uses_isotonic": False,
        "phase3_model_name": EXPECTED_CLEAN_MODEL,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with PHASE4_CLEAN_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(phase4_meta, fh, indent=2)

    inference_meta = {
        "artifact_lineage": "clean_uncalibrated_freeze_v1",
        "artifact_version": ARTIFACT_VERSION,
        "phase3_model_name": EXPECTED_CLEAN_MODEL,
        "model_type": "sklearn.pipeline.Pipeline",
        "model_parameters": phase3_meta["model_parameters"],
        "model_calibration": MODEL_CALIBRATION,
        "calibration_status": CALIBRATION_STATUS,
        "calibration_status_reason": CALIBRATION_STATUS_REASON,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "random_seed": SPLIT_SEED,
        "split_policy": manifest["split_policy"],
        "feature_count": 32,
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "training_sample_count": int(len(model_train_idx)),
        "cal_dev_sample_count": int(len(cal_dev_idx)),
        "test_sample_count": int(len(locked_test_idx)),
        "phase3_artifact_path": str(PHASE3_CLEAN_MODEL_PATH),
        "risk_policy_path": str(PHASE5_CLEAN_RISK_POLICY_PATH),
        "legacy_phase3_path": str(LEGACY_PHASE3_DIR / "selected_model.joblib"),
        "legacy_phase4_calibrator_path": str(LEGACY_PHASE4_DIR / "risk_calibrator.joblib"),
        "uses_legacy_artifacts": False,
        "loads_legacy_phase4_calibrator": False,
        "risk_thresholds": risk_policy.get_thresholds(),
        "risk_level_semantics": RISK_LEVEL_SEMANTICS,
        "inference_input_contract": {
            "shape": "(32,)",
            "dtype": "float64",
            "finite_required": True,
            "feature_order": FEATURE_NAMES,
        },
        "inference_output_contract": [
            "predicted_class_id",
            "predicted_class",
            "confidence",
            "class_probabilities",
            "threat_score",
            "risk_level",
            "model_calibration",
        ],
        "ai_model_status": "TECHNICALLY_FROZEN_READY_FOR_INTEGRATION_REVIEW",
        "production_validation": "NOT_OTA_VALIDATED",
        "backend_modified": False,
        "frontend_modified": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    service = InferenceService(
        model=model,
        calibrator=risk_policy,
        phase3_metadata=phase3_meta,
        phase4_metadata=inference_meta,
        model_calibration=MODEL_CALIBRATION,
    )

    locked_metrics = evaluate_service(
        service,
        X_locked,
        y_locked,
        sample_ids=locked_ids,
        partition="locked_test_final",
    )
    risk_metrics = evaluate_risk(risk_policy, X_locked, y_locked, partition="locked_test_final")
    locked_summary = {
        k: locked_metrics[k]
        for k in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "benign_recall",
            "benign_false_positive_rate",
            "threat_precision",
            "threat_recall",
            "per_class",
            "confusion_matrix",
            "risk_distribution",
            "latency_ms_mean",
            "latency_ms_per_sample",
        )
    }
    locked_summary["benign_precision"] = risk_metrics["per_class"]["benign"]["precision"]
    locked_summary["brier_score_threat"] = risk_metrics["brier_score_threat"]
    locked_summary["expected_calibration_error_threat"] = risk_metrics[
        "expected_calibration_error_threat"
    ]
    inference_meta["locked_test_metrics"] = locked_summary

    PHASE5_CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    service.save(PHASE5_CLEAN_SERVICE_PATH)
    repro = _reproducibility(service, X_locked[: min(50, len(X_locked))])
    inference_meta["reproducibility"] = repro

    with PHASE5_CLEAN_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(inference_meta, fh, indent=2)
    with PHASE5_CLEAN_EVAL_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "evaluation_partition": "locked_test",
                "metrics": locked_summary,
                "reference_from_final_model_comparison": locked_ref,
            },
            fh,
            indent=2,
        )

    for legacy_dir in (LEGACY_PHASE3_DIR, LEGACY_PHASE4_DIR, LEGACY_PHASE5_DIR):
        marker = legacy_dir / "LEGACY_README.txt"
        if legacy_dir.exists() and not marker.exists():
            marker.write_text(
                f"{LEGACY_MARKER}\n\nPreserved for history. Not used by frozen clean inference.\n",
                encoding="utf-8",
            )

    _write_handoff(inference_meta, locked_summary, repro)

    return {
        "status": "FINAL MODEL FROZEN",
        "model": EXPECTED_CLEAN_MODEL,
        "calibration": MODEL_CALIBRATION.upper(),
        "dataset": DATASET_VERSION,
        "artifact": str(PHASE5_CLEAN_SERVICE_PATH),
        "locked_test_metrics": locked_summary,
        "reproducibility": repro,
    }


def _write_handoff(
    inference_meta: dict[str, Any],
    locked: dict[str, Any],
    repro: dict[str, Any],
) -> None:
    lines = [
        "# AI/ML Final Handoff",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## AI model status",
        "",
        "**TECHNICALLY FROZEN / READY FOR INTEGRATION REVIEW**",
        "",
        "NOT production validated. NOT OTA validated. Backend remains BLOCKED until team lead "
        "approves performance criteria.",
        "",
        "## 1. Final model",
        "",
        f"- Model: `{EXPECTED_CLEAN_MODEL}` (sklearn Pipeline: StandardScaler + linear SVC)",
        f"- Calibration: `{MODEL_CALIBRATION}` (no isotonic in inference path)",
        "",
        "## 2. Dataset version",
        "",
        f"- `{DATASET_VERSION}` at `{CLEAN_DATASET_DIR}`",
        f"- Generator: `{GENERATOR_VERSION}`",
        "",
        "## 3. Feature contract",
        "",
        "- Exactly **32 finite numeric features** in Phase-1 order (see `feature_names` in metadata)",
        "",
        "## 4. Class mapping",
        "",
        "| ID | Class |",
        "|----|-------|",
    ]
    for cid, name in CLASS_NAMES.items():
        lines.append(f"| {cid} | {name} |")

    lines.extend(
        [
            "",
            "## 5. Calibration decision",
            "",
            f"- `calibration_status`: `{CALIBRATION_STATUS}`",
            f"- Reason: {CALIBRATION_STATUS_REASON}",
            "- Phase 4 isotonic implementation preserved for research; **not loaded** by Phase 5 clean.",
            "",
            "## 6. Final locked-test metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {locked['accuracy']:.4f} |",
            f"| Macro F1 | {locked['macro_f1']:.4f} |",
            f"| Weighted F1 | {locked['weighted_f1']:.4f} |",
            f"| Benign precision | {locked['benign_precision']:.4f} |",
            f"| Benign recall | {locked['benign_recall']:.4f} |",
            f"| Benign FPR | {locked['benign_false_positive_rate']:.4f} |",
            f"| Threat precision | {locked['threat_precision']:.4f} |",
            f"| Threat recall | {locked['threat_recall']:.4f} |",
            f"| Brier | {locked['brier_score_threat']:.4f} |",
            f"| ECE | {locked['expected_calibration_error_threat']:.4f} |",
            f"| Latency ms/sample | {locked['latency_ms_per_sample']:.4f} |",
            "",
            "## 7. Artifact paths",
            "",
            f"- Phase 3 clean model: `{PHASE3_CLEAN_MODEL_PATH}`",
            f"- Phase 4 clean metadata (calibration not used): `{PHASE4_CLEAN_METADATA_PATH}`",
            f"- Phase 5 risk policy: `{PHASE5_CLEAN_RISK_POLICY_PATH}`",
            f"- Phase 5 inference service: `{PHASE5_CLEAN_SERVICE_PATH}`",
            f"- Phase 5 metadata: `{PHASE5_CLEAN_METADATA_PATH}`",
            f"- Locked test evaluation: `{PHASE5_CLEAN_EVAL_PATH}`",
            f"- Frozen split: `{FROZEN_SPLIT_MANIFEST_PATH}`",
            "",
            "## 8. Inference input contract",
            "",
            "- Shape: `(32,)` or batch `(N, 32)`",
            "- dtype: float64 after coercion",
            "- All values finite (NaN/Inf rejected)",
            "- Feature order: Phase-1 `FEATURE_NAMES`",
            "",
            "## 9. Inference output contract",
            "",
            "Minimum fields per prediction:",
            "",
            "- `predicted_class_id`, `predicted_class`, `confidence`",
            "- `class_probabilities` (raw model probabilities; sum to 1)",
            "- `threat_score` (sum of threat-class probabilities; **not calibrated**)",
            "- `risk_level` (`LOW` / `MEDIUM` / `HIGH`)",
            "- `model_calibration`: `\"uncalibrated\"`",
            "",
            "Load service:",
            "",
            "```python",
            "from ml.phase5.service import InferenceService",
            "service = InferenceService.from_clean_artifacts()",
            "# or: service = InferenceService.load_frozen()",
            "result = service.predict(features_32)",
            "```",
            "",
            "## 10. Risk-level semantics",
            "",
            RISK_LEVEL_SEMANTICS,
            "",
            "## 11. Known limitations",
            "",
            "- Synthetic Dataset V2 clean only",
            "- No over-the-air validation",
            "- Chirp near-perfect separability is a synthetic artifact",
            "- Performance thresholds require team sign-off",
            "",
            "## 12. Backend integration instructions",
            "",
            "1. Do **not** use legacy `output/phase3/` or `output/phase4/` artifacts.",
            "2. Load `InferenceService.from_clean_artifacts()` or `InferenceService.load_frozen()`.",
            "3. Pass exactly 32 Phase-1 features per inference call.",
            "4. Treat outputs as synthetic-data validated only.",
            "5. Complete OTA validation before production deployment.",
            "",
            "## 13. OTA validation requirement",
            "",
            "Mandatory before production: over-the-air capture on target hardware, with team-approved "
            "FPR/recall thresholds.",
            "",
            "## 14. Backend / frontend",
            "",
            "**Backend: NOT modified.** **Frontend: NOT modified.**",
            "",
            "## Reproducibility",
            "",
            f"- Double inference match: {repro['double_inference_predictions_match']}",
            f"- Reload match: {repro['artifact_reload_predictions_match']}",
        ]
    )
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze final uncalibrated clean AI pipeline")
    args = parser.parse_args()
    result = run_freeze()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
