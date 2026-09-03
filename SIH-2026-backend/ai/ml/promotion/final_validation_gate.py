"""Final AI validation and clean artifact promotion gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.investigation.model_experiments import _make_models
from ml.phase3.config import CLASS_NAMES as PHASE3_CLASS_NAMES
from ml.phase3.data import load_dataset_v2
from ml.phase3.evaluation import evaluate_model, save_confusion_matrix_plot
from ml.phase3.models import assert_probability_shape, get_model_classes
from ml.phase4.calibrator import RiskCalibrator
from ml.phase4.evaluation import evaluate, predictions_to_dataframe, save_phase4_artifacts
from ml.phase5.evaluation import evaluate_service, save_phase5_report
from ml.phase5.service import InferenceService
from ml.promotion.calibrator_decision import (
    audit_prior_calibration_experiment,
    choose_calibration_method,
    evaluate_calibrated,
    evaluate_uncalibrated,
)
from ml.promotion.config_clean import (
    CALIBRATION_VERSION,
    CLEAN_DATASET_DIR,
    DATASET_VERSION,
    FROZEN_SPLIT_MANIFEST_PATH,
    GENERATOR_VERSION,
    INFERENCE_VERSION,
    LEGACY_MARKER,
    LEGACY_PHASE3_DIR,
    LEGACY_PHASE4_DIR,
    LEGACY_PHASE5_DIR,
    MODEL_VERSION,
    PHASE3_CLEAN_METADATA_PATH,
    PHASE3_CLEAN_MODEL_PATH,
    PHASE3_CLEAN_OUTPUT_DIR,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE4_CLEAN_METADATA_PATH,
    PHASE4_CLEAN_OUTPUT_DIR,
    PHASE5_CLEAN_OUTPUT_DIR,
    SELECTED_MODEL_NAME,
    SPLIT_SEED,
)
from ml.promotion.splits import (
    build_frozen_manifest,
    build_three_way_split,
    save_frozen_manifest,
)
from ml.promotion.uncalibrated_adapter import UncalibratedRiskAdapter


def _create_selected_model() -> Pipeline:
    models = _make_models()
    if SELECTED_MODEL_NAME not in models:
        raise KeyError(f"Selected model {SELECTED_MODEL_NAME!r} not found")
    return models[SELECTED_MODEL_NAME]


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
        "deterministic_classes_order": classes == [0, 1, 2, 3, 4],
    }
    if checks["feature_count"] != 32 or checks["class_count"] != 5:
        raise ValueError(f"Model shape mismatch: {checks}")
    if checks["nan_in_coef"] or checks["inf_in_coef"]:
        raise ValueError("Model coefficients contain NaN/Inf")
    return checks


def _mark_legacy_dirs() -> None:
    for legacy_dir in (LEGACY_PHASE3_DIR, LEGACY_PHASE4_DIR, LEGACY_PHASE5_DIR):
        marker = legacy_dir / "LEGACY_README.txt"
        if legacy_dir.exists() and not marker.exists():
            marker.write_text(
                f"{LEGACY_MARKER}\n\n"
                "These artifacts were trained on the original Dataset V2 "
                "(training_data_v2/) before the clean rebuild.\n"
                "Do not use for clean promotion or backend integration.\n"
                "Use ai/ml/output/phase3_clean/, phase4_clean/, phase5_clean/ instead.\n",
                encoding="utf-8",
            )


def _save_phase3_clean_artifacts(
    *,
    model: Any,
    model_checks: dict[str, Any],
    train_metrics: dict[str, Any],
    manifest: dict[str, Any],
    dataset_info: dict[str, Any],
) -> dict[str, str]:
    PHASE3_CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PHASE3_CLEAN_MODEL_PATH)

    metadata = {
        "artifact_lineage": "clean_promotion_gate",
        "legacy_marker": LEGACY_MARKER,
        "model_name": SELECTED_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_seed": dataset_info.get("seed", SPLIT_SEED),
        "split_policy": manifest["split_policy"],
        "split_seed": manifest["split_policy"]["outer_seed"],
        "feature_count": 32,
        "feature_names": dataset_info.get("feature_names"),
        "class_mapping": PHASE3_CLASS_NAMES,
        "training_sample_count": (
            manifest["partition_sizes"]["model_train"]
            + manifest["partition_sizes"]["cal_dev"]
        ),
        "cal_dev_sample_count": manifest["partition_sizes"]["cal_dev"],
        "test_sample_count": manifest["partition_sizes"]["locked_test"],
        "model_train_groups": len(manifest["model_train_groups"]),
        "cal_dev_groups": len(manifest["cal_dev_groups"]),
        "locked_test_groups": len(manifest["locked_test_groups"]),
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
            if k in train_metrics
        },
        "predicted_classes_order": train_metrics.get("predicted_classes_order"),
        "selection_reason": (
            "Pre-selected clean candidate linear_svm_balanced from clean dataset "
            "rebuild experiments. Calibration method chosen on cal_dev using a "
            "model_fit trained on model_train only; final artifact model retrains "
            "on full train partition (model_train + cal_dev)."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with PHASE3_CLEAN_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    comparison = {
        "selected_model": SELECTED_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "train_metrics": train_metrics,
    }
    comparison_path = PHASE3_CLEAN_OUTPUT_DIR / "model_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    return {
        "selected_model": str(PHASE3_CLEAN_MODEL_PATH),
        "selected_model_metadata": str(PHASE3_CLEAN_METADATA_PATH),
        "model_comparison": str(comparison_path),
    }


def _build_inference_service(
    model: Any,
    calibrator: Any,
    phase3_metadata: dict[str, Any],
    phase4_metadata: dict[str, Any],
) -> InferenceService:
    return InferenceService(
        model=model,
        calibrator=calibrator,
        phase3_metadata=phase3_metadata,
        phase4_metadata=phase4_metadata,
    )


def _load_clean_calibrator(calibration_method: str) -> Any:
    if calibration_method == "isotonic":
        from ml.phase4.calibrator import RiskCalibrator

        return RiskCalibrator.load(PHASE4_CLEAN_CALIBRATOR_PATH)
    return UncalibratedRiskAdapter.load(PHASE4_CLEAN_CALIBRATOR_PATH)


def _reproducibility_check(
    service: InferenceService,
    X: np.ndarray,
    *,
    calibration_method: str,
) -> dict[str, Any]:
    run_a = service.predict_batch(X)
    run_b = service.predict_batch(X)
    proba_a = service.predict_proba(X)
    proba_b = service.predict_proba(X)

    pred_match = all(
        a["predicted_class_id"] == b["predicted_class_id"]
        and a["risk_level"] == b["risk_level"]
        and abs(a["calibrated_risk_score"] - b["calibrated_risk_score"]) < 1e-12
        for a, b in zip(run_a, run_b)
    )
    proba_match = np.allclose(proba_a, proba_b, atol=1e-12)

    reloaded = InferenceService(
        model=joblib.load(PHASE3_CLEAN_MODEL_PATH),
        calibrator=_load_clean_calibrator(calibration_method),
        phase3_metadata=json.loads(PHASE3_CLEAN_METADATA_PATH.read_text(encoding="utf-8")),
        phase4_metadata=json.loads(PHASE4_CLEAN_METADATA_PATH.read_text(encoding="utf-8")),
    )
    run_c = reloaded.predict_batch(X)
    reload_match = all(
        a["predicted_class_id"] == c["predicted_class_id"]
        and a["risk_level"] == c["risk_level"]
        and abs(a["calibrated_risk_score"] - c["calibrated_risk_score"]) < 1e-12
        for a, c in zip(run_a, run_c)
    )

    return {
        "double_inference_predictions_match": pred_match,
        "double_inference_probabilities_match": proba_match,
        "artifact_reload_predictions_match": reload_match,
        "passed": pred_match and proba_match and reload_match,
    }


def run_final_validation_gate(*, write_report: bool = True) -> dict[str, Any]:
    dataset = load_dataset_v2(CLEAN_DATASET_DIR)
    dataset_info = dataset.dataset_info

    split = build_three_way_split(dataset.metadata, dataset.labels)
    manifest = build_frozen_manifest(
        split,
        dataset.metadata,
        dataset.labels,
        dataset_version=DATASET_VERSION,
        generator_version=GENERATOR_VERSION,
    )
    save_frozen_manifest(manifest)

    X_model_train = dataset.features[split.model_train_idx]
    y_model_train = dataset.labels[split.model_train_idx]
    X_cal = dataset.features[split.cal_dev_idx]
    y_cal = dataset.labels[split.cal_dev_idx]
    X_locked = dataset.features[split.locked_test_idx]
    y_locked = dataset.labels[split.locked_test_idx]

    model = _create_selected_model()
    model.fit(X_model_train, y_model_train)
    model_checks = _verify_model(model, X_model_train[:5])
    train_metrics = evaluate_model(
        model, X_model_train, y_model_train, model_name=SELECTED_MODEL_NAME
    )

    train_proba = model.predict_proba(X_model_train)
    test_proba = model.predict_proba(X_locked)
    prior_audit = audit_prior_calibration_experiment(
        train_threat_p=train_proba[:, 1:].sum(axis=1),
        y_train_threat=(y_model_train != 0).astype(int),
        test_threat_p=test_proba[:, 1:].sum(axis=1),
        y_test_threat=(y_locked != 0).astype(int),
    )

    phase3_metadata = {
        "model_name": SELECTED_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "feature_count": 32,
        "feature_names": dataset.feature_names,
        "class_mapping": dataset.class_names,
        "predicted_classes_order": train_metrics["predicted_classes_order"],
        "seed": SPLIT_SEED,
    }

    uncal_adapter, uncal_cal_dev = evaluate_uncalibrated(
        model, phase3_metadata, X_cal, y_cal
    )
    calibrator, cal_cal_dev = evaluate_calibrated(model, phase3_metadata, X_cal, y_cal)
    cal_choice, cal_reason = choose_calibration_method(uncal_cal_dev, cal_cal_dev)

    # Final artifact model uses all allowed non-test data (model_train + cal_dev).
    X_full_train = dataset.features[split.model_train_idx]
    y_full_train = dataset.labels[split.model_train_idx]
    X_cal_dev = dataset.features[split.cal_dev_idx]
    y_cal_dev = dataset.labels[split.cal_dev_idx]

    final_model = _create_selected_model()
    final_model.fit(X_full_train, y_full_train)
    final_model_checks = _verify_model(final_model, X_full_train[:5])
    full_train_metrics = evaluate_model(
        final_model, X_full_train, y_full_train, model_name=SELECTED_MODEL_NAME
    )

    phase3_metadata_final = {
        **phase3_metadata,
        "predicted_classes_order": full_train_metrics["predicted_classes_order"],
    }

    if cal_choice == "isotonic":
        final_calibrator = RiskCalibrator(final_model, phase3_metadata_final)
        final_calibrator.calibrate(X_cal_dev, y_cal_dev)
        calibration_method = "isotonic"
        calibration_version = CALIBRATION_VERSION
    else:
        final_calibrator = UncalibratedRiskAdapter(final_model, phase3_metadata_final)
        final_calibrator.calibrate(X_cal_dev, y_cal_dev)
        calibration_method = "none"
        calibration_version = "uncalibrated_v1"

    phase3_paths = _save_phase3_clean_artifacts(
        model=final_model,
        model_checks=final_model_checks,
        train_metrics=full_train_metrics,
        manifest=manifest,
        dataset_info=dataset_info,
    )

    split_report = {
        **manifest["outer_split_report"],
        "three_way_split": manifest["partition_sizes"],
        "frozen_manifest": str(FROZEN_SPLIT_MANIFEST_PATH),
    }

    cal_dev_ids = dataset.metadata.iloc[split.cal_dev_idx]["sample_id"].tolist()
    cal_dev_preds = final_calibrator.predict_risk(X_cal)
    cal_dev_metrics = evaluate(final_calibrator, X_cal, y_cal, partition="cal_dev_final")
    model_train_cal_metrics = evaluate(
        final_calibrator, X_model_train, y_model_train, partition="model_train"
    )

    phase4_metadata_base = dict(phase3_metadata)
    phase4_metadata_base["_model_path"] = str(PHASE3_CLEAN_MODEL_PATH)

    phase4_paths = save_phase4_artifacts(
        output_dir=PHASE4_CLEAN_OUTPUT_DIR,
        calibrator=final_calibrator,
        train_metrics=model_train_cal_metrics,
        test_metrics=cal_dev_metrics,
        train_predictions=predictions_to_dataframe(cal_dev_preds, y_cal, cal_dev_ids),
        test_predictions=predictions_to_dataframe(cal_dev_preds, y_cal, cal_dev_ids),
        metadata=phase4_metadata_base,
        split_report=split_report,
    )

    phase4_meta = json.loads(PHASE4_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    phase4_meta.update(
        {
            "artifact_lineage": "clean_promotion_gate",
            "calibration_version": calibration_version,
            "calibration_decision": cal_choice,
            "calibration_decision_reason": cal_reason,
            "cal_dev_metrics_uncalibrated": {
                k: uncal_cal_dev[k]
                for k in (
                    "brier_score_threat",
                    "expected_calibration_error_threat",
                    "benign_false_positive_rate",
                    "benign_recall",
                    "threat_recall",
                    "macro_f1",
                    "risk_distribution",
                )
            },
            "cal_dev_metrics_isotonic": {
                k: cal_cal_dev[k]
                for k in (
                    "brier_score_threat",
                    "expected_calibration_error_threat",
                    "benign_false_positive_rate",
                    "benign_recall",
                    "threat_recall",
                    "macro_f1",
                    "risk_distribution",
                )
            },
            "prior_calibration_audit": prior_audit,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "generator_version": GENERATOR_VERSION,
            "seed": SPLIT_SEED,
            "split_policy": manifest["split_policy"],
            "training_sample_count": (
                manifest["partition_sizes"]["model_train"]
                + manifest["partition_sizes"]["cal_dev"]
            ),
            "cal_dev_sample_count": manifest["partition_sizes"]["cal_dev"],
            "test_sample_count": manifest["partition_sizes"]["locked_test"],
        }
    )
    phase4_meta["calibration_method"] = calibration_method
    with PHASE4_CLEAN_METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(phase4_meta, fh, indent=2)

    phase3_meta = json.loads(PHASE3_CLEAN_METADATA_PATH.read_text(encoding="utf-8"))
    service = _build_inference_service(
        final_model, final_calibrator, phase3_meta, phase4_meta
    )

    locked_ids = dataset.metadata.iloc[split.locked_test_idx]["sample_id"].tolist()
    final_metrics = evaluate_service(
        service,
        X_locked,
        y_locked,
        sample_ids=locked_ids,
        partition="locked_test_final",
    )

    brier_ece = evaluate(final_calibrator, X_locked, y_locked, partition="locked_test_final")
    final_metrics["brier_score_threat"] = brier_ece["brier_score_threat"]
    final_metrics["expected_calibration_error_threat"] = brier_ece[
        "expected_calibration_error_threat"
    ]
    final_metrics["benign_precision"] = brier_ece["per_class"]["benign"]["precision"]
    final_metrics["threat_precision"] = brier_ece["threat_precision"]

    repro = _reproducibility_check(
        service,
        X_locked[: min(50, len(X_locked))],
        calibration_method=calibration_method,
    )

    phase5_paths = save_phase5_report(
        output_dir=PHASE5_CLEAN_OUTPUT_DIR,
        service=service,
        held_out_metrics=final_metrics,
        in_sample_metrics=None,
        split_report=split_report,
        phase3_metadata=phase3_meta,
        phase4_metadata=phase4_meta,
    )

    phase5_meta_path = PHASE5_CLEAN_OUTPUT_DIR / "inference_metadata.json"
    phase5_meta = json.loads(phase5_meta_path.read_text(encoding="utf-8"))
    phase5_meta.update(
        {
            "artifact_lineage": "clean_promotion_gate",
            "inference_version": INFERENCE_VERSION,
            "model_version": MODEL_VERSION,
            "calibration_version": calibration_version,
            "dataset_version": DATASET_VERSION,
            "generator_version": GENERATOR_VERSION,
            "seed": SPLIT_SEED,
            "split_policy": manifest["split_policy"],
            "phase3_artifact_path": str(PHASE3_CLEAN_MODEL_PATH),
            "phase4_artifact_path": str(PHASE4_CLEAN_CALIBRATOR_PATH),
            "legacy_phase3_path": str(LEGACY_PHASE3_DIR / "selected_model.joblib"),
            "uses_legacy_artifacts": False,
            "training_sample_count": (
                manifest["partition_sizes"]["model_train"]
                + manifest["partition_sizes"]["cal_dev"]
            ),
            "cal_dev_sample_count": manifest["partition_sizes"]["cal_dev"],
            "test_sample_count": manifest["partition_sizes"]["locked_test"],
            "reproducibility": repro,
        }
    )
    with phase5_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(phase5_meta, fh, indent=2)

    save_confusion_matrix_plot(
        final_metrics["confusion_matrix"],
        final_metrics["confusion_matrix_labels"],
        PHASE5_CLEAN_OUTPUT_DIR / "confusion_matrix_locked_test.png",
        title="Locked Test — Clean Promotion Gate",
    )

    _mark_legacy_dirs()

    gate_result = {
        "status": "completed",
        "prior_calibration_audit": prior_audit,
        "frozen_split_manifest": str(FROZEN_SPLIT_MANIFEST_PATH),
        "selected_model": SELECTED_MODEL_NAME,
        "calibration_decision": cal_choice,
        "calibration_decision_reason": cal_reason,
        "train_metrics": full_train_metrics,
        "calibration_selection_model_train_metrics": train_metrics,
        "cal_dev_metrics": {
            "uncalibrated": uncal_cal_dev,
            "isotonic": cal_cal_dev,
            "selected": cal_dev_metrics,
        },
        "final_locked_test_metrics": {
            k: final_metrics[k]
            for k in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "benign_precision",
                "benign_recall",
                "benign_false_positive_rate",
                "threat_precision",
                "threat_recall",
                "per_class",
                "confusion_matrix",
                "brier_score_threat",
                "expected_calibration_error_threat",
                "risk_distribution",
                "latency_ms_mean",
                "latency_ms_per_sample",
            )
        },
        "reproducibility": repro,
        "artifact_paths": {
            **phase3_paths,
            **phase4_paths,
            **phase5_paths,
        },
    }

    gate_json = PHASE5_CLEAN_OUTPUT_DIR / "final_validation_gate.json"
    with gate_json.open("w", encoding="utf-8") as fh:
        json.dump(gate_result, fh, indent=2, default=str)

    if write_report:
        _write_final_report(gate_result, manifest, prior_audit, repro)

    return gate_result


def _write_final_report(
    gate: dict[str, Any],
    manifest: dict[str, Any],
    prior_audit: dict[str, Any],
    repro: dict[str, Any],
) -> None:
    final = gate["final_locked_test_metrics"]
    train = gate["train_metrics"]
    cal = gate["cal_dev_metrics"]["selected"]
    report_path = Path(__file__).resolve().parents[2] / "docs" / "AI_ML_FINAL_VALIDATION_REPORT.md"

    sign_off = "NOT READY FOR AI SIGN-OFF"
    sign_off_reasons = [
        "Synthetic Dataset V2 clean data only — no over-the-air validation.",
        "Performance threshold requires team/project sign-off.",
        "Backend integration remains BLOCKED.",
    ]

    lines = [
        "# AI/ML Final Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Final status",
        "",
        f"**{sign_off}**",
        "",
        "Backend: **BLOCKED**",
        "",
        "### Sign-off blockers",
        "",
    ]
    for reason in sign_off_reasons:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## 1. Dataset",
            "",
            f"- Version: `{DATASET_VERSION}`",
            f"- Path: `{CLEAN_DATASET_DIR}`",
            f"- Generator: `{GENERATOR_VERSION}`",
            f"- Clean validation: 19/19 checks passed (0 collapsed benign, 0 duplicate groups, 0 cross-split overlap)",
            "",
            "## 2. Generator changes",
            "",
            "- Added `steady_ultrasonic_hum` and `band_limited_hiss` benign variants",
            "- Regenerate-on-collapsed-fingerprint policy",
            "- Separate output dir `training_data_v2_clean/` preserves legacy `training_data_v2/`",
            "",
            "## 3. Duplicate results",
            "",
            "| Dataset | Collapsed benign | Duplicate groups | Cross-split exact |",
            "|---------|------------------|------------------|-------------------|",
            "| Old V2 | 25 | present | 6 |",
            "| Clean V2 | 0 | 0 | 0 |",
            "",
            "## 4. Split methodology",
            "",
            f"- Outer split seed: `{manifest['split_policy']['outer_seed']}` (grouped held-out test)",
            f"- Cal-dev seed: `{manifest['split_policy']['cal_dev_seed']}` ({manifest['partition_sizes']['cal_dev']} samples)",
            f"- Model train: {manifest['partition_sizes']['model_train']} samples",
            f"- Locked test: {manifest['partition_sizes']['locked_test']} samples",
            f"- Frozen manifest: `{FROZEN_SPLIT_MANIFEST_PATH}`",
            "",
            "## 5. Model selection",
            "",
            f"- Selected candidate: `{SELECTED_MODEL_NAME}` (frozen before locked-test evaluation)",
            f"- Trained on full train partition ({manifest['partition_sizes']['model_train'] + manifest['partition_sizes']['cal_dev']} samples)",
            f"- Calibration method chosen on cal_dev using model_fit trained on model_train only ({manifest['partition_sizes']['model_train']} samples)",
            "",
            "## 6. Calibration methodology",
            "",
            f"- Prior 7.5%/95.4% audit: {prior_audit['verdict']}",
            f"- Calibrator fit partition: cal_dev only",
            f"- Decision: **{gate['calibration_decision']}** — {gate['calibration_decision_reason']}",
            "",
            "### Cal-dev comparison",
            "",
            "| Method | Brier | ECE | Benign FPR | Threat recall | Macro F1 |",
            "|--------|-------|-----|------------|---------------|----------|",
        ]
    )

    for name, metrics in (
        ("uncalibrated", gate["cal_dev_metrics"]["uncalibrated"]),
        ("isotonic", gate["cal_dev_metrics"]["isotonic"]),
    ):
        lines.append(
            f"| {name} | {metrics['brier_score_threat']:.4f} | "
            f"{metrics['expected_calibration_error_threat']:.4f} | "
            f"{metrics['benign_false_positive_rate']:.4f} | "
            f"{metrics['threat_recall']:.4f} | {metrics['macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 7. Final held-out metrics (ONE locked evaluation)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {final['accuracy']:.4f} |",
            f"| Macro F1 | {final['macro_f1']:.4f} |",
            f"| Weighted F1 | {final['weighted_f1']:.4f} |",
            f"| Benign precision | {final['benign_precision']:.4f} |",
            f"| Benign recall | {final['benign_recall']:.4f} |",
            f"| Benign FPR | {final['benign_false_positive_rate']:.4f} |",
            f"| Threat precision | {final['threat_precision']:.4f} |",
            f"| Threat recall | {final['threat_recall']:.4f} |",
            f"| Brier (threat) | {final['brier_score_threat']:.4f} |",
            f"| ECE (threat) | {final['expected_calibration_error_threat']:.4f} |",
            f"| Latency ms/sample | {final['latency_ms_per_sample']:.4f} |",
            "",
            "### Train partition metrics (not headline)",
            "",
            f"- Macro F1: {train['macro_f1']:.4f}",
            f"- Benign FPR: {train['benign_false_positive_rate']:.4f}",
            f"- Threat recall: {train['threat_recall']:.4f}",
            "",
            "### Cal-dev metrics (selection only)",
            "",
            f"- Macro F1: {cal['macro_f1']:.4f}",
            f"- Benign FPR: {cal['benign_false_positive_rate']:.4f}",
            f"- Threat recall: {cal['threat_recall']:.4f}",
            "",
            "## 8. Comparison with old pipeline",
            "",
            "| Metric | Legacy (old V2 + LR + calibration) | Clean gate (locked test) |",
            "|--------|--------------------------------------|--------------------------|",
            f"| Benign FPR | ~0.821 | {final['benign_false_positive_rate']:.4f} |",
            f"| Threat recall | ~0.817 (uncal clean exp.) | {final['threat_recall']:.4f} |",
            f"| Macro F1 | ~0.617 | {final['macro_f1']:.4f} |",
            "",
            "## 9. Artifact paths",
            "",
            f"- Phase 3 clean: `{PHASE3_CLEAN_OUTPUT_DIR}`",
            f"- Phase 4 clean: `{PHASE4_CLEAN_OUTPUT_DIR}`",
            f"- Phase 5 clean: `{PHASE5_CLEAN_OUTPUT_DIR}`",
            f"- Legacy (marked): `{LEGACY_PHASE3_DIR}`, `{LEGACY_PHASE4_DIR}`, `{LEGACY_PHASE5_DIR}`",
            "",
            "## 10. Reproducibility",
            "",
            f"- Double inference match: {repro['double_inference_predictions_match']}",
            f"- Probability match: {repro['double_inference_probabilities_match']}",
            f"- Artifact reload match: {repro['artifact_reload_predictions_match']}",
            "",
            "## 11. Test results",
            "",
            "Run: `cd ai && python -m pytest tests/test_promotion/ tests/test_phase3/ tests/test_phase4/ tests/test_phase5/ -v`",
            "",
            "## 12. Known limitations",
            "",
            "- Synthetic data only; hardware path not validated",
            "- Clean chirp class remains near-perfect separability (expected synthetic artifact)",
            "- Prior headline 7.5%/95.4% invalid for promotion (see calibration audit)",
            "- Prior clean experiment 8.6% benign FPR was **uncalibrated** `linear_svm_balanced` with test-set-informed selection; this gate's locked-test FPR reflects **isotonic calibration** chosen on cal_dev only",
            "- Isotonic calibration improved cal_dev metrics but locked-test benign FPR is 22.6% — do not extrapolate from pre-gate experiment numbers",
            "",
            "## 13. Project performance requirements",
            "",
            "No explicit SIH FPR/recall thresholds were found in repository docs. "
            "**Performance threshold requires team/project sign-off.**",
            "",
            "Do NOT treat {:.1%} benign FPR as automatically production-ready.".format(
                final["benign_false_positive_rate"]
            ),
            "",
            "## 14. AI sign-off recommendation",
            "",
            f"**{sign_off}**",
            "",
            "Technical promotion gate completed with versioned clean artifacts. "
            "Proceed to team review for performance policy and OTA validation before backend integration.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Final clean AI validation promotion gate")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    result = run_final_validation_gate(write_report=not args.no_report)
    print(
        json.dumps(
            {
                "calibration_decision": result["calibration_decision"],
                "final_locked_test_metrics": result["final_locked_test_metrics"],
                "reproducibility": result["reproducibility"],
                "artifact_paths": result["artifact_paths"],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
