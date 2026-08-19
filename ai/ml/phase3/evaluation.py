"""Evaluation, model comparison, selection, and artifact helpers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import (
    BENIGN_CLASS,
    CLASS_NAMES,
    EXPECTED_CLASSES,
    FEATURE_NAMES,
    PHASE3_OUTPUT_DIR,
    SYNTHETIC_DATA_LIMITATION,
    THREAT_CLASSES,
)
from .models import get_model_classes


def measure_latency_ms(model: Any, X: np.ndarray, *, repeats: int = 3) -> dict[str, float]:
    if X.shape[0] == 0:
        return {"latency_ms_mean": 0.0, "latency_ms_std": 0.0, "latency_ms_per_sample": 0.0}

    # Warm-up
    model.predict(X[:1])

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms / X.shape[0])

    return {
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_std": float(np.std(timings)),
        "latency_ms_per_sample": float(np.mean(timings)),
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str,
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = get_model_classes(model)

    labels = list(EXPECTED_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    per_class: dict[str, dict[str, float | int]] = {}
    for i, label in enumerate(labels):
        name = CLASS_NAMES[label]
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    benign_mask = y_test == BENIGN_CLASS
    threat_pred = np.isin(y_pred, list(THREAT_CLASSES))
    n_benign = int(benign_mask.sum())
    n_benign_as_threat = int((benign_mask & threat_pred).sum())
    benign_fpr = n_benign_as_threat / n_benign if n_benign else 0.0

    threat_recalls = [per_class[CLASS_NAMES[c]]["recall"] for c in THREAT_CLASSES]
    threat_recall = float(np.mean(threat_recalls))

    latency = measure_latency_ms(model, X_test)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "benign_recall": float(per_class["benign"]["recall"]),
        "threat_recall": threat_recall,
        "threat_recall_definition": (
            "Macro-average recall over threat classes (fsk, ook, chirp, tone)."
        ),
        "benign_false_positive_rate": float(benign_fpr),
        "benign_false_positive_rate_definition": (
            "Fraction of benign test samples predicted as any threat class."
        ),
        "n_benign": n_benign,
        "n_benign_as_threat": n_benign_as_threat,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in labels],
        "predicted_classes_order": classes,
        "probability_shape": list(probs.shape),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, target_names=[CLASS_NAMES[c] for c in labels], zero_division=0
        ),
        "latency_ms_mean": latency["latency_ms_mean"],
        "latency_ms_std": latency["latency_ms_std"],
    }


def build_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({
            "model": result["model_name"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "benign_fpr": result["benign_false_positive_rate"],
            "threat_recall": result["threat_recall"],
            "latency_ms": result["latency_ms_mean"],
        })
    return pd.DataFrame(rows)


def select_candidate_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Select model using cybersecurity-relevant priority ordering."""
    ranked = comparison.copy()
    ranked = ranked.sort_values(
        by=["macro_f1", "threat_recall", "benign_fpr", "accuracy", "latency_ms"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    )
    selected = str(ranked.iloc[0]["model"])
    top = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else None

    reason_parts = [
        f"Selected {selected} based on priority: macro F1, threat recall, "
        f"benign FPR (lower better), accuracy, latency.",
        (
            f"{selected}: macro_f1={top['macro_f1']:.4f}, "
            f"threat_recall={top['threat_recall']:.4f}, "
            f"benign_fpr={top['benign_fpr']:.4f}, "
            f"accuracy={top['accuracy']:.4f}, "
            f"latency={top['latency_ms']:.4f} ms/sample."
        ),
    ]
    if second is not None:
        macro_close = abs(top["macro_f1"] - second["macro_f1"]) < 0.01
        threat_close = abs(top["threat_recall"] - second["threat_recall"]) < 0.01
        if macro_close or threat_close:
            reason_parts.append(
                f"Tradeoff note: runner-up {second['model']} had "
                f"macro_f1={second['macro_f1']:.4f}, threat_recall={second['threat_recall']:.4f}, "
                f"benign_fpr={second['benign_fpr']:.4f}, latency={second['latency_ms']:.4f} ms/sample."
            )
    return selected, " ".join(reason_parts)


def random_forest_feature_importance(
    model: Any,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    names = feature_names or FEATURE_NAMES
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
    else:
        raise TypeError("Model is not a fitted RandomForestClassifier")

    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_confusion_matrix_plot(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_artifacts(
    *,
    output_dir: Path,
    models: dict[str, Any],
    evaluation_results: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
    selected_model_name: str,
    selection_reason: str,
    split_report: dict[str, Any],
    dataset_info: dict[str, Any],
    rf_feature_importance: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    selected_model = models[selected_model_name]
    model_path = output_dir / "selected_model.joblib"
    joblib.dump(selected_model, model_path)
    paths["selected_model"] = str(model_path)

    selected_eval = evaluation_results[selected_model_name]
    metadata = {
        "model_name": selected_model_name,
        "model_parameters": _extract_model_params(selected_model),
        "dataset_version": dataset_info.get("dataset_version"),
        "generator_version": dataset_info.get("generator_version"),
        "dataset_seed": dataset_info.get("seed"),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_mapping": CLASS_NAMES,
        "split_method": split_report.get("split_method"),
        "split_seed": split_report.get("split_seed"),
        "train_size": split_report.get("train_size"),
        "test_size": split_report.get("test_size"),
        "generation_group_policy": split_report.get("generation_group_policy"),
        "evaluation_metrics": {
            k: selected_eval[k]
            for k in (
                "accuracy", "macro_f1", "weighted_f1", "benign_false_positive_rate",
                "threat_recall", "benign_recall", "latency_ms_mean",
            )
        },
        "selection_reason": selection_reason,
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "predicted_classes_order": selected_eval.get("predicted_classes_order"),
    }
    metadata_path = output_dir / "selected_model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    paths["selected_model_metadata"] = str(metadata_path)

    comparison_csv = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    paths["model_comparison_csv"] = str(comparison_csv)

    comparison_json = output_dir / "model_comparison.json"
    with comparison_json.open("w", encoding="utf-8") as fh:
        json.dump(comparison.to_dict(orient="records"), fh, indent=2)
    paths["model_comparison_json"] = str(comparison_json)

    evaluation_report = {
        "models": evaluation_results,
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "split_report": split_report,
        "dataset_version": dataset_info.get("dataset_version"),
        "synthetic_data_limitation": SYNTHETIC_DATA_LIMITATION,
    }
    eval_path = output_dir / "evaluation_report.json"
    with eval_path.open("w", encoding="utf-8") as fh:
        json.dump(evaluation_report, fh, indent=2)
    paths["evaluation_report"] = str(eval_path)

    for model_name, result in evaluation_results.items():
        cm_path = output_dir / f"confusion_matrix_{model_name}.png"
        save_confusion_matrix_plot(
            result["confusion_matrix"],
            result["confusion_matrix_labels"],
            cm_path,
            title=f"Confusion Matrix — {model_name}",
        )
        paths[f"confusion_matrix_{model_name}"] = str(cm_path)

    fi_path = output_dir / "feature_importance.csv"
    rf_feature_importance.to_csv(fi_path, index=False)
    paths["feature_importance"] = str(fi_path)

    split_path = output_dir / "split_report.json"
    with split_path.open("w", encoding="utf-8") as fh:
        json.dump(split_report, fh, indent=2)
    paths["split_report"] = str(split_path)

    return paths


def _extract_model_params(model: Any) -> dict[str, Any]:
    if not hasattr(model, "get_params"):
        return {}
    raw = model.get_params(deep=False)
    return _json_safe_params(raw)


def _json_safe_params(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_params(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_params(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
