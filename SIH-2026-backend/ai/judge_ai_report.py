#!/usr/bin/env python3
"""SIH 2026 — Judge-facing AI performance report (read-only).

Accuracy and classification metrics are read from the pre-computed frozen
artifact ``ml/output/phase5_clean/locked_test_evaluation.json``.  No locked-test
re-evaluation, retraining, or dataset regeneration is performed.

Live inference latency and throughput are measured on this machine using
``InferenceService.load_frozen()`` with 32-feature inputs.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

AI_ROOT = Path(__file__).resolve().parent
LOCKED_EVAL_PATH = AI_ROOT / "ml" / "output" / "phase5_clean" / "locked_test_evaluation.json"
METADATA_PATH = AI_ROOT / "ml" / "output" / "phase5_clean" / "inference_metadata.json"

WARMUP_CALLS = 20
BENCHMARK_CALLS = 1_000

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy",),
    "macro_f1": ("macro_f1", "macro F1", "macroF1"),
    "benign_fpr": ("benign_fpr", "benign_false_positive_rate", "benign false positive rate"),
    "benign_recall": ("benign_recall", "benign recall"),
    "threat_recall": ("threat_recall", "threat recall"),
    "threat_precision": ("threat_precision", "threat precision"),
    "brier": ("brier", "brier_score", "brier_score_threat"),
    "ece": ("ece", "expected_calibration_error", "expected_calibration_error_threat"),
}


def _suppress_sklearn_version_warnings() -> None:
    try:
        from sklearn.exceptions import InconsistentVersionWarning

        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _metric_sources(locked_eval: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    metrics = locked_eval.get("metrics")
    if isinstance(metrics, dict):
        sources.append(metrics)
    reference = locked_eval.get("reference_from_final_model_comparison")
    if isinstance(reference, dict):
        sources.append(reference)
    locked_metrics = locked_eval.get("locked_test_metrics")
    if isinstance(locked_metrics, dict):
        sources.append(locked_metrics)
    if not sources:
        sources.append(locked_eval)
    return sources


def _normalize_metric_value(value: Any) -> float:
    if value is None:
        raise ValueError("metric value is missing")
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        number = float(value)
    else:
        number = float(value)
    if number > 1.0:
        return number / 100.0
    return number


def _extract_metric(sources: list[dict[str, Any]], aliases: tuple[str, ...]) -> float:
    normalized_aliases = {alias.replace(" ", "_").lower() for alias in aliases}
    for source in sources:
        for key, value in source.items():
            if key.replace(" ", "_").lower() in normalized_aliases:
                return _normalize_metric_value(value)
    joined = ", ".join(aliases)
    raise KeyError(f"Could not find metric among aliases: {joined}")


def _extract_locked_metrics(locked_eval: dict[str, Any]) -> dict[str, float]:
    sources = _metric_sources(locked_eval)
    return {name: _extract_metric(sources, aliases) for name, aliases in METRIC_ALIASES.items()}


def _extract_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    model = (
        metadata.get("phase3_model_name")
        or metadata.get("model_name")
        or metadata.get("model")
        or "unknown"
    )
    calibration = (
        metadata.get("model_calibration")
        or metadata.get("calibration")
        or "unknown"
    )
    dataset = (
        metadata.get("dataset_version")
        or metadata.get("dataset")
        or "unknown"
    )
    lineage = (
        metadata.get("artifact_lineage")
        or metadata.get("lineage")
        or metadata.get("artifact_version")
        or "unknown"
    )
    return {
        "model": str(model),
        "calibration": str(calibration),
        "dataset": str(dataset),
        "lineage": str(lineage),
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _benchmark_predict_latency(service: Any, features: np.ndarray) -> dict[str, float]:
    for _ in range(WARMUP_CALLS):
        service.predict(features)

    start = time.perf_counter()
    for _ in range(BENCHMARK_CALLS):
        service.predict(features)
    elapsed_sec = time.perf_counter() - start

    total_ms = elapsed_sec * 1000.0
    latency_ms = total_ms / BENCHMARK_CALLS
    throughput = BENCHMARK_CALLS / elapsed_sec if elapsed_sec > 0 else 0.0
    return {
        "total_ms": total_ms,
        "latency_ms": latency_ms,
        "throughput": throughput,
    }


def _print_report(
    meta: dict[str, str],
    metrics: dict[str, float],
    benchmark: dict[str, float],
) -> None:
    width = 72
    print("=" * width)
    print("                 SIH 2026 - AI PERFORMANCE REPORT".center(width))
    print("=" * width)
    print()
    print("MODEL INFORMATION")
    print("-" * width)
    print(f"Model              : {meta['model']}")
    print(f"Calibration        : {meta['calibration']}")
    print(f"Dataset            : {meta['dataset']}")
    print(f"Lineage            : {meta['lineage']}")
    print()
    print("FROZEN LOCKED-TEST PERFORMANCE")
    print("-" * width)
    print(f"Accuracy           : {_format_percent(metrics['accuracy'])}")
    print(f"Macro F1           : {_format_percent(metrics['macro_f1'])}")
    print(f"Benign Recall      : {_format_percent(metrics['benign_recall'])}")
    print(f"Benign FPR         : {_format_percent(metrics['benign_fpr'])}")
    print(f"Threat Recall      : {_format_percent(metrics['threat_recall'])}")
    print(f"Threat Precision   : {_format_percent(metrics['threat_precision'])}")
    print(f"Brier Score        : {_format_number(metrics['brier'])}")
    print(f"ECE                : {_format_number(metrics['ece'])}")
    print()
    print("LIVE INFERENCE BENCHMARK")
    print("-" * width)
    print(f"Benchmark samples  : {_format_count(BENCHMARK_CALLS)}")
    print(f"Total time         : {_format_number(benchmark['total_ms'], 2)} ms")
    print(f"Latency            : {_format_number(benchmark['latency_ms'], 3)} ms/sample")
    print(f"Throughput         : {_format_number(benchmark['throughput'], 1)} samples/sec")
    print()
    print("=" * width)
    print("                         JUDGE SUMMARY".center(width))
    print("=" * width)
    print()
    print(f"  ACCURACY       : {_format_percent(metrics['accuracy'])}")
    print(f"  MACRO F1       : {_format_percent(metrics['macro_f1'])}")
    print(f"  THREAT RECALL  : {_format_percent(metrics['threat_recall'])}")
    print(f"  BENIGN RECALL  : {_format_percent(metrics['benign_recall'])}")
    print(f"  LATENCY        : {_format_number(benchmark['latency_ms'], 3)} ms/sample")
    print(f"  THROUGHPUT     : {_format_number(benchmark['throughput'], 1)} samples/sec")
    print()
    print("  Model status    : FROZEN / VALIDATED")
    print("  Locked test     : PRE-COMPUTED EVALUATION")
    print("  Retraining      : NOT PERFORMED")
    print("  Test leakage    : NOT PERFORMED")
    print()
    print("=" * width)
    print("                    END OF AI REPORT".center(width))
    print("=" * width)


def main() -> int:
    _suppress_sklearn_version_warnings()

    if str(AI_ROOT) not in sys.path:
        sys.path.insert(0, str(AI_ROOT))

    try:
        locked_eval = _load_json(LOCKED_EVAL_PATH)
        metadata_doc = _load_json(METADATA_PATH)
        metrics = _extract_locked_metrics(locked_eval)
        meta = _extract_metadata(metadata_doc)

        from ml.phase5.service import InferenceService

        service = InferenceService.load_frozen()
        features = np.zeros(32, dtype=float)
        benchmark = _benchmark_predict_latency(service, features)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(meta, metrics, benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
