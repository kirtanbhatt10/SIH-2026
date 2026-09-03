"""Full Phase 3 integration test on the 2500-sample Dataset V2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml.phase3.config import PHASE3_OUTPUT_DIR
from ml.phase3.train import run_phase3


@pytest.mark.integration
def test_full_phase3_integration(tmp_path):
    output_dir = tmp_path / "phase3_integration"
    summary = run_phase3(output_dir=output_dir, seed=42)

    assert summary["dataset_shape"] == [2500, 32]
    assert summary["split_report"]["leakage_check_passed"] is True
    assert summary["split_report"]["train_size"] == 2039
    assert summary["split_report"]["test_size"] == 461
    assert len(summary["comparison"]) == 3
    assert summary["selected_model"] in {"random_forest", "rbf_svm", "logistic_regression"}

    required = [
        "selected_model.joblib",
        "selected_model_metadata.json",
        "model_comparison.csv",
        "model_comparison.json",
        "evaluation_report.json",
        "feature_importance.csv",
        "split_report.json",
    ]
    for name in required:
        assert (output_dir / name).exists(), f"missing artifact: {name}"

    for model_name in ("random_forest", "rbf_svm", "logistic_regression"):
        assert (output_dir / f"confusion_matrix_{model_name}.png").exists()

    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    assert len(comparison) == 3

    loaded = joblib.load(output_dir / "selected_model.joblib")
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "predict_proba")

    with (output_dir / "split_report.json").open(encoding="utf-8") as fh:
        split_report = json.load(fh)
    assert split_report["group_overlap_count"] == 0


def test_reproducibility(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    summary_a = run_phase3(output_dir=out_a, seed=42)
    summary_b = run_phase3(output_dir=out_b, seed=42)

    assert summary_a["split_report"] == summary_b["split_report"]
    assert summary_a["selected_model"] == summary_b["selected_model"]

    def _without_latency(rows):
        return [{k: v for k, v in row.items() if k != "latency_ms"} for row in rows]

    assert _without_latency(summary_a["comparison"]) == _without_latency(summary_b["comparison"])
