"""Phase 4 evaluation and integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.phase4.evaluation import evaluate
from ml.phase4.train import run_phase4


@pytest.mark.integration
def test_full_phase4_integration(tmp_path):
    output_dir = tmp_path / "phase4_out"
    summary = run_phase4(output_dir=output_dir, seed=42)

    assert summary["model_frozen"] is True
    assert summary["split_report"]["leakage_check_passed"] is True
    assert summary["split_report"]["train_size"] == 2039
    assert summary["split_report"]["test_size"] == 461

    required = [
        "calibration_report.json",
        "calibration_metrics.json",
        "risk_distribution.csv",
        "calibrated_predictions.csv",
        "calibration_metadata.json",
    ]
    for name in required:
        assert (output_dir / name).exists(), f"missing {name}"

    metrics = json.loads((output_dir / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert "train" in metrics and "test" in metrics
    assert metrics["test"]["probability_checks"]["nan_count"] == 0
    assert metrics["test"]["probability_checks"]["proba_sum_close_to_one"] is True

    preds = pd.read_csv(output_dir / "calibrated_predictions.csv")
    assert len(preds) == 2500
    assert set(preds["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})


def test_held_out_test_evaluation_metrics(tmp_path):
    summary = run_phase4(output_dir=tmp_path / "phase4_eval", seed=42)
    test_metrics = summary["test_metrics"]
    assert "accuracy" in test_metrics
    assert "brier_score_threat" in test_metrics
    assert "expected_calibration_error_threat" in test_metrics
    assert "confusion_matrix" in test_metrics
    assert test_metrics["probability_checks"]["inf_count"] == 0
