"""Phase 5 integration tests on real Dataset V2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.phase3.data import load_dataset_v2
from ml.phase5.evaluation import evaluate_service
from ml.phase5.service import InferenceService
from ml.phase5.train import run_phase5


@pytest.fixture
def service():
    return InferenceService.from_artifacts()


@pytest.mark.integration
def test_full_dataset_v2_inference(service):
    dataset = load_dataset_v2()
    metrics = evaluate_service(
        service,
        dataset.features,
        dataset.labels,
        sample_ids=dataset.metadata["sample_id"].tolist(),
    )
    assert metrics["n_samples"] == 2500
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert sum(metrics["risk_distribution"].values()) == 2500
    assert metrics["latency_ms_per_sample"] >= 0.0


@pytest.mark.integration
def test_run_phase5_cli(tmp_path):
    output_dir = tmp_path / "phase5_out"
    summary = run_phase5(output_dir=output_dir)

    required = [
        "inference_report.json",
        "inference_metrics.json",
        "inference_predictions.csv",
        "inference_metadata.json",
        "inference_service.joblib",
    ]
    for name in required:
        assert (output_dir / name).exists(), f"missing {name}"

    assert summary["metrics"]["n_samples"] == 2500
    report = json.loads((output_dir / "inference_report.json").read_text(encoding="utf-8"))
    assert report["phase"] == 5

    meta = json.loads((output_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    assert meta["no_training"] is True

    loaded = InferenceService.load(output_dir / "inference_service.joblib")
    dataset = load_dataset_v2()
    sample = dataset.features[0]
    assert service_predict_eq(loaded, service_from_artifacts(), sample)


def service_from_artifacts():
    return InferenceService.from_artifacts()


def service_predict_eq(a: InferenceService, b: InferenceService, sample):
    return a.predict(sample) == b.predict(sample)
