"""Phase 4 training / calibration entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.phase3.data import build_split_report, grouped_train_test_split, load_dataset_v2

from .artifacts import (
    assert_model_frozen,
    load_phase3_metadata,
    load_phase3_model,
    snapshot_model,
)
from .calibrator import RiskCalibrator
from .config import PHASE4_OUTPUT_DIR, SPLIT_SEED, TEST_FRACTION
from .evaluation import evaluate, predictions_to_dataframe, save_phase4_artifacts


def run_phase4(
    *,
    output_dir: Path | None = None,
    seed: int = SPLIT_SEED,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else PHASE4_OUTPUT_DIR

    model = load_phase3_model(model_path)
    metadata = load_phase3_metadata(metadata_path)
    metadata = dict(metadata)
    metadata["_model_path"] = str(model_path or "")

    model_before = snapshot_model(model)

    dataset = load_dataset_v2()
    split = grouped_train_test_split(
        dataset.metadata,
        dataset.labels,
        test_fraction=TEST_FRACTION,
        seed=seed,
    )
    split.assert_no_group_leakage(dataset.metadata)
    split_report = build_split_report(split, dataset.metadata, dataset.labels)

    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]

    calibrator = RiskCalibrator(model, metadata)
    calibrator.calibrate(X_train, y_train)

    assert_model_frozen(model_before, snapshot_model(model))

    train_ids = dataset.metadata.iloc[split.train_idx]["sample_id"].tolist()
    test_ids = dataset.metadata.iloc[split.test_idx]["sample_id"].tolist()

    train_preds = calibrator.predict_risk(X_train)
    test_preds = calibrator.predict_risk(X_test)

    train_metrics = evaluate(calibrator, X_train, y_train, partition="train")
    test_metrics = evaluate(calibrator, X_test, y_test, partition="test")

    artifact_paths = save_phase4_artifacts(
        output_dir=out,
        calibrator=calibrator,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_predictions=predictions_to_dataframe(train_preds, y_train, train_ids),
        test_predictions=predictions_to_dataframe(test_preds, y_test, test_ids),
        metadata=metadata,
        split_report=split_report,
    )

    return {
        "split_report": split_report,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "risk_thresholds": calibrator.get_thresholds(),
        "artifact_paths": artifact_paths,
        "model_frozen": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 risk calibration")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, default=PHASE4_OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_phase4(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps({
        "risk_thresholds": summary["risk_thresholds"],
        "test_metrics_summary": {
            "accuracy": summary["test_metrics"]["accuracy"],
            "macro_f1": summary["test_metrics"]["macro_f1"],
            "benign_fpr": summary["test_metrics"]["benign_false_positive_rate"],
            "threat_recall": summary["test_metrics"]["threat_recall"],
            "brier_score_threat": summary["test_metrics"]["brier_score_threat"],
            "ece_threat": summary["test_metrics"]["expected_calibration_error_threat"],
            "risk_distribution": summary["test_metrics"]["risk_distribution"],
        },
        "artifact_paths": summary["artifact_paths"],
        "model_frozen": summary["model_frozen"],
    }, indent=2))


if __name__ == "__main__":
    main()
