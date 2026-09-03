"""Compare calibration behavior on held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ml.phase3.data import grouped_train_test_split, load_dataset_v2
from ml.phase3.evaluation import evaluate_model
from ml.phase5.artifacts import load_phase3_model


def _threat_probability(proba: np.ndarray, threat_classes=(1, 2, 3, 4)) -> np.ndarray:
    return proba[:, list(threat_classes)].sum(axis=1)


def run_calibration_audit(*, output_path: Path) -> dict:
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    model = load_phase3_model()

    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]

    uncalibrated = evaluate_model(model, X_test, y_test, model_name="phase3_saved_model")

    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    y_train_threat = (y_train != 0).astype(int)
    y_test_threat = (y_test != 0).astype(int)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(_threat_probability(train_proba), y_train_threat)
    calibrated_scores = iso.predict(_threat_probability(test_proba))
    calibrated_pred = np.where(calibrated_scores >= 0.5, 1, 0)

    benign_mask = y_test == 0
    benign_fpr_iso = float((benign_mask & (calibrated_pred == 1)).sum() / benign_mask.sum())

    report = {
        "uncalibrated_phase3_test": {
            "accuracy": uncalibrated["accuracy"],
            "macro_f1": uncalibrated["macro_f1"],
            "benign_fpr": uncalibrated["benign_false_positive_rate"],
            "benign_recall": uncalibrated["benign_recall"],
            "threat_recall": uncalibrated["threat_recall"],
        },
        "isotonic_threat_binary_on_test": {
            "benign_fpr": benign_fpr_iso,
            "threat_recall": float(calibrated_pred[y_test_threat == 1].mean()),
        },
        "assessment": (
            "Isotonic calibration fit on train threat probability can worsen held-out "
            "benign FPR if threat scores for benign samples are miscalibrated upward."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/investigation/calibration_audit_report.json"),
    )
    args = parser.parse_args()
    report = run_calibration_audit(output_path=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
