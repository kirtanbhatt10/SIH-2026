"""Phase 5 CLI — validate artifacts and run frozen inference evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.phase3.data import load_dataset_v2

from .config import PHASE5_OUTPUT_DIR, SPLIT_SEED
from .evaluation import evaluate_service, save_phase5_report
from .service import InferenceService


def run_phase5(
    *,
    output_dir: Path | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else PHASE5_OUTPUT_DIR

    service = InferenceService.from_artifacts()
    dataset = load_dataset_v2()

    metrics = evaluate_service(
        service,
        dataset.features,
        dataset.labels,
        sample_ids=dataset.metadata["sample_id"].tolist(),
    )

    artifact_paths = save_phase5_report(
        output_dir=out,
        service=service,
        dataset_metrics=metrics,
        phase3_metadata=service.phase3_metadata,
        phase4_metadata=service.phase4_metadata,
    )

    return {
        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
        "artifact_paths": artifact_paths,
        "risk_thresholds": service.get_risk_thresholds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 frozen inference validation and evaluation"
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, default=PHASE5_OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_phase5(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps({
        "risk_thresholds": summary["risk_thresholds"],
        "metrics": {
            "n_samples": summary["metrics"]["n_samples"],
            "accuracy": summary["metrics"]["accuracy"],
            "macro_f1": summary["metrics"]["macro_f1"],
            "benign_fpr": summary["metrics"]["benign_false_positive_rate"],
            "threat_recall": summary["metrics"]["threat_recall"],
            "risk_distribution": summary["metrics"]["risk_distribution"],
            "latency_ms_per_sample": summary["metrics"]["latency_ms_per_sample"],
        },
        "artifact_paths": summary["artifact_paths"],
    }, indent=2))


if __name__ == "__main__":
    main()
