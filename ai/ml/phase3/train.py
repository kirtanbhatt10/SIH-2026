"""Phase 3 training and evaluation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PHASE3_OUTPUT_DIR, SPLIT_SEED, load_dataset_info
from .data import (
    build_split_report,
    class_balance_report,
    grouped_train_test_split,
    load_dataset_v2,
)
from .evaluation import (
    build_comparison_table,
    evaluate_model,
    random_forest_feature_importance,
    save_artifacts,
    select_candidate_model,
)
from .models import assert_probability_shape, create_baseline_models, get_model_classes


def run_phase3(
    *,
    output_dir: Path | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else PHASE3_OUTPUT_DIR
    dataset = load_dataset_v2()
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=seed)
    split_report = build_split_report(split, dataset.metadata, dataset.labels)

    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]

    models = create_baseline_models()
    evaluation_results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        assert_probability_shape(model, X_test[:5])
        classes = get_model_classes(model)
        evaluation_results[model_name] = evaluate_model(
            model, X_test, y_test, model_name=model_name
        )
        evaluation_results[model_name]["fitted_classes_order"] = classes

    comparison = build_comparison_table(list(evaluation_results.values()))
    selected_name, selection_reason = select_candidate_model(comparison)

    rf_model = models["random_forest"]
    rf_importance = random_forest_feature_importance(rf_model, dataset.feature_names)

    dataset_info = load_dataset_info()
    artifact_paths = save_artifacts(
        output_dir=out,
        models=models,
        evaluation_results=evaluation_results,
        comparison=comparison,
        selected_model_name=selected_name,
        selection_reason=selection_reason,
        split_report=split_report,
        dataset_info=dataset_info,
        rf_feature_importance=rf_importance,
    )

    return {
        "dataset_shape": list(dataset.features.shape),
        "class_balance": class_balance_report(dataset.labels),
        "split_report": split_report,
        "comparison": comparison.to_dict(orient="records"),
        "selected_model": selected_name,
        "selection_reason": selection_reason,
        "evaluation_results": evaluation_results,
        "artifact_paths": artifact_paths,
        "feature_importance_top5": rf_importance.head(5).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 ML baseline comparison")
    parser.add_argument("--output-dir", type=Path, default=PHASE3_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    summary = run_phase3(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps({
        "selected_model": summary["selected_model"],
        "selection_reason": summary["selection_reason"],
        "comparison": summary["comparison"],
        "split_report": summary["split_report"],
        "artifact_paths": summary["artifact_paths"],
    }, indent=2))


if __name__ == "__main__":
    main()
