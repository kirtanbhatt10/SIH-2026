"""Phase 3 evaluation, comparison, selection, and artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml.phase3.data import grouped_train_test_split, load_dataset_v2
from ml.phase3.evaluation import (
    build_comparison_table,
    evaluate_model,
    random_forest_feature_importance,
    save_artifacts,
    select_candidate_model,
)
from ml.phase3.models import create_baseline_models


def _fit_all_models(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    y_test = dataset.labels[split.test_idx]
    models = create_baseline_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, model_name=name)
    return models, results, split, dataset


def test_evaluation_metrics(mini_dataset_dir):
    _, results, _, _ = _fit_all_models(mini_dataset_dir)
    for name, result in results.items():
        assert 0.0 <= result["accuracy"] <= 1.0
        assert "macro_f1" in result
        assert "benign_false_positive_rate" in result
        assert "threat_recall" in result
        assert len(result["confusion_matrix"]) == 5


def test_confusion_matrix_shape(mini_dataset_dir):
    _, results, _, _ = _fit_all_models(mini_dataset_dir)
    for result in results.values():
        cm = result["confusion_matrix"]
        assert len(cm) == 5 and len(cm[0]) == 5


def test_model_comparison_table(mini_dataset_dir):
    _, results, _, _ = _fit_all_models(mini_dataset_dir)
    table = build_comparison_table(list(results.values()))
    assert set(table.columns) == {
        "model", "accuracy", "macro_f1", "weighted_f1",
        "benign_fpr", "threat_recall", "latency_ms",
    }
    assert len(table) == 3


def test_model_selection(mini_dataset_dir):
    _, results, _, _ = _fit_all_models(mini_dataset_dir)
    table = build_comparison_table(list(results.values()))
    selected, reason = select_candidate_model(table)
    assert selected in results
    assert reason


def test_feature_importance(mini_dataset_dir):
    models, _, _, dataset = _fit_all_models(mini_dataset_dir)
    fi = random_forest_feature_importance(models["random_forest"], dataset.feature_names)
    assert len(fi) == 32
    assert fi["feature"].tolist() == dataset.feature_names or set(fi["feature"]) == set(dataset.feature_names)


def test_save_load_consistency(mini_dataset_dir, tmp_path):
    models, results, split, dataset = _fit_all_models(mini_dataset_dir)
    from ml.phase3.data import build_split_report

    comparison = build_comparison_table(list(results.values()))
    selected, reason = select_candidate_model(comparison)
    split_report = build_split_report(split, dataset.metadata, dataset.labels)
    rf_fi = random_forest_feature_importance(models["random_forest"], dataset.feature_names)

    out = tmp_path / "phase3_out"
    paths = save_artifacts(
        output_dir=out,
        models=models,
        evaluation_results=results,
        comparison=comparison,
        selected_model_name=selected,
        selection_reason=reason,
        split_report=split_report,
        dataset_info=dataset.dataset_info,
        rf_feature_importance=rf_fi,
    )

    loaded = joblib.load(paths["selected_model"])
    X_test = dataset.features[split.test_idx]
    original_preds = models[selected].predict(X_test)
    loaded_preds = loaded.predict(X_test)
    assert (original_preds == loaded_preds).all()

    with open(paths["selected_model_metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["model_name"] == selected
    assert Path(paths["model_comparison_csv"]).exists()
    assert Path(paths["feature_importance"]).exists()
    assert Path(paths["split_report"]).exists()
