"""Hyperparameter search on inner development data."""

from __future__ import annotations

import copy
from typing import Any

from .evaluation import evaluate_candidate, rank_candidates
from .models import create_model_catalog, hyperparameter_grids


def tune_model_family(
    model_name: str,
    X_train: Any,
    y_train: Any,
    X_inner: Any,
    y_inner: Any,
) -> dict[str, Any]:
    catalog = create_model_catalog()
    if model_name not in catalog:
        raise KeyError(model_name)
    grid = hyperparameter_grids().get(model_name, [{}])

    best_row = None
    best_model = None
    trials: list[dict[str, Any]] = []

    for params in grid:
        model = copy.deepcopy(catalog[model_name])
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)
        metrics = evaluate_candidate(
            model, X_inner, y_inner, partition="inner_dev", model_name=model_name
        )
        row = {
            "model_name": model_name,
            "params": params,
            **{k: metrics[k] for k in (
                "accuracy", "macro_f1", "weighted_f1", "benign_false_positive_rate",
                "benign_recall", "threat_recall", "threat_precision", "benign_precision",
                "latency_ms_per_sample",
            )},
        }
        trials.append(row)
        if best_row is None or rank_candidates([row, best_row])[0] == row:
            best_row = row
            best_model = model

    return {
        "model_name": model_name,
        "best_params": best_row["params"] if best_row else {},
        "best_inner_dev": best_row,
        "trials": trials,
        "best_model": best_model,
    }
