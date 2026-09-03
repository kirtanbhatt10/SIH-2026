"""Baseline model factories for Phase 3."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_NAMES,
    RANDOM_FOREST_PARAMS,
    SVM_PARAMS,
)


def create_random_forest(**overrides: Any) -> RandomForestClassifier:
    params = dict(RANDOM_FOREST_PARAMS)
    params.update(overrides)
    return RandomForestClassifier(**params)


def create_rbf_svm(**overrides: Any) -> Pipeline:
    params = dict(SVM_PARAMS)
    params.update(overrides)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**params)),
    ])


def create_logistic_regression(**overrides: Any) -> Pipeline:
    params = dict(LOGISTIC_REGRESSION_PARAMS)
    params.update(overrides)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def create_baseline_models() -> dict[str, Any]:
    return {
        "random_forest": create_random_forest(),
        "rbf_svm": create_rbf_svm(),
        "logistic_regression": create_logistic_regression(),
    }


def get_model_classes(model: Any) -> list[int]:
    if hasattr(model, "classes_"):
        return [int(c) for c in model.classes_]
    if hasattr(model, "named_steps") and hasattr(model.named_steps["clf"], "classes_"):
        return [int(c) for c in model.named_steps["clf"].classes_]
    raise AttributeError("Model does not expose classes_ after fit")


def assert_probability_shape(model: Any, X: Any, n_classes: int = 5) -> None:
    probs = model.predict_proba(X)
    if probs.shape[1] != n_classes:
        raise ValueError(
            f"predict_proba expected {n_classes} columns, got {probs.shape[1]}"
        )
