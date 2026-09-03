"""Model factories and hyperparameter grids for accuracy_82."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.investigation.model_experiments import _make_models


def baseline_linear_svm_balanced(**overrides: Any) -> Pipeline:
    model = _make_models()["linear_svm_balanced"]
    model.set_params(**overrides)
    return model


def create_model_catalog() -> dict[str, Any]:
    return {
        "linear_svm_balanced": baseline_linear_svm_balanced(),
        "rbf_svm_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
        ]),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=42, max_depth=6, learning_rate=0.08
        ),
    }


def hyperparameter_grids() -> dict[str, list[dict[str, Any]]]:
    return {
        "linear_svm_balanced": [{"clf__C": c} for c in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)],
        "rbf_svm_balanced": [
            {"clf__C": c, "clf__gamma": g}
            for c in (0.5, 1.0, 2.0, 5.0)
            for g in ("scale", "auto", 0.01, 0.05)
        ],
        "extra_trees_balanced": [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n in (300, 500)
            for d in (None, 24, 48)
            for leaf in (1, 2)
        ],
        "hist_gradient_boosting": [
            {"max_depth": d, "learning_rate": lr, "max_leaf_nodes": nodes}
            for d in (4, 6, 8)
            for lr in (0.05, 0.08, 0.12)
            for nodes in (31, 63)
        ],
    }
