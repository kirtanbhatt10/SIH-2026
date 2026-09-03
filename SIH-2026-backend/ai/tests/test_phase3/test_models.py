"""Phase 3 model training and interface tests."""

from __future__ import annotations

import numpy as np

from ml.phase3.data import grouped_train_test_split, load_dataset_v2
from ml.phase3.models import (
    assert_probability_shape,
    create_baseline_models,
    create_logistic_regression,
    create_random_forest,
    create_rbf_svm,
    get_model_classes,
)


def _train_test_from_mini(mini_dataset_dir):
    dataset = load_dataset_v2(mini_dataset_dir)
    split = grouped_train_test_split(dataset.metadata, dataset.labels, seed=42)
    X_train = dataset.features[split.train_idx]
    y_train = dataset.labels[split.train_idx]
    X_test = dataset.features[split.test_idx]
    return X_train, y_train, X_test


def test_random_forest_training(mini_dataset_dir):
    X_train, y_train, X_test = _train_test_from_mini(mini_dataset_dir)
    model = create_random_forest()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert_probability_shape(model, X_test)


def test_svm_training(mini_dataset_dir):
    X_train, y_train, X_test = _train_test_from_mini(mini_dataset_dir)
    model = create_rbf_svm()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert probs.shape == (X_test.shape[0], 5)


def test_logistic_regression_training(mini_dataset_dir):
    X_train, y_train, X_test = _train_test_from_mini(mini_dataset_dir)
    model = create_logistic_regression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert probs.shape == (X_test.shape[0], 5)


def test_predict_proba_shape_all_models(mini_dataset_dir):
    X_train, y_train, X_test = _train_test_from_mini(mini_dataset_dir)
    for model in create_baseline_models().values():
        model.fit(X_train, y_train)
        classes = get_model_classes(model)
        assert len(classes) == 5
        assert_probability_shape(model, X_test)
