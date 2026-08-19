"""Load and validate Phase 3 artifacts for Phase 4."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy as np

from .config import (
    BENIGN_CLASS,
    CLASS_NAMES,
    EXPECTED_CLASSES,
    FEATURE_NAMES,
    N_FEATURES,
    PHASE3_METADATA_PATH,
    PHASE3_MODEL_PATH,
    SPLIT_METHOD,
    SPLIT_SEED,
    TEST_FRACTION,
)


class ArtifactValidationError(ValueError):
    """Raised when a Phase 3 artifact fails validation."""


def load_phase3_model(model_path: Path | None = None) -> Any:
    path = Path(model_path) if model_path is not None else PHASE3_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 model not found: {path}")
    return joblib.load(path)


def load_phase3_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    path = Path(metadata_path) if metadata_path is not None else PHASE3_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Phase 3 metadata not found: {path}")
    with path.open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    validate_phase3_metadata(metadata)
    return metadata


def validate_phase3_metadata(metadata: dict[str, Any]) -> None:
    feature_names = metadata.get("feature_names")
    if feature_names is None:
        raise ArtifactValidationError("metadata missing feature_names")
    if len(feature_names) != N_FEATURES:
        raise ArtifactValidationError(
            f"Expected {N_FEATURES} feature names, got {len(feature_names)}"
        )
    if list(feature_names) != list(FEATURE_NAMES):
        raise ArtifactValidationError(
            "feature_names do not match Phase-1 authoritative order"
        )

    raw_mapping = metadata.get("class_mapping", {})
    class_mapping = {int(k): v for k, v in raw_mapping.items()}
    if class_mapping != CLASS_NAMES:
        raise ArtifactValidationError(
            f"class_mapping mismatch: expected {CLASS_NAMES}, got {class_mapping}"
        )

    if metadata.get("split_method") != SPLIT_METHOD:
        raise ArtifactValidationError(
            f"split_method must be {SPLIT_METHOD!r}"
        )
    if metadata.get("split_seed") != SPLIT_SEED:
        raise ArtifactValidationError(f"split_seed must be {SPLIT_SEED}")
    if metadata.get("train_size") != 2039 or metadata.get("test_size") != 461:
        raise ArtifactValidationError(
            "train_size/test_size do not match expected Phase 3 split"
        )


def snapshot_model(model: Any) -> dict[str, Any]:
    """Capture fitted model weights for freeze verification."""
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        return {
            "coef": np.asarray(clf.coef_).copy(),
            "intercept": np.asarray(clf.intercept_).copy(),
            "mean": np.asarray(scaler.mean_).copy(),
            "scale": np.asarray(scaler.scale_).copy(),
        }
    if hasattr(model, "coef_"):
        return {
            "coef": np.asarray(model.coef_).copy(),
            "intercept": np.asarray(model.intercept_).copy(),
        }
    if hasattr(model, "get_params"):
        return {"params": copy.deepcopy(model.get_params(deep=False))}
    return {}


def assert_model_frozen(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in before:
        if key not in after:
            raise AssertionError(f"Frozen model snapshot key missing after calibration: {key}")
        b_val, a_val = before[key], after[key]
        if isinstance(b_val, np.ndarray):
            if not np.array_equal(b_val, a_val):
                raise AssertionError("Frozen Phase 3 model was mutated during calibration")
        elif b_val != a_val:
            raise AssertionError("Frozen Phase 3 model was mutated during calibration")


def get_benign_class_index(model: Any, metadata: dict[str, Any]) -> int:
    order = metadata.get("predicted_classes_order")
    if order is not None:
        classes = [int(c) for c in order]
    elif hasattr(model, "classes_"):
        classes = [int(c) for c in model.classes_]
    elif hasattr(model, "named_steps"):
        classes = [int(c) for c in model.named_steps["clf"].classes_]
    else:
        classes = list(EXPECTED_CLASSES)

    if BENIGN_CLASS not in classes:
        raise ArtifactValidationError("benign class 0 not found in model.classes_")
    return classes.index(BENIGN_CLASS)
