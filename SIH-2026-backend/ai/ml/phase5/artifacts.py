"""Load and validate frozen Phase 3 + Phase 4 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import (
    ARTIFACT_VERSION,
    CLASS_NAMES,
    EXPECTED_CLEAN_MODEL,
    EXPECTED_PHASE3_MODEL,
    FEATURE_NAMES,
    LEGACY_PHASE3_MODEL_PATH,
    LEGACY_PHASE4_CALIBRATOR_PATH,
    ML_DIR,
    MODEL_CALIBRATION,
    N_FEATURES,
    PHASE3_CLEAN_METADATA_PATH,
    PHASE3_CLEAN_MODEL_PATH,
    PHASE3_METADATA_PATH,
    PHASE3_MODEL_PATH,
    PHASE4_CALIBRATOR_PATH,
    PHASE4_CLEAN_CALIBRATOR_PATH,
    PHASE4_METADATA_PATH,
    PHASE4_METRICS_PATH,
    PHASE5_CLEAN_METADATA_PATH,
    PHASE5_CLEAN_RISK_POLICY_PATH,
)


class ArtifactValidationError(ValueError):
    """Raised when frozen artifacts fail validation."""


class ArtifactCompatibilityError(ValueError):
    """Raised when Phase 3 and Phase 4 artifacts are incompatible."""


def _normalize_path_reference(path_value: str | Path) -> str:
    return str(path_value).replace("\\", "/")


def _canonical_clean_suffix(canonical: Path) -> str:
    return canonical.relative_to(ML_DIR).as_posix()


def metadata_references_canonical_clean_artifact(
    metadata_path: str | Path,
    canonical: Path,
) -> bool:
    """True when frozen metadata path suffix matches the canonical clean artifact."""
    suffix = _canonical_clean_suffix(canonical)
    normalized = _normalize_path_reference(metadata_path)
    if not normalized.endswith(suffix):
        return False
    partition = suffix.split("/")[1]
    if partition not in normalized:
        return False
    legacy_partition = partition.replace("_clean", "")
    if f"output/{legacy_partition}/" in normalized and partition not in normalized:
        return False
    return True


def _validate_clean_artifact_path_reference(
    path_value: str | Path,
    canonical: Path,
) -> None:
    """Validate a frozen metadata path refers to the canonical clean artifact."""
    suffix = _canonical_clean_suffix(canonical)
    normalized = _normalize_path_reference(path_value)
    partition = suffix.split("/")[1]
    if not normalized.endswith(suffix):
        raise ArtifactValidationError(
            f"Clean artifact path must end with {suffix!r}, got {path_value!r}"
        )
    if partition not in normalized:
        raise ArtifactValidationError(
            f"Clean artifact path must reference {partition!r}, got {path_value!r}"
        )
    legacy_partition = partition.replace("_clean", "")
    legacy_marker = f"output/{legacy_partition}/"
    if legacy_marker in normalized and partition not in normalized:
        raise ArtifactValidationError(
            f"Clean artifact path must not reference legacy {legacy_partition!r}"
        )


def _assert_canonical_clean_path(requested: Path, canonical: Path) -> None:
    if requested.resolve() != canonical.resolve():
        raise ArtifactCompatibilityError(
            f"Requested path must be canonical clean artifact {canonical}, got {requested}"
        )
    normalized = _normalize_path_reference(requested)
    if f"output/{canonical.parent.name}/" not in normalized:
        raise ArtifactCompatibilityError(
            f"Requested path must live under clean output partition, got {requested}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_phase3_model(path: Path | None = None) -> Any:
    model_path = Path(path) if path is not None else PHASE3_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Phase 3 model not found: {model_path}")
    return joblib.load(model_path)


def load_phase3_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata = _load_json(Path(path) if path is not None else PHASE3_METADATA_PATH)
    validate_phase3_metadata(metadata)
    return metadata


def load_phase4_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata = _load_json(Path(path) if path is not None else PHASE4_METADATA_PATH)
    validate_phase4_metadata(metadata)
    return metadata


def load_phase4_metrics(path: Path | None = None) -> dict[str, Any]:
    return _load_json(Path(path) if path is not None else PHASE4_METRICS_PATH)


def load_risk_calibrator(path: Path | None = None):
    calibrator_path = Path(path) if path is not None else PHASE4_CALIBRATOR_PATH
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Phase 4 calibrator not found: {calibrator_path}")
    from ml.phase4.calibrator import RiskCalibrator
    return RiskCalibrator.load(calibrator_path)


def validate_phase3_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("model_name") != EXPECTED_PHASE3_MODEL:
        raise ArtifactValidationError(
            f"Expected Phase 3 model {EXPECTED_PHASE3_MODEL!r}, "
            f"got {metadata.get('model_name')!r}"
        )
    _validate_feature_contract(metadata.get("feature_names"), "Phase 3 metadata")
    _validate_class_mapping(metadata.get("class_mapping"), "Phase 3 metadata")


def validate_phase4_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("phase3_model_name") != EXPECTED_PHASE3_MODEL:
        raise ArtifactValidationError(
            f"Phase 4 metadata references unexpected model: "
            f"{metadata.get('phase3_model_name')!r}"
        )
    _validate_feature_contract(metadata.get("feature_names"), "Phase 4 metadata")
    _validate_class_mapping(metadata.get("class_mapping"), "Phase 4 metadata")
    thresholds = metadata.get("risk_thresholds")
    if not thresholds or "high_threshold" not in thresholds or "medium_threshold" not in thresholds:
        raise ArtifactValidationError("Phase 4 metadata missing risk_thresholds")


def validate_artifact_compatibility(
    phase3_metadata: dict[str, Any],
    phase4_metadata: dict[str, Any],
) -> None:
    if list(phase3_metadata["feature_names"]) != list(phase4_metadata["feature_names"]):
        raise ArtifactCompatibilityError("Phase 3/4 feature_names mismatch")
    p3_map = {int(k): v for k, v in phase3_metadata["class_mapping"].items()}
    p4_map = {int(k): v for k, v in phase4_metadata["class_mapping"].items()}
    if p3_map != p4_map or p3_map != CLASS_NAMES:
        raise ArtifactCompatibilityError("Phase 3/4 class_mapping mismatch")


def _validate_feature_contract(feature_names: Any, source: str) -> None:
    if feature_names is None:
        raise ArtifactValidationError(f"{source} missing feature_names")
    if len(feature_names) != N_FEATURES:
        raise ArtifactValidationError(
            f"{source} expected {N_FEATURES} features, got {len(feature_names)}"
        )
    if list(feature_names) != list(FEATURE_NAMES):
        raise ArtifactValidationError(
            f"{source} feature_names do not match Phase-1 authoritative order"
        )


def _validate_class_mapping(class_mapping: Any, source: str) -> None:
    if class_mapping is None:
        raise ArtifactValidationError(f"{source} missing class_mapping")
    mapping = {int(k): v for k, v in class_mapping.items()}
    if mapping != CLASS_NAMES:
        raise ArtifactValidationError(f"{source} class_mapping mismatch: {mapping}")


def snapshot_frozen_model(model: Any) -> dict[str, np.ndarray]:
    """Capture fitted weights to verify the classifier stays frozen."""
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
    return {}


def assert_model_unchanged(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    for key in before:
        if key not in after or not np.array_equal(before[key], after[key]):
            raise AssertionError("Frozen Phase 3 model was mutated during inference")


def validate_clean_phase3_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("model_name") != EXPECTED_CLEAN_MODEL:
        raise ArtifactValidationError(
            f"Expected clean Phase 3 model {EXPECTED_CLEAN_MODEL!r}, "
            f"got {metadata.get('model_name')!r}"
        )
    _validate_feature_contract(metadata.get("feature_names"), "Clean Phase 3 metadata")
    _validate_class_mapping(metadata.get("class_mapping"), "Clean Phase 3 metadata")


def validate_clean_inference_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("model_calibration") != MODEL_CALIBRATION:
        raise ArtifactValidationError(
            f"Expected model_calibration={MODEL_CALIBRATION!r}, "
            f"got {metadata.get('model_calibration')!r}"
        )
    if metadata.get("uses_legacy_artifacts") is True:
        raise ArtifactValidationError("Clean inference metadata must not use legacy artifacts")
    if metadata.get("calibration_status") != "NOT_USED_IN_FINAL_INFERENCE":
        raise ArtifactValidationError(
            "Clean inference metadata must record calibration_status="
            "'NOT_USED_IN_FINAL_INFERENCE'"
        )
    phase3_path = str(metadata.get("phase3_artifact_path", "")).replace("\\", "/")
    if "phase3_clean" not in phase3_path:
        raise ArtifactValidationError(
            f"Clean inference must reference phase3_clean artifact, got {phase3_path!r}"
        )
    legacy = "output/phase3/selected_model.joblib"
    if legacy in phase3_path:
        raise ArtifactValidationError("Clean inference references legacy Phase 3 artifact")
    _validate_feature_contract(metadata.get("feature_names"), "Clean inference metadata")
    _validate_class_mapping(metadata.get("class_mapping"), "Clean inference metadata")
    thresholds = metadata.get("risk_thresholds")
    if not thresholds or "high_threshold" not in thresholds or "medium_threshold" not in thresholds:
        raise ArtifactValidationError("Clean inference metadata missing risk_thresholds")


def load_clean_phase3_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata = _load_json(Path(path) if path is not None else PHASE3_CLEAN_METADATA_PATH)
    validate_clean_phase3_metadata(metadata)
    return metadata


def load_clean_inference_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata = _load_json(Path(path) if path is not None else PHASE5_CLEAN_METADATA_PATH)
    validate_clean_inference_metadata(metadata)
    return metadata


def load_risk_policy(path: Path | None = None):
    policy_path = Path(path) if path is not None else PHASE5_CLEAN_RISK_POLICY_PATH
    if not policy_path.exists():
        raise FileNotFoundError(f"Clean risk policy not found: {policy_path}")
    from ml.promotion.uncalibrated_adapter import UncalibratedRiskAdapter

    return UncalibratedRiskAdapter.load(policy_path)


def load_clean_artifacts(
    *,
    model_path: Path | None = None,
    phase3_metadata_path: Path | None = None,
    risk_policy_path: Path | None = None,
    phase5_metadata_path: Path | None = None,
) -> dict[str, Any]:
    phase3_metadata = load_clean_phase3_metadata(phase3_metadata_path)
    inference_metadata = load_clean_inference_metadata(phase5_metadata_path)

    model_path_resolved = Path(model_path) if model_path is not None else PHASE3_CLEAN_MODEL_PATH
    _assert_canonical_clean_path(model_path_resolved, PHASE3_CLEAN_MODEL_PATH)
    _validate_clean_artifact_path_reference(
        inference_metadata["phase3_artifact_path"],
        PHASE3_CLEAN_MODEL_PATH,
    )
    if inference_metadata.get("risk_policy_path") is not None:
        _validate_clean_artifact_path_reference(
            inference_metadata["risk_policy_path"],
            PHASE5_CLEAN_RISK_POLICY_PATH,
        )

    if model_path_resolved.resolve() == LEGACY_PHASE3_MODEL_PATH.resolve():
        raise ArtifactCompatibilityError("Clean loader cannot use legacy Phase 3 model")

    model = load_phase3_model(model_path_resolved)
    risk_policy_path = (
        Path(risk_policy_path)
        if risk_policy_path is not None
        else PHASE5_CLEAN_RISK_POLICY_PATH
    )
    _assert_canonical_clean_path(risk_policy_path, PHASE5_CLEAN_RISK_POLICY_PATH)
    risk_policy = load_risk_policy(risk_policy_path)

    legacy_ref = str(inference_metadata.get("legacy_phase4_calibrator_path", "")).replace(
        "\\", "/"
    )
    if LEGACY_PHASE4_CALIBRATOR_PATH.as_posix() in legacy_ref and inference_metadata.get(
        "loads_legacy_phase4_calibrator"
    ):
        raise ArtifactValidationError("Clean inference must not load legacy Phase 4 calibrator")

    return {
        "model": model,
        "phase3_metadata": phase3_metadata,
        "inference_metadata": inference_metadata,
        "risk_policy": risk_policy,
        "model_snapshot": snapshot_frozen_model(model),
    }


def load_all_artifacts(
    *,
    model_path: Path | None = None,
    phase3_metadata_path: Path | None = None,
    calibrator_path: Path | None = None,
    phase4_metadata_path: Path | None = None,
) -> dict[str, Any]:
    phase3_metadata = load_phase3_metadata(phase3_metadata_path)
    phase4_metadata = load_phase4_metadata(phase4_metadata_path)
    validate_artifact_compatibility(phase3_metadata, phase4_metadata)

    model = load_phase3_model(model_path)
    calibrator = load_risk_calibrator(calibrator_path)

    return {
        "model": model,
        "phase3_metadata": phase3_metadata,
        "phase4_metadata": phase4_metadata,
        "calibrator": calibrator,
        "model_snapshot": snapshot_frozen_model(model),
    }
