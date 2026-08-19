# Phase 5 — Frozen ML Inference / Service Layer

Phase 5 is an **inference and service layer only**. It loads frozen Phase 3 and
Phase 4 artifacts and exposes a deterministic prediction API. **No model
training or recalibration occurs in this phase.**

## Frozen Inputs

| Artifact | Path |
|----------|------|
| Phase 3 classifier | `ai/ml/output/phase3/selected_model.joblib` |
| Phase 3 metadata | `ai/ml/output/phase3/selected_model_metadata.json` |
| Phase 4 calibrator | `ai/ml/output/phase4/risk_calibrator.joblib` |
| Phase 4 metadata | `ai/ml/output/phase4/calibration_metadata.json` |

## API

```python
from ml.phase5 import InferenceService

service = InferenceService.from_artifacts()

# Single sample (32 features, Phase-1 order)
result = service.predict(features)

# Batch (N x 32)
results = service.predict_batch(features_batch)
```

### Single-sample output

```json
{
  "predicted_class_id": 0,
  "predicted_class": "benign",
  "confidence": 0.42,
  "class_probabilities": {"benign": 0.42, "fsk": 0.1, ...},
  "threat_score": 0.58,
  "threat_probability": 0.55,
  "calibrated_risk_score": 0.51,
  "risk_level": "LOW"
}
```

- **confidence** — calibrated probability of the predicted class
- **threat_score** — raw (uncalibrated) threat probability from frozen classifier
- **threat_probability** / **calibrated_risk_score** — Phase 4 calibrated values
- **risk_level** — `LOW` / `MEDIUM` / `HIGH` from Phase 4 thresholds

## Risk Thresholds (from Phase 4, not re-derived)

- `HIGH` — calibrated risk ≥ high_threshold (≈ 0.6441)
- `MEDIUM` — calibrated risk ≥ medium_threshold (≈ 0.5278)
- `LOW` — calibrated risk < medium_threshold

## Input Validation

Each request validates:
- exactly 32 numeric features
- finite values (no NaN / Inf)
- Phase-1 feature order (via metadata contract)

## Run

From the `ai` directory:

```bash
python -m ml.phase5.train --seed 42 --output-dir ml/output/phase5
```

This loads frozen artifacts, validates compatibility, runs inference on Dataset V2,
and writes reports. It does **not** train or recalibrate anything.

## Tests

```bash
cd ai
python -m pytest tests/test_phase5/ -v
```

## Output Artifacts (`ai/ml/output/phase5/`)

| File | Description |
|------|-------------|
| `inference_report.json` | Summary report |
| `inference_metrics.json` | Full evaluation metrics |
| `inference_predictions.csv` | Per-sample inference outputs |
| `inference_metadata.json` | Provenance and thresholds |
| `inference_service.joblib` | Serializable `InferenceService` |

## Limitations

- Inference quality reflects synthetic Dataset V2 and frozen Phase 3/4 behavior.
- High benign false-positive rates from earlier phases propagate through.
- Not integrated with backend/frontend in this phase.

## Phase Boundaries

- Phase 1–4: read-only (not modified)
- **Phase 5: inference/service only**
- Backend / frontend: not touched
