# Phase 4 — Risk Calibration

Phase 4 builds a deterministic, testable **risk-calibration layer** on top of the
frozen Phase 3 classifier (`logistic_regression`). It does **not** retrain the
classifier or modify Phase 3 artifacts.

## Inputs

| Input | Path |
|-------|------|
| Frozen classifier | `ai/ml/output/phase3/selected_model.joblib` |
| Phase 3 metadata | `ai/ml/output/phase3/selected_model_metadata.json` |
| Dataset V2 | `ai/ml/training_data_v2/` (`features.npy`, `labels.npy`, `metadata.csv`) |

Metadata is validated for:
- exactly 32 Phase-1 feature names in authoritative order
- class mapping `0=benign, 1=fsk, 2=ook, 3=chirp, 4=tone`
- Phase 3 split policy (`stratified_grouped_by_generation_group`, seed 42)

## Train / Calibration / Test Separation

1. **Split** — Reproduce the exact Phase 3 grouped split (seed 42, 20% test).
   No `generation_group` may appear in both train and test.
2. **Frozen classifier** — Phase 3 model is loaded and never refit.
3. **Calibration fit** — Uses **train partition only** (n=2039).
4. **Evaluation** — Held-out **test partition** (n=461) is never used for fitting.

## Calibration Method

1. **Multiclass probability calibration** — `CalibratedClassifierCV(method='isotonic', cv='prefit')`
   wraps a copy of the frozen model. Only the calibration mapping is fit on train data.
2. **Threat/risk score calibration** — Isotonic regression maps raw threat probability
   (`1 - P(benign)` from frozen model) to empirical threat prevalence on train.
3. **Risk thresholds** — Data-driven on train benign `calibrated_risk_score`:
   - **HIGH**: benign FPR into HIGH band ≤ 5%
   - **MEDIUM+**: benign FPR into MEDIUM or HIGH ≤ 15%

## Risk Score Definitions

| Field | Meaning |
|-------|---------|
| `confidence` | Calibrated probability of the predicted class |
| `threat_score` | Raw (uncalibrated) threat probability from frozen classifier |
| `threat_probability` | Sum of calibrated probabilities over threat classes |
| `calibrated_risk_score` | Isotonic-calibrated threat risk in [0, 1] |
| `risk_level` | `LOW` / `MEDIUM` / `HIGH` from calibrated risk score |

Confidence and threat/risk are kept conceptually separate.

## Thresholds

Configured in `config.py`:

- `TARGET_HIGH_BENIGN_FPR = 0.05`
- `TARGET_MEDIUM_PLUS_BENIGN_FPR = 0.15`

Actual cut points are computed from train benign scores and saved in artifacts.

## Run

From the `ai` directory:

```bash
python -m ml.phase4.train --seed 42 --output-dir ml/output/phase4
```

## Tests

```bash
cd ai
python -m pytest tests/test_phase4/ -v
```

## Output Artifacts (`ai/ml/output/phase4/`)

| File | Description |
|------|-------------|
| `calibration_report.json` | Method, thresholds, split, metric summaries |
| `calibration_metrics.json` | Train/test calibration and classification metrics |
| `calibration_metadata.json` | Provenance and definitions |
| `risk_distribution.csv` | LOW/MEDIUM/HIGH counts per partition |
| `calibrated_predictions.csv` | Per-sample calibrated outputs |
| `risk_calibrator.joblib` | Serializable fitted `RiskCalibrator` |

## Limitations

- All metrics are on **synthetic Dataset V2** only — not real-world/OTA performance.
- High benign false-positive rates from Phase 3 propagate into risk calibration.
- Isotonic calibration can overfit small train partitions.
- Risk bands are calibrated to synthetic benign statistics, not production telemetry.

## Phase Boundaries

- Phase 3 artifacts: read-only
- Backend / frontend: not touched
- Phase 5: not started
