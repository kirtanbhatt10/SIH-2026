# AI/ML Final Handoff

Generated: 2026-08-19T05:34:06.869607+00:00

## AI model status

**TECHNICALLY FROZEN / READY FOR INTEGRATION REVIEW**

NOT production validated. NOT OTA validated. Backend remains BLOCKED until team lead approves performance criteria.

## 1. Final model

- Model: `linear_svm_balanced` (sklearn Pipeline: StandardScaler + linear SVC)
- Calibration: `uncalibrated` (no isotonic in inference path)

## 2. Dataset version

- `dataset_v2_clean.0` at `C:\Users\kirtan ishankumar\Desktop\ai\ml\training_data_v2_clean`
- Generator: `dataset_v2_generator-1.2`

## 3. Feature contract

- Exactly **32 finite numeric features** in Phase-1 order (see `feature_names` in metadata)

## 4. Class mapping

| ID | Class |
|----|-------|
| 0 | benign |
| 1 | fsk |
| 2 | ook |
| 3 | chirp |
| 4 | tone |

## 5. Calibration decision

- `calibration_status`: `NOT_USED_IN_FINAL_INFERENCE`
- Reason: Uncalibrated Linear SVM achieved substantially lower benign FPR, higher macro F1, better Brier/ECE, better accuracy, and lower latency on the locked test. Isotonic improved threat recall by only 0.68 pp while increasing benign FPR by 12.9 pp.
- Phase 4 isotonic implementation preserved for research; **not loaded** by Phase 5 clean.

## 6. Final locked-test metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7948 |
| Macro F1 | 0.8246 |
| Weighted F1 | 0.7973 |
| Benign precision | 0.8750 |
| Benign recall | 0.9032 |
| Benign FPR | 0.0968 |
| Threat precision | 0.8277 |
| Threat recall | 0.8046 |
| Brier | 0.0414 |
| ECE | 0.0413 |
| Latency ms/sample | 0.0285 |

## 7. Artifact paths

- Phase 3 clean model: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase3_clean\selected_model.joblib`
- Phase 4 clean metadata (calibration not used): `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase4_clean\calibration_metadata.json`
- Phase 5 risk policy: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5_clean\risk_policy.joblib`
- Phase 5 inference service: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5_clean\inference_service.joblib`
- Phase 5 metadata: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5_clean\inference_metadata.json`
- Locked test evaluation: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5_clean\locked_test_evaluation.json`
- Frozen split: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\clean_splits\frozen_split_manifest.json`

## 8. Inference input contract

- Shape: `(32,)` or batch `(N, 32)`
- dtype: float64 after coercion
- All values finite (NaN/Inf rejected)
- Feature order: Phase-1 `FEATURE_NAMES`

## 9. Inference output contract

Minimum fields per prediction:

- `predicted_class_id`, `predicted_class`, `confidence`
- `class_probabilities` (raw model probabilities; sum to 1)
- `threat_score` (sum of threat-class probabilities; **not calibrated**)
- `risk_level` (`LOW` / `MEDIUM` / `HIGH`)
- `model_calibration`: `"uncalibrated"`

Load service:

```python
from ml.phase5.service import InferenceService
service = InferenceService.from_clean_artifacts()
# or: service = InferenceService.load_frozen()
result = service.predict(features_32)
```

## 10. Risk-level semantics

risk_level is derived from raw model threat_score (= sum of predict_proba mass on threat classes fsk/ook/chirp/tone). Thresholds are fit on cal_dev benign samples only to target HIGH<=5% and MEDIUM+<=15% benign band FPR on that partition. This is NOT isotonic calibration and does NOT imply calibrated probability semantics.

## 11. Known limitations

- Synthetic Dataset V2 clean only
- No over-the-air validation
- Chirp near-perfect separability is a synthetic artifact
- Performance thresholds require team sign-off

## 12. Backend integration instructions

1. Do **not** use legacy `output/phase3/` or `output/phase4/` artifacts.
2. Load `InferenceService.from_clean_artifacts()` or `InferenceService.load_frozen()`.
3. Pass exactly 32 Phase-1 features per inference call.
4. Treat outputs as synthetic-data validated only.
5. Complete OTA validation before production deployment.

## 13. OTA validation requirement

Mandatory before production: over-the-air capture on target hardware, with team-approved FPR/recall thresholds.

## 14. Backend / frontend

**Backend: NOT modified.** **Frontend: NOT modified.**

## Reproducibility

- Double inference match: True
- Reload match: True