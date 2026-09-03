# AI/ML Final Validation Report

Generated: 2026-08-19T04:42:59.275464+00:00

## Final status

**NOT READY FOR AI SIGN-OFF**

Backend: **BLOCKED**

### Sign-off blockers

- Synthetic Dataset V2 clean data only — no over-the-air validation.
- Performance threshold requires team/project sign-off.
- Backend integration remains BLOCKED.

## 1. Dataset

- Version: `dataset_v2_clean.0`
- Path: `C:\Users\kirtan ishankumar\Desktop\ai\ml\training_data_v2_clean`
- Generator: `dataset_v2_generator-1.2`
- Clean validation: 19/19 checks passed (0 collapsed benign, 0 duplicate groups, 0 cross-split overlap)

## 2. Generator changes

- Added `steady_ultrasonic_hum` and `band_limited_hiss` benign variants
- Regenerate-on-collapsed-fingerprint policy
- Separate output dir `training_data_v2_clean/` preserves legacy `training_data_v2/`

## 3. Duplicate results

| Dataset | Collapsed benign | Duplicate groups | Cross-split exact |
|---------|------------------|------------------|-------------------|
| Old V2 | 25 | present | 6 |
| Clean V2 | 0 | 0 | 0 |

## 4. Split methodology

- Outer split seed: `42` (grouped held-out test)
- Cal-dev seed: `43` (289 samples)
- Model train: 1748 samples
- Locked test: 463 samples
- Frozen manifest: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\clean_splits\frozen_split_manifest.json`

## 5. Model selection

- Selected candidate: `linear_svm_balanced` (frozen before locked-test evaluation)
- Trained on full train partition (2037 samples)
- Calibration method chosen on cal_dev using model_fit trained on model_train only (1748 samples)

## 6. Calibration methodology

- Prior 7.5%/95.4% audit: Calibrator was NOT fit on the test set, but the 7.5%/95.4% headline numbers are INVALID for promotion because test-set metrics were used for selection and the methodology differs from the Phase 4 RiskCalibrator path.
- Calibrator fit partition: cal_dev only
- Decision: **isotonic** — Isotonic calibration selected on cal_dev: Brier 0.0504->0.0448, ECE 0.0639->0.0313, macro F1 0.8218->0.8524, benign FPR 0.0357->0.0357, threat recall 0.7903->0.8220.

### Cal-dev comparison

| Method | Brier | ECE | Benign FPR | Threat recall | Macro F1 |
|--------|-------|-----|------------|---------------|----------|
| uncalibrated | 0.0504 | 0.0639 | 0.0357 | 0.7903 | 0.8218 |
| isotonic | 0.0448 | 0.0313 | 0.0357 | 0.8220 | 0.8524 |

## 7. Final held-out metrics (ONE locked evaluation)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7732 |
| Macro F1 | 0.8027 |
| Weighted F1 | 0.7745 |
| Benign precision | 0.8780 |
| Benign recall | 0.7742 |
| Benign FPR | 0.2258 |
| Threat precision | 0.8075 |
| Threat recall | 0.8115 |
| Brier (threat) | 0.0461 |
| ECE (threat) | 0.0432 |
| Latency ms/sample | 0.0427 |

### Train partition metrics (not headline)

- Macro F1: 0.8048
- Benign FPR: 0.1652
- Threat recall: 0.7999

### Cal-dev metrics (selection only)

- Macro F1: 0.8524
- Benign FPR: 0.0357
- Threat recall: 0.8220

## 8. Comparison with old pipeline

| Metric | Legacy (old V2 + LR + calibration) | Clean gate (locked test) |
|--------|--------------------------------------|--------------------------|
| Benign FPR | ~0.821 | 0.2258 |
| Threat recall | ~0.817 (uncal clean exp.) | 0.8115 |
| Macro F1 | ~0.617 | 0.8027 |

## 9. Artifact paths

- Phase 3 clean: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase3_clean`
- Phase 4 clean: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase4_clean`
- Phase 5 clean: `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5_clean`
- Legacy (marked): `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase3`, `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase4`, `C:\Users\kirtan ishankumar\Desktop\ai\ml\output\phase5`

## 10. Reproducibility

- Double inference match: True
- Probability match: True
- Artifact reload match: True

## 11. Test results

Run: `cd ai && python -m pytest tests/test_promotion/ tests/test_phase3/ tests/test_phase4/ tests/test_phase5/ -v`

## 12. Known limitations

- Synthetic data only; hardware path not validated
- Clean chirp class remains near-perfect separability (expected synthetic artifact)
- Prior headline 7.5%/95.4% invalid for promotion (see calibration audit)
- Prior clean experiment 8.6% benign FPR was **uncalibrated** `linear_svm_balanced` with test-set-informed selection; this gate's locked-test FPR reflects **isotonic calibration** chosen on cal_dev only
- Isotonic calibration improved cal_dev metrics but locked-test benign FPR is 22.6% — do not extrapolate from pre-gate experiment numbers

## 13. Project performance requirements

No explicit SIH FPR/recall thresholds were found in repository docs. **Performance threshold requires team/project sign-off.**

Do NOT treat 22.6% benign FPR as automatically production-ready.

## 14. AI sign-off recommendation

**NOT READY FOR AI SIGN-OFF**

Technical promotion gate completed with versioned clean artifacts. Proceed to team review for performance policy and OTA validation before backend integration.