# AI/ML DSP & Feature-Space Realism Audit

Generated: 2026-08-22T10:58:36.875358+00:00

## Executive summary

Primary bottleneck: **D: CLASS OVERLAP IS THE PRIMARY BOTTLENECK**
Confidence: HIGH

This audit investigated why the frozen `linear_svm_balanced` baseline achieves ~79.48% locked-test accuracy. **No frozen artifacts were modified.** **Locked test was not accessed.**

## Data used

- Dataset: `C:\Users\kirtan ishankumar\Desktop\SIH-2026-backend-dinl(1)\SIH-2026-backend\ai\ml\training_data_v2_clean`
- Partitions: model_train + cal_dev (2037 samples)
- Locked test accessed: **False**

## Frozen assets protected

- phase3_clean/, phase4_clean/, phase5_clean/ — untouched
- training_data_v2_clean/ — untouched
- frozen_split_manifest.json — untouched

## Error boundary analysis

| Pair | Verified count | Historical | Match |
|------|----------------|------------|-------|
| fsk → to → tone | 149 | 149 | yes |
| benign → to → ook | 57 | 57 | yes |
| ook → to → benign | 52 | 52 | yes |
| ook → to → tone | 50 | 50 | yes |
| tone → to → benign | 23 | 23 | yes |

## Feature analysis

Top global features (MI): magnitude_range, max_magnitude, peak_magnitude, mean_magnitude, peak_prominence
Top FSK vs Tone: harmonic_ratio, mean_magnitude, min_magnitude, coefficient_of_variation, tonal_prominence
Top OOK vs Benign: amplitude_envelope_std, temporal_flatness, onset_strength, bit_rate_estimate, spectral_kurtosis

## Pairwise class overlap

- **fsk_vs_tone**: centroid_dist=2.10, kNN purity=0.71, overlapping=True
- **ook_vs_benign**: centroid_dist=3.83, kNN purity=0.81, overlapping=True
- **ook_vs_tone**: centroid_dist=5.58, kNN purity=0.86, overlapping=False
- **tone_vs_benign**: centroid_dist=5.64, kNN purity=0.96, overlapping=False

## Realism scorecard

| Area | Status | Evidence |
|------|--------|----------|
| Benign diversity | PASS | SNR/frequency/amplitude spread + duplicate rate |
| OOK diversity | PASS | Parameter std + NN distance |
| FSK diversity | PASS | Frequency deviation spread |
| Tone diversity | PASS | Carrier frequency spread |
| FSK/Tone separation | FAIL | Centroid distance + kNN purity |
| OOK/Benign separation | FAIL | Duty cycle + envelope overlap |
| OOK/Tone separation | PASS | Pairwise Mahalanobis + overlap |
| DSP preservation | WARN | Single-window aggregation + squelch noted in dsp audit |
| Synthetic shortcuts | WARN | 2 medium/high shortcuts flagged |
| Feature sufficiency | FAIL | Top FSK/tone features: ['harmonic_ratio', 'mean_magnitude', 'min_magnitude'] |
| Dataset realism | WARN | Classes overlap in 32-D space despite clean validation |

## Primary conclusion

**D: CLASS OVERLAP IS THE PRIMARY BOTTLENECK**

Secondary causes:
- Single-chunk FFT + squelch limit modulation-sensitive discrimination
- Metadata/feature shortcuts indicate narrow synthetic manifolds

Recommended next experiment: Prototype multi-window / modulation-aware features on regenerated diagnostic audio only; evaluate on clean development split without touching locked test.

## What should NOT be changed

- Phase-1 FeatureExtractor (32-feature contract)
- Frozen split and locked test
- Production model artifacts

## Reports

- `ai/ml/experiments/realism_validation/reports/realism_audit.json`
- `ai/ml/experiments/realism_validation/plots/`