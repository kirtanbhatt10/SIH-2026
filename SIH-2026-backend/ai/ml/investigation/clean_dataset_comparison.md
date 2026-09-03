# Clean Dataset Comparison

| Metric | Old (saved model, old data) | New (clean data, linear_svm_balanced) | Change |
|--------|------------------------------|----------------------------------------|--------|
| accuracy | 0.6811 | 0.8078 | +0.1266 |
| macro_f1 | 0.6825 | 0.8342 | +0.1517 |
| benign_fpr | 0.6211 | 0.0860 | -0.5350 |
| benign_recall | 0.3789 | 0.9140 | +0.5350 |
| threat_recall | 0.7647 | 0.8175 | +0.0527 |

## Duplicate stats
- Old cross-split exact matches: 6
- Clean cross-split exact matches: 0

## Backend status
**BLOCKED** — synthetic validation only; verify FPR targets before integration.