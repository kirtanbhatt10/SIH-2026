# Error Analysis

Partition: `cal_dev_baseline_reproduction`

## Dominant confusion pairs

| True | Predicted | Count | Rate within true |
|------|-----------|-------|------------------|
| fsk | tone | 13 | 0.302 |
| ook | benign | 10 | 0.130 |
| ook | tone | 9 | 0.117 |
| tone | benign | 5 | 0.075 |
| tone | fsk | 4 | 0.060 |
| fsk | benign | 3 | 0.070 |
| benign | ook | 2 | 0.036 |
| fsk | ook | 2 | 0.047 |
| ook | fsk | 2 | 0.026 |

Benign false positives: 2
Threat false negatives: 18