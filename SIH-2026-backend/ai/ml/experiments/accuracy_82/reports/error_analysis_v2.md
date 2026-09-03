# Error Analysis v2

Dataset: `C:\Users\kirtan ishankumar\Desktop\ai\ml\training_data_v2_clean`
Partition: `clean_development (model_train + cal_dev)`
Samples: 2037

## Dominant confusion pairs

| True | Predicted | Count | Rate within true |
|------|-----------|-------|------------------|
| fsk | tone | 149 | 0.393 |
| benign | ook | 57 | 0.140 |
| ook | benign | 52 | 0.128 |
| ook | tone | 50 | 0.123 |
| tone | benign | 23 | 0.058 |
| tone | fsk | 17 | 0.043 |
| fsk | ook | 11 | 0.029 |
| fsk | benign | 5 | 0.013 |
| chirp | fsk | 5 | 0.011 |
| benign | tone | 3 | 0.007 |

## Known pair profiles

### fsk to tone (n=149)
- SNR mean: 10.47 dB
- Frequency mean: 18950.5 Hz
- Frequency deviation mean: 1256.6 Hz
- Duty cycle distribution: {}
- Harmonic ratio mean: 0.9762
- Spectral flatness mean: 0.0110

### ook to benign (n=52)
- SNR mean: 6.56 dB
- Frequency mean: 19471.4 Hz
- Duty cycle distribution: {}
- Harmonic ratio mean: 0.2724
- Spectral flatness mean: 0.4399

### ook to tone (n=50)
- SNR mean: 9.07 dB
- Frequency mean: 19429.6 Hz
- Duty cycle distribution: {}
- Harmonic ratio mean: 0.9672
- Spectral flatness mean: 0.0178

### tone to benign (n=23)
- SNR mean: 23.58 dB
- Frequency mean: 19745.2 Hz
- Duty cycle distribution: {}
- Harmonic ratio mean: 0.9827
- Spectral flatness mean: 0.0077

## Metadata error correlates (incorrect vs correct)

- snr: delta_mean=+0.0641
- amplitude: delta_mean=-0.0004
- frequency: delta_mean=-201.8130
- frequency_deviation: delta_mean=+28.3617
- bit_rate: delta_mean=-9.0511
- peak_amplitude: delta_mean=-0.1409
- source_duration_sec: delta_mean=-0.0009
- crop_attempts: delta_mean=+0.0987

Benign false positives: 60
Threat false negatives: 80