# AI 2 — Detection / ML Results Summary

**Data source:** `dsp_synthetic` — synthetic AUDIO through the real
`dsp.FeatureExtractor` (AI 1's pipeline).
**Not** over-the-air data. Do not quote these as real-world accuracy.

## Provenance tiers

| Tier | Meaning | Used here |
|---|---|---|
| `dsp_synthetic` | Synthetic audio → real DSP feature extractor | **yes** |
| `fabricated` | Invented feature vectors, no audio, no FFT | no |
| `dsp_real` | Real over-the-air recordings → real DSP | blocked on AI 1's sweep |

An earlier revision used `fabricated` data and labelled it "synthetic". That
conflation produced a wrong Phase 2 conclusion — see §2.

Data: 1,500 samples × 32 features, 300/class, labels 0=benign 1=fsk 2=ook
3=chirp 4=tone. Reproduce with `python run_all.py --samples_per_class 300`.

---

## 1. Baseline (Phase 1) — Random Forest, 100 trees

Held-out 20%, stratified. **Accuracy 0.84**

| class | precision | recall | F1 | FPR | support |
|---|---|---|---|---|---|
| benign | 0.78 | 0.87 | 0.82 | 0.062 | 60 |
| fsk | 0.70 | 0.63 | 0.67 | 0.067 | 60 |
| ook | 0.78 | 0.75 | 0.76 | 0.054 | 60 |
| chirp | 1.00 | 1.00 | 1.00 | 0.000 | 60 |
| tone | 0.95 | 0.97 | 0.96 | 0.013 | 60 |

Threat-level FPR (benign → any threat): **0.133** (8/60).
Inference latency ≈ **0.026 ms/sample**.

FSK and OOK are where accuracy is lost; chirp and tone are solved. This
matches the DSP layer's known weakness (rule-based OOK ≈ 75%), so the two are
independent evidence of the same difficulty.

---

## 2. Model comparison (Phase 2) — 5-fold CV

| model | scaled | cv_acc | std | f1_macro |
|---|---|---|---|---|
| RandomForest | no | **0.8527** | 0.0125 | 0.8494 |
| SVM_rbf | yes | 0.8280 | 0.0134 | 0.8249 |
| LogisticRegression | yes | 0.8140 | 0.0161 | 0.8084 |
| SVM_rbf_UNSCALED | no | 0.4807 | 0.0215 | 0.4409 |

**Finding — feature scaling is not optional.** Unscaled RBF SVM scores
**0.481**; with `StandardScaler` it reaches **0.828** (**+0.347**). Real DSP
features span very different ranges (`energy_db` negative, `peak_frequency`
in the thousands, `spectral_flatness` 0–1). RBF SVM is distance-based and
LogisticRegression shares one regularisation strength, so both are dominated
by the largest-magnitude feature. Random Forest splits per feature and is
scale-invariant, which is why it was unaffected.

The unscaled row is kept in the table deliberately as evidence.

**Selection: Random Forest.** Highest measured CV accuracy, plus native
feature importances (a required deliverable), no scaling dependency at serve
time, explainable, trains in seconds.

**Top features:** _(fill from `output/feature_importance.txt`)_
Features contributing < 0.005 are listed there — a feature that does nothing
is a real finding worth reporting.

---

## 3. Evaluation (Phase 3)

_(fill from `output/evaluation_report.txt`)_

Confusion is concentrated in FSK ↔ OOK. Plots: `output/confusion_final.png`,
`output/calibration_curve.png`.

---

## 4. Risk calibration (Phase 4)

Risk = **1 − P(benign)**. Calibrated against benign-FPR budgets on validation:

| band | cut | achieved benign FPR | target |
|---|---|---|---|
| HIGH | ≥ 0.947 | 0.040 | ≤ 0.05 |
| MEDIUM | ≥ 0.603 | 0.147 | ≤ 0.15 |
| LOW | < 0.603 | — | — |

These replace AI 1's placeholder `RISK_THRESHOLDS` (0.75 / 0.45), which were
never validated against a measured FPR. **Coordinate with AI 1 before changing
`dsp_api.py`** — Backend 1's audit point 10.

---

## 5. Inference service (Phase 5)

`classify(32_features)` returns, as **separate fields** (audit point 9):

```json
{
  "pattern": "fsk",
  "pattern_id": 1,
  "confidence": 0.51,
  "suspicion_score": 1.0,
  "risk": "HIGH",
  "probabilities": { "benign": 0.0, "fsk": 0.51, "ook": 0.49 }
}
```

- `confidence` — certainty of the *pattern label*
- `suspicion_score` — how threat-like (= 1 − P(benign))
- `risk` — HIGH / MEDIUM / LOW band

Input must be the 32 features from `DSPPipeline.process_features_only()` —
same preprocessing path as training, which is what prevents train/serve skew.

Held-out benign promoted to HIGH: **0/60**.

---

## Reproduce

```bash
cd ai/ml
python run_all.py --samples_per_class 500
```

Requires the `dsp` package importable from `ai/`. If it is not,
`data_generation.py` falls back to fabricated vectors and prints a loud
warning — those numbers are not measurements.

## Next steps

1. Fold in Backend 2's WAV files (Source B) once available.
2. Fold in AI 1's real over-the-air recordings (Source C) after the sweep.
3. Keep synthetic and real strictly separated in train/test — no leakage.
4. Report real-data numbers with `source: dsp_real` and keep this page as the
   synthetic baseline.
