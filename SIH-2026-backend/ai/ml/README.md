# ai/ml — AI 2 (Detection / ML)

**Owner:** Denish (AI 2) · **Feature/DSP layer:** AI 1 (Kashish)

Everything under `ai/ml/` is owned by AI 2. **Nothing under `ai/dsp/` is edited
here.** The DSP layer (32-feature extraction, preprocessing, STFT, detection
rules, `ThreatEvent` contract) is pinned by 111 tests and is off-limits.

---

## Important caveat about THIS folder's data

The real training data comes from `DSPPipeline.generate_training_dataset()`
(live-preprocessed 32 features). That DSP layer is **not present in this
workspace**, so `data_generation.py` synthesises a stand-in dataset that obeys
the exact same 32-feature contract.

**Every number produced here is therefore SYNTHETIC and must be labelled as
such — it is NOT over-the-air accuracy.** When real data arrives, replace the
generator call in `phase1_baseline.load_or_generate()` with the real
`DSPPipeline` call; the rest of the pipeline is unchanged.

---

## How to run (canonical pipeline)

From the `ai/` directory:

```bash
# Phase 3 — train baselines, select model (Dataset V2)
python -m ml.phase3.train --seed 42 --output-dir ml/output/phase3

# Phase 4 — calibrate risk on frozen Phase 3 model
python -m ml.phase4.train --seed 42 --output-dir ml/output/phase4

# Phase 5 — frozen inference service evaluation
python -m ml.phase5.train --seed 42 --output-dir ml/output/phase5
```

Tests:

```bash
python -m pytest tests/test_phase3/ tests/test_phase4/ tests/test_phase5/ -v
```

### requirements
Add to `ai/requirements.txt` when committing:

```
scikit-learn>=1.3.0
joblib>=1.2.0
matplotlib>=3.6        # only needed for the .png reports
```

---

## Phases / deliverables (handoff plan §10)

| # | Deliverable | Script | Output |
|---|---|---|---|
| 1 | Dataset strategy + baseline | `data_generation.py`, `phase1_baseline.py` | `training_data/`, `output/baseline_report.txt` |
| 2 | Model comparison + justification | `phase2_model_comparison.py` | `output/model_comparison.*`, `output/feature_importance.*` |
| 3 | Model comparison + selection | `ml/phase3/train.py` | `output/phase3/selected_model.joblib` |
| 4 | Risk calibration | `ml/phase4/train.py` | `output/phase4/risk_calibrator.joblib` |
| 5 | Inference service | `ml/phase5/train.py` | `output/phase5/inference_service.joblib` |

---

## Audit rules honoured (handoff plan §7)

* **No unmeasured accuracy claims** — every number comes from a real eval run
  in `metrics.evaluate()` / `cross_val_score`.
* **Keep `confidence` / `suspicion_score` / `risk` distinct** — the service
  exposes all three as separate fields (`confidence` = label certainty,
  `suspicion_score` = 1−P(benign), `risk` = band label).
* **No independent backend changes** — this folder changes nothing Backend 1
  consumes; schema changes go AI 2 → AI 1 → Backend 1.
* **Synthetic ≠ real** — every report records `source: synthetic`.
* **Report FPR alongside accuracy, always** — `metrics.evaluate()` returns
  per-class and threat-level FPR, printed in Phases 1 and 3.

---

## Off-limits (per AI 1 handoff plan §3)

Do not modify: `dsp/feature_extraction.py`, `dsp/audio_preprocessing.py`,
`dsp/dsp_api.py`, `dsp/frequency_analyzer.py`, `dsp/segmentation.py`,
`dsp/stft.py`, `dsp/spectrogram_generator.py`, `tests/*`, or the 32-feature
order/names. The one exception AI 2 is cleared for: calibrate
`DSPPipeline.RISK_THRESHOLDS` (Phase 4).
