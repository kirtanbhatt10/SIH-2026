# Accuracy Improvement Experiment

Isolated experiment area for controlled accuracy improvement **without**
modifying frozen production-review artifacts.

## Rules

- Dataset: `training_data_v2_clean/` (`dataset_v2_clean.0`)
- Split: `output/clean_splits/frozen_split_manifest.json`
- **Locked test is off limits** until one final one-shot evaluation
- Do **not** write to `output/phase3_clean/`, `phase4_clean/`, `phase5_clean/`

## Run

```bash
cd ai
python -m ml.experiments.accuracy_improvement.run_experiments
```

## Outputs

- `reports/` — JSON/MD analysis and comparisons
- `artifacts/` — experimental candidate model only
- `ai/docs/AI_ML_ACCURACY_IMPROVEMENT_REPORT.md`

## Selection policy

Rank on **cal_dev** by macro F1, benign FPR, threat recall, accuracy, latency.
Qualification targets:

- Accuracy >= 85%
- Macro F1 >= 84%
- Benign FPR <= 10%
- Threat recall >= 80%
