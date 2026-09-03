# DSP / Feature-Space Realism Audit

Investigation-only experiment. Does **not** train, promote, or modify frozen production artifacts.

## Run

```bash
cd ai
python -m ml.experiments.realism_validation.run_audit
python -m pytest tests/test_experiments/test_realism_validation.py -q
```

## Scope

- Uses `training_data_v2_clean/` development partitions only (model_train + cal_dev)
- **Never** loads locked-test indices
- Frozen baseline locked-test numbers are reference-only in reports

## Outputs

- `reports/realism_audit.json` — master report
- `reports/feature_separability.json` / `.csv`
- `reports/confusion_boundary_analysis.json`
- `reports/dataset_diversity_report.json`
- `reports/generation_parameter_analysis.json`
- `plots/` — distribution and pairwise scatter plots
- `ai/docs/AI_ML_DSP_REALISM_AUDIT.md` — handoff document
