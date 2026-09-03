"""Main orchestrator for the DSP / feature-space realism audit."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .confusion_analysis import confusion_boundary_analysis
from .config import (
    EXPERIMENT_DIR,
    FROZEN_BASELINE,
    REPORTS_DIR,
    FROZEN_SPLIT_MANIFEST,
)
from .data import audit_scope_report, load_audit_data
from .distributions import feature_distributions, parameter_distributions
from .feature_analysis import (
    feature_separability_analysis,
    pairwise_overlap_analysis,
    save_feature_separability_csv,
)
from .plots import generate_plots
from .signal_analysis import (
    dataset_diversity_audit,
    dsp_preprocessing_audit,
    generation_parameter_analysis,
    realism_scorecard,
    signal_level_audit,
    synthetic_shortcut_audit,
)

HANDOFF_REPORT = Path(__file__).resolve().parents[3] / "docs" / "AI_ML_DSP_REALISM_AUDIT.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def _determine_primary_bottleneck(
    overlap: dict[str, Any],
    confusion: dict[str, Any],
    dsp: dict[str, Any],
    diversity: dict[str, Any],
    shortcuts: dict[str, Any],
) -> dict[str, Any]:
    overlapping_pairs = sum(
        1 for p in overlap["pairs"].values() if p.get("classes_overlapping_in_feature_space")
    )
    top_error = confusion["top_confusion_pairs"][0] if confusion["top_confusion_pairs"] else {}
    dsp_high = sum(1 for f in dsp["findings"] if f["severity"] == "HIGH")

    scores = {
        "DATA_GENERATION": 0,
        "FEATURE_REPRESENTATION": 0,
        "DSP_PREPROCESSING": 0,
        "CLASS_OVERLAP": 0,
    }
    scores["CLASS_OVERLAP"] += overlapping_pairs * 3
    scores["CLASS_OVERLAP"] += int(confusion.get("total_errors", 0) / 50)
    scores["FEATURE_REPRESENTATION"] += 2 if overlapping_pairs >= 3 else 0
    scores["DSP_PREPROCESSING"] += dsp_high * 2
    scores["DATA_GENERATION"] += sum(
        1 for c in diversity["per_class"].values() if c.get("snr_std", 99) < 3.0
    )
    scores["DATA_GENERATION"] += shortcuts.get("n_flagged", 0)

    primary_key = max(scores, key=scores.get)
    mapping = {
        "DATA_GENERATION": "A: DATA GENERATION IS THE PRIMARY BOTTLENECK",
        "FEATURE_REPRESENTATION": "B: FEATURE REPRESENTATION IS THE PRIMARY BOTTLENECK",
        "DSP_PREPROCESSING": "C: DSP PREPROCESSING IS THE PRIMARY BOTTLENECK",
        "CLASS_OVERLAP": "D: CLASS OVERLAP IS THE PRIMARY BOTTLENECK",
    }
    if max(scores.values()) < 3:
        primary = "E: NO SINGLE BOTTLENECK IDENTIFIED"
        confidence = "LOW"
    else:
        primary = mapping[primary_key]
        confidence = "HIGH" if scores[primary_key] >= scores[sorted(scores, key=scores.get, reverse=True)[1]] + 2 else "MEDIUM"

    secondary = []
    if primary_key != "CLASS_OVERLAP" and overlapping_pairs >= 2:
        secondary.append("Heavy 32-D class overlap drives top confusion pairs")
    if dsp_high:
        secondary.append("Single-chunk FFT + squelch limit modulation-sensitive discrimination")
    if shortcuts.get("n_flagged", 0):
        secondary.append("Metadata/feature shortcuts indicate narrow synthetic manifolds")

    return {
        "primary_conclusion": primary,
        "scores": scores,
        "secondary_causes": secondary or ["Boundary SNR/amplitude regimes concentrate errors"],
        "evidence": {
            "overlapping_pairs_in_feature_space": overlapping_pairs,
            "top_confusion_pair": top_error,
            "dsp_high_severity_findings": dsp_high,
        },
        "confidence": confidence,
        "recommended_next_experiment": (
            "Prototype multi-window / modulation-aware features on regenerated diagnostic audio "
            "only; evaluate on clean development split without touching locked test."
        ),
    }


def _write_markdown_report(audit: dict[str, Any]) -> None:
    conclusion = audit["conclusion"]
    scorecard = audit["realism_scorecard"]
    confusion = audit["confusion_boundary_analysis"]
    verified = confusion["verified_known_pairs"]

    lines = [
        "# AI/ML DSP & Feature-Space Realism Audit",
        "",
        f"Generated: {audit['generated_at_utc']}",
        "",
        "## Executive summary",
        "",
        f"Primary bottleneck: **{conclusion['primary_conclusion']}**",
        f"Confidence: {conclusion['confidence']}",
        "",
        "This audit investigated why the frozen `linear_svm_balanced` baseline achieves "
        "~79.48% locked-test accuracy. **No frozen artifacts were modified.** "
        "**Locked test was not accessed.**",
        "",
        "## Data used",
        "",
        f"- Dataset: `{audit['scope']['dataset']}`",
        f"- Partitions: model_train + cal_dev ({audit['scope']['n_development']} samples)",
        f"- Locked test accessed: **{audit['scope']['locked_test_accessed']}**",
        "",
        "## Frozen assets protected",
        "",
        "- phase3_clean/, phase4_clean/, phase5_clean/ — untouched",
        "- training_data_v2_clean/ — untouched",
        "- frozen_split_manifest.json — untouched",
        "",
        "## Error boundary analysis",
        "",
        "| Pair | Verified count | Historical | Match |",
        "|------|----------------|------------|-------|",
    ]
    for key, row in verified.items():
        lines.append(
            f"| {key.replace('_', ' → ')} | {row['count']} | {row['historical_count']} | "
            f"{'yes' if row['match_historical'] else 'no'} |"
        )

    lines.extend(["", "## Feature analysis", ""])
    sep = audit["feature_separability"]
    lines.append(f"Top global features (MI): {', '.join(sep['top_5_global'])}")
    lines.append(f"Top FSK vs Tone: {', '.join(sep['top_5_fsk_tone'])}")
    lines.append(f"Top OOK vs Benign: {', '.join(sep['top_5_ook_benign'])}")

    lines.extend(["", "## Pairwise class overlap", ""])
    for pair, stats in audit["pairwise_overlap"]["pairs"].items():
        lines.append(
            f"- **{pair}**: centroid_dist={stats['centroid_distance_standardized']:.2f}, "
            f"kNN purity={stats['knn_local_purity_mean']:.2f}, "
            f"overlapping={stats['classes_overlapping_in_feature_space']}"
        )

    lines.extend(["", "## Realism scorecard", "", "| Area | Status | Evidence |", "|------|--------|----------|"])
    for row in scorecard:
        lines.append(f"| {row['area']} | {row['status']} | {row['evidence']} |")

    lines.extend(["", "## Primary conclusion", "", f"**{conclusion['primary_conclusion']}**", ""])
    lines.append("Secondary causes:")
    for s in conclusion["secondary_causes"]:
        lines.append(f"- {s}")
    lines.extend([
        "",
        f"Recommended next experiment: {conclusion['recommended_next_experiment']}",
        "",
        "## What should NOT be changed",
        "",
        "- Phase-1 FeatureExtractor (32-feature contract)",
        "- Frozen split and locked test",
        "- Production model artifacts",
        "",
        "## Reports",
        "",
        "- `ai/ml/experiments/realism_validation/reports/realism_audit.json`",
        "- `ai/ml/experiments/realism_validation/plots/`",
    ])
    HANDOFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_audit(*, generate_signal_diagnostics: bool = True) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_audit_data()
    scope = audit_scope_report(data)

    confusion = confusion_boundary_analysis(data)
    feat_dist = feature_distributions(data)
    param_dist = parameter_distributions(data)
    separability = feature_separability_analysis(data)
    overlap = pairwise_overlap_analysis(data)
    diversity = dataset_diversity_audit(data)
    shortcuts = synthetic_shortcut_audit(data)
    dsp = dsp_preprocessing_audit(data)
    gen_params = generation_parameter_analysis(data, confusion)
    signal = signal_level_audit() if generate_signal_diagnostics else {"skipped": True}
    scorecard = realism_scorecard(diversity, overlap, shortcuts, dsp, separability)
    plots = generate_plots(data)
    conclusion = _determine_primary_bottleneck(overlap, confusion, dsp, diversity, shortcuts)

    _write_json(REPORTS_DIR / "feature_separability.json", separability)
    save_feature_separability_csv(
        separability["global_feature_ranking"],
        REPORTS_DIR / "feature_separability.csv",
    )
    _write_json(REPORTS_DIR / "confusion_boundary_analysis.json", confusion)
    _write_json(REPORTS_DIR / "dataset_diversity_report.json", diversity)
    _write_json(REPORTS_DIR / "generation_parameter_analysis.json", gen_params)
    _write_json(REPORTS_DIR / "feature_distributions.json", feat_dist)
    _write_json(REPORTS_DIR / "parameter_distributions.json", param_dist)
    _write_json(REPORTS_DIR / "synthetic_shortcut_audit.json", shortcuts)
    _write_json(REPORTS_DIR / "dsp_preprocessing_audit.json", dsp)
    _write_json(REPORTS_DIR / "signal_level_audit.json", signal)

    audit = {
        "audit_id": "dsp_feature_realism_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "frozen_baseline_reference_only": FROZEN_BASELINE,
        "frozen_artifacts_modified": False,
        "locked_test_accessed": False,
        "confusion_boundary_analysis": confusion,
        "feature_separability": separability,
        "pairwise_overlap": overlap,
        "dataset_diversity": diversity,
        "generation_parameter_analysis": gen_params,
        "synthetic_shortcut_audit": shortcuts,
        "dsp_preprocessing_audit": dsp,
        "signal_level_audit": signal,
        "realism_scorecard": scorecard,
        "plots": plots,
        "conclusion": conclusion,
        "reproducibility": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "git_commit": _git_hash(),
            "experiment_dir": str(EXPERIMENT_DIR),
            "frozen_split_manifest": str(FROZEN_SPLIT_MANIFEST),
        },
    }
    _write_json(REPORTS_DIR / "realism_audit.json", audit)
    _write_markdown_report(audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DSP/feature realism audit")
    parser.add_argument("--skip-signal-diagnostics", action="store_true")
    args = parser.parse_args()
    audit = run_audit(generate_signal_diagnostics=not args.skip_signal_diagnostics)
    print(json.dumps(
        {
            "primary_conclusion": audit["conclusion"]["primary_conclusion"],
            "confidence": audit["conclusion"]["confidence"],
            "locked_test_accessed": audit["locked_test_accessed"],
            "reports": str(REPORTS_DIR / "realism_audit.json"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
