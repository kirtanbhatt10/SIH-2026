"""
Dataset V2 quality validator.

    python validate_dataset.py --dataset-dir ../../training_data_v2

Checks (Phase 2, Task 13):
    1.  expected classes exist
    2.  class counts
    3.  no missing labels
    4.  no duplicate sample_ids
    5.  no NaN features
    6.  no Inf features
    7.  exactly 32 features/sample
    8.  feature names/order match Phase 1
    9.  metadata rows match generated samples
    10. sample rates valid
    11. durations valid
    12. signals not silent (rms_amplitude feature above floor)
    13. frequency parameters within intended ranges
    13b. frequencies within Phase-1 detection band (18–21 kHz)
    14. SNR/noise metadata consistent
    15. reproducibility -- delegated to test_dataset_v2_reproducibility.py,
        this script only confirms the dataset declares seed/version.

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AI_DIR = _HERE.parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

try:
    from . import config_v2 as cfg
except ImportError:  # pragma: no cover
    import config_v2 as cfg  # type: ignore

from dsp.feature_extraction import FeatureExtractor  # noqa: E402


class Check:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name}" + (f" -- {self.detail}" if self.detail else "")


def _to_float(v):
    if v in (None, "", "None"):
        return None
    return float(v)


def validate_dataset(dataset_dir: Path) -> list[Check]:
    dataset_dir = Path(dataset_dir)
    checks: list[Check] = []

    X = np.load(dataset_dir / "features.npy")
    y = np.load(dataset_dir / "labels.npy")
    with open(dataset_dir / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(dataset_dir / "dataset_info.json") as f:
        info = json.load(f)

    # 1. expected classes exist
    expected_classes = set(cfg.CLASS_NAMES.keys())
    found_classes = set(int(v) for v in np.unique(y))
    checks.append(Check(
        "expected classes exist",
        found_classes == expected_classes,
        f"found={sorted(found_classes)} expected={sorted(expected_classes)}",
    ))

    # 2. class counts / balance
    counts = {cfg.CLASS_NAMES[int(k)]: int((y == k).sum()) for k in sorted(expected_classes)}
    total = len(y)
    pct = {k: round(100 * v / total, 2) if total else 0.0 for k, v in counts.items()}
    max_pct = max(pct.values()) if pct else 0
    min_pct = min(pct.values()) if pct else 0
    balanced = (max_pct - min_pct) <= 5.0   # within 5 percentage points
    checks.append(Check(
        "class balance (<=5pp spread)",
        balanced,
        f"counts={counts} pct={pct}",
    ))

    # 3. no missing labels
    no_missing_labels = len(y) == len(X) and not np.any(y < 0) and not np.isnan(y.astype(float)).any()
    checks.append(Check("no missing labels", bool(no_missing_labels), f"n={len(y)}"))

    # 4. no duplicate sample_ids
    ids = [r["sample_id"] for r in rows]
    dup = len(ids) - len(set(ids))
    checks.append(Check("no duplicate sample_ids", dup == 0, f"duplicates={dup}"))

    # 5 / 6. NaN / Inf
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    checks.append(Check("no NaN features", nan_count == 0, f"nan_count={nan_count}"))
    checks.append(Check("no Inf features", inf_count == 0, f"inf_count={inf_count}"))

    # 7. exactly 32 features/sample
    checks.append(Check("exactly 32 features/sample", X.shape[1] == 32, f"shape={X.shape}"))

    # 8. feature names/order match Phase 1
    names_match = info.get("feature_names") == FeatureExtractor.FEATURE_NAMES
    checks.append(Check("feature names/order match Phase 1 FeatureExtractor", names_match))

    # 9. metadata rows match generated samples
    rows_match = len(rows) == X.shape[0] == len(y)
    checks.append(Check(
        "metadata rows match generated samples",
        rows_match,
        f"metadata_rows={len(rows)} X={X.shape[0]} y={len(y)}",
    ))

    # 10. sample rates valid
    rates = {int(r["sample_rate"]) for r in rows}
    checks.append(Check("sample rates valid (single, ==48000)", rates == {48000}, f"rates={rates}"))

    # 11. durations valid
    durations = {round(float(r["duration"]), 6) for r in rows}
    expected_duration = round(2048 / 48000, 6)
    checks.append(Check(
        "durations valid (== fixed chunk duration)",
        durations == {expected_duration},
        f"durations={durations} expected={expected_duration}",
    ))

    # 12. signals not silent (proxy: rms_amplitude feature, if present)
    #     Benign samples CAN legitimately be silent (noise fully squelched by
    #     AudioPreprocessor's bandpass -- that IS what "no ultrasonic content"
    #     looks like). Communication classes should never be all-zero.
    if "rms_amplitude" in info.get("feature_names", []):
        idx = info["feature_names"].index("rms_amplitude")
        rms = X[:, idx]
        comm_mask = y != 0  # non-benign
        n_silent_comm = int((rms[comm_mask] < cfg.MIN_RMS_AMPLITUDE).sum())
        n_silent_benign = int((rms[~comm_mask] < cfg.MIN_RMS_AMPLITUDE).sum())
        checks.append(Check(
            "no silent COMMUNICATION samples (rms_amplitude)",
            n_silent_comm == 0,
            f"silent_comm={n_silent_comm} silent_benign={n_silent_benign}(ok) "
            f"threshold={cfg.MIN_RMS_AMPLITUDE}",
        ))
    else:
        checks.append(Check("no suspiciously silent samples", False, "rms_amplitude feature not found"))

    # 13. frequency parameters within intended ranges
    range_by_class = {
        "tone": cfg.TONE_FREQ_RANGE,
        "fsk": cfg.FSK_MARK_RANGE,
        "ook": cfg.OOK_CARRIER_RANGE,
    }
    freq_ok = True
    freq_detail = []
    for cls, (lo, hi) in range_by_class.items():
        vals = [_to_float(r["frequency"]) for r in rows if r["signal_type"] == cls]
        vals = [v for v in vals if v is not None]
        bad = [v for v in vals if not (lo - 1e-6 <= v <= hi + 1e-6)]
        if bad:
            freq_ok = False
            freq_detail.append(f"{cls}: {len(bad)} out of range")
    # chirp: check start/end ranges separately
    chirp_starts = [_to_float(r["start_frequency"]) for r in rows if r["signal_type"] == "chirp"]
    chirp_ends = [_to_float(r["end_frequency"]) for r in rows if r["signal_type"] == "chirp"]
    lo, hi = cfg.CHIRP_START_RANGE
    bad_start = [v for v in chirp_starts if v is not None and not (lo - 1e-6 <= v <= hi + 1e-6)]
    lo, hi = cfg.CHIRP_END_RANGE
    bad_end = [v for v in chirp_ends if v is not None and not (lo - 1e-6 <= v <= hi + 1e-6)]
    if bad_start or bad_end:
        freq_ok = False
        freq_detail.append(f"chirp: {len(bad_start)} start / {len(bad_end)} end out of range")
    checks.append(Check("frequency parameters within intended ranges", freq_ok, "; ".join(freq_detail)))

    # 13b. signal frequencies within Phase-1 detection band (18–21 kHz)
    band_lo, band_hi = cfg.DETECTION_BAND_LOW, cfg.DETECTION_BAND_HIGH
    band_ok = True
    band_detail = []
    for cls in ("tone", "fsk", "ook"):
        vals = [_to_float(r["frequency"]) for r in rows if r["signal_type"] == cls]
        bad = [v for v in vals if v is not None and not (band_lo - 1e-6 <= v <= band_hi + 1e-6)]
        if bad:
            band_ok = False
            band_detail.append(f"{cls}: {len(bad)} outside {band_lo}-{band_hi} Hz")
    chirp_rows = [r for r in rows if r["signal_type"] == "chirp"]
    bad_chirp = 0
    for r in chirp_rows:
        start = _to_float(r["start_frequency"])
        end = _to_float(r["end_frequency"])
        if start is not None and not (band_lo - 1e-6 <= start <= band_hi + 1e-6):
            bad_chirp += 1
        if end is not None and not (band_lo - 1e-6 <= end <= band_hi + 1e-6):
            bad_chirp += 1
    fsk_rows = [r for r in rows if r["signal_type"] == "fsk"]
    for r in fsk_rows:
        mark = _to_float(r["frequency"])
        dev = _to_float(r["frequency_deviation"])
        if mark is not None and dev is not None:
            space = mark + dev
            if space > band_hi + 1e-6:
                band_ok = False
                band_detail.append(f"fsk: space freq {space:.0f} > {band_hi}")
    if bad_chirp:
        band_ok = False
        band_detail.append(f"chirp: {bad_chirp} start/end outside {band_lo}-{band_hi} Hz")
    checks.append(Check(
        "frequencies within Phase-1 detection band",
        band_ok,
        "; ".join(band_detail) or f"band={band_lo}-{band_hi} Hz",
    ))

    # 14. SNR/noise metadata consistent
    snr_ok = True
    snr_detail = []
    valid_noise_types = {"white", "ambient", "none", ""}
    bad_noise_type = [r["noise_type"] for r in rows if r["noise_type"] not in valid_noise_types]
    if bad_noise_type:
        snr_ok = False
        snr_detail.append(f"{len(bad_noise_type)} unexpected noise_type values")
    threat_rows = [r for r in rows if r["signal_type"] != "benign"]
    bad_snr = 0
    for r in threat_rows:
        snr = _to_float(r["snr"])
        if snr is None:
            bad_snr += 1
            continue
        if not (-5.0 - 1e-6 <= snr <= 30.0 + 1e-6):
            bad_snr += 1
    if bad_snr:
        snr_ok = False
        snr_detail.append(f"{bad_snr} threat-class rows with missing/out-of-range snr")
    checks.append(Check("SNR/noise metadata consistent", snr_ok, "; ".join(snr_detail)))

    # 15. reproducibility -- declared provenance only (full check is a
    #     separate regenerate-twice test, see test_dataset_v2_reproducibility.py)
    has_provenance = bool(info.get("seed") is not None and info.get("dataset_version") and info.get("generator_version"))
    checks.append(Check(
        "reproducibility metadata present (seed/dataset_version/generator_version)",
        has_provenance,
        f"seed={info.get('seed')} dataset_version={info.get('dataset_version')} "
        f"generator_version={info.get('generator_version')}",
    ))

    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(_AI_DIR / "training_data_v2"))
    args = ap.parse_args()

    checks = validate_dataset(Path(args.dataset_dir))

    print("=" * 60)
    print("  Dataset V2 Validation Report")
    print("=" * 60)
    for c in checks:
        print(c)

    n_pass = sum(c.passed for c in checks)
    n_total = len(checks)
    print("-" * 60)
    print(f"  {n_pass}/{n_total} checks passed")

    if n_pass != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
