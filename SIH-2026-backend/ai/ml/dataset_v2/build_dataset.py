"""
Build Synthetic Dataset V2.

    python build_dataset.py
    python build_dataset.py --samples-per-class 500 --seed 42
    python build_dataset.py --output-dir ../../training_data_v2 --no-audio

Writes:
    <output_dir>/
        audio/<class>/<sample_id>.wav      (unless --no-audio)
        features.npy        (N, 32) float64
        labels.npy           (N,)   int32
        metadata.csv          one row per sample (Task 10 schema)
        dataset_info.json     dataset-level summary + provenance
        label_map.json         {"0": "benign", ...}
        README.md              local copy of the dataset-card text

SOURCE = SYNTHETIC. See dataset_v2/README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AI_DIR = _HERE.parent.parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

try:
    from . import config_v2 as cfg
    from .generator import generate_sample, make_shared_dsp_components, CLASS_NAMES
except ImportError:  # pragma: no cover - direct script execution
    import config_v2 as cfg  # type: ignore
    from generator import generate_sample, make_shared_dsp_components, CLASS_NAMES  # type: ignore

from dsp import FeatureExtractor  # noqa: E402

METADATA_FIELDS = [
    "sample_id", "label", "signal_type", "sample_rate", "duration",
    "frequency", "snr", "noise_level", "amplitude", "peak_amplitude", "noise_type",
    "seed", "local_seed", "global_seed", "generator_version", "dataset_version",
    "bit_rate", "frequency_deviation", "start_frequency", "end_frequency",
    "duty_cycle", "chirp_sweeps", "benign_variant",
    "source_duration_sec", "crop_offset_sec", "crop_attempts", "generation_group",
]

DATASET_README = """\
# Synthetic Dataset V2

**SOURCE = SYNTHETIC.**

This dataset is generated synthetically through the project's
signal-generation and DSP pipeline (`dsp.SignalGenerator` ->
`dsp.AudioPreprocessor` -> `dsp.FeatureExtractor`). It is intended for
reproducible ML development and does not establish real-world / OTA
detection performance.

It is not a real recording, not an over-the-air capture, and not a
microphone recording.

Regenerate with:

    cd ai/ml/dataset_v2
    python build_dataset.py --samples-per-class 500 --seed 42

See `ai/ml/dataset_v2/README.md` (committed to git) for the full dataset
card: parameter ranges, metadata schema, and known limitations. This copy
exists for local convenience only -- this whole directory is gitignored,
same as `ai/training_data/`.
"""


def build_dataset(
    output_dir: Path,
    samples_per_class: int = cfg.DEFAULT_SAMPLES_PER_CLASS,
    seed: int = 42,
    save_audio: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor, extractor = make_shared_dsp_components()
    expected_names = extractor.get_feature_names()
    assert expected_names == FeatureExtractor.FEATURE_NAMES, (
        "FeatureExtractor.get_feature_names() disagrees with its own "
        "FEATURE_NAMES class attribute -- STOP, this is a Phase 1 bug, "
        "not something Dataset V2 should silently work around."
    )
    assert len(expected_names) == 32, "Phase 1 feature contract must stay at 32 features"

    all_features = []
    all_labels = []
    all_metadata = []

    t0 = time.time()
    for class_idx, class_name in sorted(CLASS_NAMES.items()):
        if save_audio:
            (audio_dir / class_name).mkdir(parents=True, exist_ok=True)
        print(f"  Generating {samples_per_class} samples for class '{class_name}'...")
        for i in range(samples_per_class):
            sample = generate_sample(class_idx, i, seed, preprocessor, extractor)
            all_features.append(sample.features)
            all_labels.append(class_idx)
            all_metadata.append(sample.metadata)

            if save_audio:
                from scipy.io import wavfile
                wav_path = audio_dir / class_name / f"{sample.metadata['sample_id']}.wav"
                peak = float(np.max(np.abs(sample.audio))) or 1.0
                int16 = np.int16(np.clip(sample.audio / peak, -1.0, 1.0) * 32767)
                wavfile.write(str(wav_path), sample.metadata["sample_rate"], int16)

    elapsed = time.time() - t0

    X = np.asarray(all_features, dtype=np.float64)
    y = np.asarray(all_labels, dtype=np.int32)

    # Shuffle once, deterministically, using a seed independent of the
    # per-sample seeds above (so shuffling never perturbs any sample's
    # own reproducibility).
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xDA7A5E7]))
    perm = shuffle_rng.permutation(len(X))
    X, y = X[perm], y[perm]
    all_metadata = [all_metadata[i] for i in perm]

    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", y)

    with open(output_dir / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)

    with open(output_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in all_metadata:
            writer.writerow(row)

    duplicate_ids = len(all_metadata) - len({m["sample_id"] for m in all_metadata})

    dataset_info = {
        "source": "synthetic",
        "source_note": (
            "This dataset is generated synthetically through the project's "
            "signal-generation and DSP pipeline. It is intended for "
            "reproducible ML development and does not establish "
            "real-world/OTA detection performance."
        ),
        "dataset_version": cfg.DATASET_VERSION,
        "generator_version": cfg.GENERATOR_VERSION,
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": len(CLASS_NAMES),
        "samples_per_class": samples_per_class,
        "label_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "feature_names": expected_names,
        "sample_rate": 48000,
        "chunk_size": 2048,
        "chunk_duration_sec": 2048 / 48000,
        "duplicate_sample_ids": duplicate_ids,
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "generation_seconds": round(elapsed, 2),
        "audio_saved": save_audio,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    with open(output_dir / "README.md", "w") as f:
        f.write(DATASET_README)

    print(f"\n  Dataset V2 written to '{output_dir}/'")
    print(f"    features.npy : {X.shape}")
    print(f"    labels.npy   : {y.shape}")
    print(f"    classes      : {CLASS_NAMES}")
    print(f"    duplicate sample_ids : {duplicate_ids}")
    print(f"    NaN / Inf            : {dataset_info['nan_count']} / {dataset_info['inf_count']}")
    print(f"    generation time       : {elapsed:.1f}s")

    return dataset_info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(_AI_DIR / "training_data_v2"))
    ap.add_argument("--samples-per-class", type=int, default=cfg.DEFAULT_SAMPLES_PER_CLASS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-audio", action="store_true", help="skip writing .wav files")
    args = ap.parse_args()

    build_dataset(
        output_dir=Path(args.output_dir),
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        save_audio=not args.no_audio,
    )
