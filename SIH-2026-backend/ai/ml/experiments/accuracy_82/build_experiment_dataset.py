"""Build the experimental diversity dataset via the real DSP pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_AI = Path(__file__).resolve().parents[3]
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from ml.dataset_v2 import generator as gen
from ml.dataset_v2.build_dataset import build_dataset
from ml.dataset_v2.validate_dataset import validate_dataset

from .experiment_diversity_config import (
    EXPERIMENT_DATASET_VERSION,
    EXPERIMENT_OUTPUT_DIR_NAME,
)


def build_experiment_dataset(
    *,
    output_dir: Path | None = None,
    samples_per_class: int = 500,
    seed: int = 42,
    save_audio: bool = False,
) -> dict:
    import ml.experiments.accuracy_82.experiment_diversity_config as exp_cfg

    gen.cfg = exp_cfg
    out = Path(output_dir) if output_dir is not None else _AI / "ml" / EXPERIMENT_OUTPUT_DIR_NAME
    info = build_dataset(
        output_dir=out,
        samples_per_class=samples_per_class,
        seed=seed,
        save_audio=save_audio,
        dataset_version=EXPERIMENT_DATASET_VERSION,
    )
    import ml.dataset_v2.validate_dataset as val_mod

    val_mod.cfg = exp_cfg
    checks = validate_dataset(out, strict_clean=True)
    failed = [c for c in checks if not c.passed]
    if failed:
        raise RuntimeError(
            "Experiment dataset validation failed: "
            + "; ".join(str(c) for c in failed)
        )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset_v2_experiment")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()
    info = build_experiment_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        save_audio=args.save_audio,
    )
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
