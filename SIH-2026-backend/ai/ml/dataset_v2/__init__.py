"""
Dataset V2 — synthetic training-dataset generator (Phase 2).

SOURCE = SYNTHETIC. Every sample in this dataset is produced by the
project's existing signal generator, pushed through the existing
AudioPreprocessor and the existing (frozen) 32-feature FeatureExtractor.
No real / over-the-air recordings are involved. See README.md.
"""

from .config_v2 import DATASET_VERSION, GENERATOR_VERSION, CLASS_NAMES

__all__ = ["DATASET_VERSION", "GENERATOR_VERSION", "CLASS_NAMES"]
