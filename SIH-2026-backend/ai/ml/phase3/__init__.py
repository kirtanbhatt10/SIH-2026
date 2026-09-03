"""Phase 3 — ML baseline model comparison on Dataset V2.

Fresh implementation. Does not depend on legacy phase2/phase3 scripts.
"""

from .config import PHASE3_OUTPUT_DIR
from .data import load_dataset_v2, grouped_train_test_split
from .train import run_phase3

__all__ = [
    "PHASE3_OUTPUT_DIR",
    "load_dataset_v2",
    "grouped_train_test_split",
    "run_phase3",
]
