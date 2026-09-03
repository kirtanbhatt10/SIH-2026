"""Phase 5 — frozen ML inference/service layer."""

from .config import PHASE5_OUTPUT_DIR
from .service import InferenceService

__all__ = [
    "PHASE5_OUTPUT_DIR",
    "InferenceService",
    "run_phase5",
]


def run_phase5(*args, **kwargs):
    from .train import run_phase5 as _run_phase5
    return _run_phase5(*args, **kwargs)
