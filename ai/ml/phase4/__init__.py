"""Phase 4 — risk calibration on top of the frozen Phase 3 classifier."""

from .calibrator import RiskCalibrator, RiskPrediction
from .config import PHASE4_OUTPUT_DIR

__all__ = [
    "PHASE4_OUTPUT_DIR",
    "RiskCalibrator",
    "RiskPrediction",
    "run_phase4",
]


def run_phase4(*args, **kwargs):
    from .train import run_phase4 as _run_phase4
    return _run_phase4(*args, **kwargs)
