"""Temporary mic/pipeline debug helpers (D1.2 diagnostics)."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def chunk_stats(samples: np.ndarray) -> dict[str, float]:
    if samples.size == 0:
        return {"count": 0, "rms": 0.0, "peak": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(samples.size),
        "rms": float(np.sqrt(np.mean(samples * samples))),
        "peak": float(np.max(np.abs(samples))),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


class RateLimitedLogger:
    """Log first `burst` events, then at most once per `interval_sec`."""

    def __init__(self, label: str, burst: int = 3, interval_sec: float = 2.0) -> None:
        self.label = label
        self.burst = burst
        self.interval_sec = interval_sec
        self._count = 0
        self._last_log = 0.0

    def should_log(self) -> bool:
        self._count += 1
        if self._count <= self.burst:
            return True
        now = time.time()
        if now - self._last_log >= self.interval_sec:
            self._last_log = now
            return True
        return False

    def log(self, message: str, *args) -> None:
        if self.should_log():
            logger.info("[%s] %s", self.label, message % args if args else message)
