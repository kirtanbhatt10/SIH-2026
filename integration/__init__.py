"""AI/DSP → Backend 1 HTTP integration layer."""

from integration.backend_client import BackendThreatClient, SubmitResult
from integration.config import BACKEND_ANALYZE_TIMEOUT_SEC, BACKEND_API_URL

__all__ = [
    "BACKEND_API_URL",
    "BACKEND_ANALYZE_TIMEOUT_SEC",
    "BackendThreatClient",
    "SubmitResult",
]
