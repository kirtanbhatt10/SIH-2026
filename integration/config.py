"""Integration layer configuration (AI/DSP → Backend 1)."""

import os

# Base URL for Backend 1 REST API (no trailing slash).
# Example: BACKEND_API_URL=http://127.0.0.1:8000
BACKEND_API_URL: str = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")

# POST /api/analyze timeout in seconds.
BACKEND_ANALYZE_TIMEOUT_SEC: float = float(os.environ.get("BACKEND_ANALYZE_TIMEOUT_SEC", "10"))
