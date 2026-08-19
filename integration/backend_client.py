"""
HTTP client: AI/DSP ThreatEvent → Backend 1 POST /api/analyze.

The DSP pipeline produces the payload; this module only transports it.
No field renaming, no score recalculation, no FastAPI imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from integration.config import BACKEND_ANALYZE_TIMEOUT_SEC, BACKEND_API_URL

logger = logging.getLogger(__name__)

ANALYZE_PATH = "/api/analyze"
CURRENT_THREAT_PATH = "/api/threats/current"

PostFn = Callable[[str, dict], httpx.Response]


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of submitting a ThreatEvent to Backend 1."""

    success: bool
    status_code: Optional[int]
    response_body: Optional[dict]
    error: Optional[str]
    payload: dict

    @property
    def stored(self) -> bool:
        """True only when Backend 1 acknowledged receipt."""
        return self.success


class BackendThreatClient:
    """
    POST the exact dict from ``DSPPipeline.to_threat_event()`` to Backend 1.

    Usage::

        from dsp import DSPPipeline
        from integration.backend_client import BackendThreatClient

        pipeline = DSPPipeline()
        # ... pipeline.process(chunk) for each audio chunk ...
        client = BackendThreatClient()
        result = client.submit_threat_event(pipeline.to_threat_event())
        if not result.success:
            logger.error("Backend submission failed: %s", result.error)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        post_fn: PostFn | None = None,
    ):
        self.base_url = (base_url or BACKEND_API_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else BACKEND_ANALYZE_TIMEOUT_SEC
        self._post_fn = post_fn

    @property
    def analyze_url(self) -> str:
        return f"{self.base_url}{ANALYZE_PATH}"

    @property
    def current_threat_url(self) -> str:
        return f"{self.base_url}{CURRENT_THREAT_PATH}"

    def submit_threat_event(self, threat_event: dict) -> SubmitResult:
        """
        POST *threat_event* unchanged to ``/api/analyze``.

        Returns a :class:`SubmitResult`. Does not raise on backend errors —
        callers decide whether to retry or continue processing audio.
        """
        payload = threat_event
        try:
            response = self._post(self.analyze_url, payload)
        except httpx.TimeoutException as exc:
            msg = f"Backend request timed out after {self.timeout}s: {exc}"
            logger.warning(msg)
            return SubmitResult(False, None, None, msg, payload)
        except httpx.RequestError as exc:
            msg = f"Backend unavailable: {exc}"
            logger.warning(msg)
            return SubmitResult(False, None, None, msg, payload)

        return self._parse_analyze_response(response, payload)

    def get_current_threat(self) -> tuple[Optional[dict], Optional[str]]:
        """
        GET ``/api/threats/current``. Returns ``(current_event, error)``.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.current_threat_url)
        except httpx.TimeoutException as exc:
            return None, f"Backend request timed out: {exc}"
        except httpx.RequestError as exc:
            return None, f"Backend unavailable: {exc}"

        if response.status_code != 200:
            return None, f"Unexpected HTTP {response.status_code} from /api/threats/current"

        try:
            body = response.json()
        except ValueError as exc:
            return None, f"Invalid JSON from backend: {exc}"

        return body.get("current"), None

    def _post(self, url: str, payload: dict) -> httpx.Response:
        if self._post_fn is not None:
            return self._post_fn(url, payload)
        with httpx.Client(timeout=self.timeout) as client:
            return client.post(url, json=payload)

    def _parse_analyze_response(self, response: httpx.Response, payload: dict) -> SubmitResult:
        status = response.status_code
        body: Optional[dict] = None

        try:
            parsed: Any = response.json()
            if isinstance(parsed, dict):
                body = parsed
        except ValueError:
            body = None

        if 200 <= status < 300:
            if body and body.get("status") == "received":
                return SubmitResult(True, status, body, None, payload)
            msg = f"Backend returned HTTP {status} but response was not a valid analyze acknowledgement"
            logger.warning("%s: %r", msg, body)
            return SubmitResult(False, status, body, msg, payload)

        if status == 422:
            msg = f"Backend rejected ThreatEvent (HTTP 422 validation error)"
            logger.warning("%s: %r", msg, body)
            return SubmitResult(False, status, body, msg, payload)

        if 400 <= status < 500:
            msg = f"Backend client error HTTP {status}"
            logger.warning("%s: %r", msg, body)
            return SubmitResult(False, status, body, msg, payload)

        msg = f"Backend server error HTTP {status}"
        logger.error("%s: %r", msg, body)
        return SubmitResult(False, status, body, msg, payload)
