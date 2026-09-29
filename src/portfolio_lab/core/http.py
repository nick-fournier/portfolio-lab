"""HTTP client with client-side rate limiting and retries.

Each data source gets its own :class:`RateLimitedClient` so its provider's limits are
respected independently. Transient failures (HTTP 429, 5xx, timeouts, connection errors)
are retried with exponential backoff and jitter, honoring ``Retry-After`` when present.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RateLimitedClient:
    """A thin wrapper over ``httpx.Client`` that spaces requests and retries failures.

    Args:
        base_url: Prefix for relative request URLs.
        headers: Headers sent with every request (e.g. auth or User-Agent).
        max_per_minute: Upper bound on request rate; ``None`` disables limiting.
        max_retries: Retries after the first attempt for transient failures.
        timeout: Per-request timeout in seconds.
        transport: Optional ``httpx`` transport, used by tests to mock responses.
        sleep: Sleep function, injectable so tests don't actually wait.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        max_per_minute: int | None = None,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._interval = 60.0 / max_per_minute if max_per_minute else 0.0
        self._max_retries = max_retries
        self._sleep = sleep
        self._last_request = 0.0

    def __enter__(self) -> "RateLimitedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def _throttle(self) -> None:
        """Wait until at least one rate-limit interval has passed since the last request."""
        wait = self._last_request + self._interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float:
        """Return seconds to wait before retry ``attempt`` (1-based)."""
        if response is not None and (retry_after := response.headers.get("Retry-After")):
            try:
                return float(retry_after)
            except ValueError:
                pass
        return min(60.0, 2.0**attempt) + random.uniform(0, 1)

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET ``url`` with rate limiting and retries.

        Args:
            url: Absolute URL, or a path relative to ``base_url``.
            params: Query parameters.

        Returns:
            The successful response.

        Raises:
            httpx.HTTPStatusError: For non-retryable errors, or when retries are exhausted.
            httpx.TransportError: When the connection keeps failing.
        """
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                delay = self._backoff(attempt, response)
                reason = response.status_code if response is not None else "transport error"
                log.warning("GET %s failed (%s); retry %d in %.1fs", url, reason, attempt, delay)
                self._sleep(delay)
            self._throttle()
            try:
                response = self._client.get(url, params=params)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                response = None
                continue
            if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                response.raise_for_status()
                return response
        raise AssertionError("unreachable")  # pragma: no cover

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``url`` and decode the JSON body."""
        return self.get(url, params).json()

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """GET ``url`` and return the body as text."""
        return self.get(url, params).text
