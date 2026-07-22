"""Shared HTTP client foundation for all API clients.

Provides, in one place:

* a synchronous ``httpx`` client with connect/read timeouts;
* polite per-source request pacing (a hard floor between requests);
* ``tenacity`` exponential backoff with jitter on transport errors, HTTP 429,
  and 5xx responses;
* honouring ``Retry-After`` on 429 responses.

Subclasses (SEC, FRED) configure ``base_url``, headers, default params, and
pacing, then call ``get_json`` / ``get_bytes``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from market_intelligence.config import HttpConfig, RetryConfig
from market_intelligence.logging_config import get_logger

_log = get_logger("clients.base")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RetryableHTTPError(Exception):
    """A response whose status warrants a retry (429 / 5xx)."""

    def __init__(self, status_code: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) to seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class RateLimiter:
    """Thread-safe minimum-interval limiter (simple, monotonic-clock based)."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


class BaseAPIClient:
    """Base synchronous API client with pacing + retries."""

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        default_params: dict[str, Any] | None = None,
        http_config: HttpConfig | None = None,
        retry_config: RetryConfig | None = None,
        requests_per_second: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        logger: Any = None,
    ) -> None:
        http_config = http_config or HttpConfig()
        self.retry_config = retry_config or RetryConfig()
        self._default_params = dict(default_params or {})
        self._limiter = RateLimiter(requests_per_second)
        self._log = logger or _log
        timeout = httpx.Timeout(
            http_config.timeout_seconds,
            connect=http_config.connect_timeout_seconds,
        )
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BaseAPIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- retry plumbing ----------------------------------------------------- #
    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return isinstance(exc, (httpx.TransportError, httpx.TimeoutException, RetryableHTTPError))

    def _wait_strategy(self) -> Callable[[RetryCallState], float]:
        base = wait_exponential_jitter(
            initial=self.retry_config.initial_backoff_seconds,
            max=self.retry_config.max_backoff_seconds,
            exp_base=self.retry_config.backoff_multiplier,
            jitter=self.retry_config.jitter_seconds,
        )

        def _wait(retry_state: RetryCallState) -> float:
            backoff = base(retry_state)
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(exc, RetryableHTTPError) and exc.retry_after is not None:
                return max(float(exc.retry_after), backoff)
            return backoff

        return _wait

    # -- requests ----------------------------------------------------------- #
    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        params = {**self._default_params, **(kwargs.pop("params", None) or {})}

        def _do() -> httpx.Response:
            self._limiter.wait()
            response = self._client.request(method, url, params=params, **kwargs)
            if response.status_code in RETRYABLE_STATUS:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                self._log.warning(
                    "http_retryable_status",
                    method=method,
                    url=str(response.request.url),
                    status=response.status_code,
                    retry_after=retry_after,
                )
                raise RetryableHTTPError(
                    response.status_code,
                    f"HTTP {response.status_code} for {response.request.url}",
                    retry_after,
                )
            response.raise_for_status()
            return response

        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(self.retry_config.max_attempts),
            wait=self._wait_strategy(),
            retry=retry_if_exception(self._is_retryable),
        )
        return retryer(_do)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs).json()

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        return self.request("GET", url, **kwargs).content
