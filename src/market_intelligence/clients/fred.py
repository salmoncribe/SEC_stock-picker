"""FRED API client.

Thin wrapper over ``BaseAPIClient`` that supplies FRED's required ``api_key``
and ``file_type`` query parameters on every request (the documented FRED auth
mechanism — the application's own key, not user PII) and exposes the two
endpoints the collector needs: series metadata and series observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from market_intelligence.clients.base import BaseAPIClient
from market_intelligence.config import Config, FredConfig, HttpConfig, RetryConfig


@dataclass(frozen=True)
class FetchResult:
    """A single FRED fetch: the resolved URL, raw bytes, and parsed JSON."""

    url: str
    raw: bytes
    data: Any


class FREDClient(BaseAPIClient):
    """Synchronous FRED client with pacing + retries from the base client."""

    def __init__(
        self,
        *,
        api_key: str,
        fred_config: FredConfig,
        http_config: HttpConfig | None = None,
        retry_config: RetryConfig | None = None,
        requests_per_second: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        default_params = {"api_key": api_key, "file_type": fred_config.file_type}
        super().__init__(
            base_url=fred_config.base_url,
            default_params=default_params,
            http_config=http_config,
            retry_config=retry_config,
            requests_per_second=requests_per_second,
            transport=transport,
        )
        self._fred = fred_config

    @classmethod
    def from_config(
        cls, config: Config, *, transport: httpx.BaseTransport | None = None
    ) -> FREDClient:
        """Build a client from resolved config (raises if the key is missing)."""
        api_key = config.require_fred_api_key()
        return cls(
            api_key=api_key,
            fred_config=config.settings.fred,
            http_config=config.settings.http,
            retry_config=config.settings.retry,
            requests_per_second=config.settings.pacing.fred.requests_per_second,
            transport=transport,
        )

    def fetch_series(self, series_id: str) -> FetchResult:
        """Fetch metadata for a single FRED series."""
        response = self.request("GET", self._fred.series_path, params={"series_id": series_id})
        return FetchResult(str(response.request.url), response.content, response.json())

    def fetch_observations(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FetchResult:
        """Fetch observations for a series, optionally windowed by date."""
        params: dict[str, Any] = {"series_id": series_id}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        response = self.request("GET", self._fred.series_observations_path, params=params)
        return FetchResult(str(response.request.url), response.content, response.json())


__all__ = ["FREDClient", "FetchResult"]
