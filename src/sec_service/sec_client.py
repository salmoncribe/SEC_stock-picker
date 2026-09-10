"""SEC EDGAR HTTP client with rate limiting."""

from __future__ import annotations

import time
from typing import Any
import httpx

SEC_WWW_URL = "https://www.sec.gov"
SEC_DATA_URL = "https://data.sec.gov"


class SECClient:
    def __init__(self, user_agent: str, requests_per_second: float = 5.0) -> None:
        self.user_agent = user_agent
        self.delay_between_requests = 1.0 / max(requests_per_second, 0.1)
        self.last_request_time = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay_between_requests:
            time.sleep(self.delay_between_requests - elapsed)
        self.last_request_time = time.time()

    def fetch_company_tickers(self) -> dict[str, Any]:
        """Fetch www.sec.gov/files/company_tickers.json."""
        self._rate_limit()
        url = f"{SEC_WWW_URL}/files/company_tickers.json"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch data.sec.gov/submissions/CIK{cik10}.json."""
        self._rate_limit()
        cik10 = str(cik).zfill(10)
        url = f"{SEC_DATA_URL}/submissions/CIK{cik10}.json"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_document_bytes(self, url: str) -> bytes:
        """Download raw filing document bytes."""
        self._rate_limit()
        response = self.client.get(url)
        response.raise_for_status()
        return response.content

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> SECClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
