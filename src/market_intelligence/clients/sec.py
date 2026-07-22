"""SEC EDGAR HTTP client.

Wraps :class:`~market_intelligence.clients.base.BaseAPIClient` with SEC's two
hosts: the ``www.sec.gov`` static files (``company_tickers.json``, filing
archives) and the ``data.sec.gov`` JSON APIs (submissions, company facts).
Because the two live on different hosts, each method passes an absolute URL to
``request`` — that overrides the client's ``base_url`` while still getting the
shared pacing + retry behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from market_intelligence.clients.base import BaseAPIClient
from market_intelligence.config import Config, HttpConfig, RetryConfig, SecConfig
from market_intelligence.schemas.sec import build_filing_index_url, normalize_cik


@dataclass(frozen=True)
class FetchResult:
    """A single SEC fetch: the resolved URL, raw bytes, and parsed JSON."""

    url: str
    raw: bytes
    data: Any


@dataclass(frozen=True)
class DocumentFetchResult:
    """A fetched filing document: resolved URL, raw bytes, and content type.

    Deliberately does *not* parse the payload — filing documents are HTML/text
    that must be preserved byte-for-byte before anything interprets them.
    """

    url: str
    raw: bytes
    content_type: str | None


class SECClient(BaseAPIClient):
    """Fetches SEC EDGAR ticker maps, submissions, and company facts."""

    def __init__(
        self,
        *,
        user_agent: str,
        sec_config: SecConfig,
        http_config: HttpConfig | None = None,
        retry_config: RetryConfig | None = None,
        requests_per_second: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        super().__init__(
            base_url=sec_config.base_url,
            headers=headers,
            http_config=http_config,
            retry_config=retry_config,
            requests_per_second=requests_per_second,
            transport=transport,
        )
        self._sec = sec_config

    def __enter__(self) -> SECClient:
        return self

    @classmethod
    def from_config(
        cls, config: Config, *, transport: httpx.BaseTransport | None = None
    ) -> SECClient:
        """Construct from the platform :class:`Config` (gated on the User-Agent)."""
        user_agent = config.require_sec_user_agent()
        return cls(
            user_agent=user_agent,
            sec_config=config.settings.sec,
            http_config=config.settings.http,
            retry_config=config.settings.retry,
            requests_per_second=config.settings.pacing.sec.requests_per_second,
            transport=transport,
        )

    def fetch_company_tickers(self) -> FetchResult:
        """Fetch ``company_tickers.json`` (the ticker/CIK map) from www.sec.gov."""
        url = self._sec.base_url + self._sec.company_tickers_path
        response = self.request("GET", url)
        return FetchResult(str(response.request.url), response.content, response.json())

    def fetch_submissions(self, cik: str) -> FetchResult:
        """Fetch a company's ``submissions`` document from data.sec.gov."""
        cik10 = normalize_cik(cik)
        url = self._sec.data_base_url + self._sec.submissions_path_template.format(cik10=cik10)
        response = self.request("GET", url)
        return FetchResult(str(response.request.url), response.content, response.json())

    def fetch_company_facts(self, cik: str) -> FetchResult:
        """Fetch a company's XBRL company-facts document (baseline extension point)."""
        url = self._sec.data_base_url + self._sec.company_facts_path_template.format(cik10=cik)
        response = self.request("GET", url)
        return FetchResult(str(response.request.url), response.content, response.json())

    def fetch_filing_index(self, cik: str, accession_number: str) -> FetchResult:
        """Fetch a filing folder's ``index.json`` (the document manifest).

        Used to corroborate a downloaded document: the manifest declares each
        file's name and byte size, which is the strongest integrity signal SEC
        publishes for archive files.
        """
        url = build_filing_index_url(cik, accession_number)
        response = self.request("GET", url)
        return FetchResult(str(response.request.url), response.content, response.json())

    def fetch_filing_document(self, url: str) -> DocumentFetchResult:
        """Fetch a filing document, preserving the response bytes unparsed."""
        response = self.request("GET", url)
        return DocumentFetchResult(
            str(response.request.url),
            response.content,
            response.headers.get("Content-Type"),
        )


__all__ = ["DocumentFetchResult", "FetchResult", "SECClient"]
