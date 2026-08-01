"""Research-only public-news scraping for relationship-decision testing.

This module is intentionally a test/research rail.  It can collect exact bytes
from public news or issuer pages and ask the local Ollama model whether the
page appears relevant to a filing, but it does not promote the result to a
live first-public proof.  The scraped bytes and hashes are evidence; the model
summary is a hypothesis.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_intelligence import hashing
from market_intelligence.llm.base import LLMError
from market_intelligence.schemas.provenance import ObservedSource, SourceObservation

if TYPE_CHECKING:
    from market_intelligence.config import Config
    from market_intelligence.llm.base import LLMProvider

DEFAULT_ALLOWED_NEWS_DOMAINS = frozenset(
    {
        "accesswire.com",
        "businesswire.com",
        "globenewswire.com",
        "newsfilecorp.com",
        "prnewswire.com",
        "sec.gov",
    }
)
MAX_NEWS_URLS = 20
MAX_NEWS_BYTES = 2_000_000
MAX_TEXT_CHARS = 12_000

NEWS_RELEVANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "mentions_target": {"type": "boolean"},
        "appears_first_public": {"type": "boolean"},
        "public_at_utc": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "is_relevant",
        "mentions_target",
        "appears_first_public",
        "public_at_utc",
        "confidence",
        "summary",
        "reasons",
    ],
    "additionalProperties": False,
}


class ScrapedNewsDocument(BaseModel):
    """One public page fetched and reduced to deterministic research evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(min_length=1)
    final_url: str = Field(min_length=1)
    fetched_at: datetime
    raw_sha256: str = Field(min_length=64, max_length=64)
    text_sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    title: str
    text_excerpt: str

    @field_validator("fetched_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("fetched_at must be timezone-aware UTC")
        if offset.total_seconds() != 0:
            raise ValueError("fetched_at must be normalized to UTC")
        return value.astimezone(UTC)


class LocalNewsEvidence(BaseModel):
    """Local-LLM interpretation of a scraped page.

    The model can be useful for triage, but it is not a trusted timestamp
    authority.  Downstream code should treat this as research evidence unless a
    deterministic source receipt independently verifies timing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    model: str
    is_relevant: bool
    mentions_target: bool
    appears_first_public: bool
    public_at_utc: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    reasons: tuple[str, ...]


def _is_allowed_url(url: str, allowed_domains: frozenset[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _visible_text(html: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = " ".join(soup.get_text(" ", strip=True).split())
    return title, text[:MAX_TEXT_CHARS]


def fetch_public_news_documents(
    urls: Sequence[str],
    config: Config,
    *,
    allowed_domains: Iterable[str] = DEFAULT_ALLOWED_NEWS_DOMAINS,
    now: datetime | None = None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[ScrapedNewsDocument, ...]:
    """Fetch a bounded list of public pages and retain exact source hashes."""
    if len(urls) > MAX_NEWS_URLS:
        raise ValueError(f"at most {MAX_NEWS_URLS} news URLs can be scraped in one batch")
    allowed = frozenset(domain.lower().removeprefix("www.") for domain in allowed_domains)
    disallowed = [url for url in urls if not _is_allowed_url(url, allowed)]
    if disallowed:
        raise ValueError(f"news URL outside allowed domains: {disallowed[0]}")

    fetched_at = (now or datetime.now(UTC)).astimezone(UTC)
    timeout = httpx.Timeout(
        config.settings.http.timeout_seconds,
        connect=config.settings.http.connect_timeout_seconds,
    )
    user_agent = config.env.sec_user_agent or "market-intelligence-testing/1.0"
    headers = {"User-Agent": user_agent}
    documents: list[ScrapedNewsDocument] = []
    with httpx.Client(
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        transport=transport,
    ) as client:
        for url in dict.fromkeys(urls):
            response = client.get(url)
            response.raise_for_status()
            raw = response.content[:MAX_NEWS_BYTES]
            title, text = _visible_text(raw)
            documents.append(
                ScrapedNewsDocument(
                    url=url,
                    final_url=str(response.url),
                    fetched_at=fetched_at,
                    raw_sha256=hashing.sha256_bytes(raw),
                    text_sha256=hashing.sha256_text(text),
                    byte_size=len(raw),
                    title=title,
                    text_excerpt=text,
                )
            )
    return tuple(documents)


def classify_news_document(
    document: ScrapedNewsDocument,
    provider: LLMProvider,
    *,
    ticker: str,
    filing_summary: str,
) -> LocalNewsEvidence:
    """Ask local Ollama whether a scraped page is useful research evidence."""
    payload = (
        f"Ticker: {ticker}\n"
        f"Filing summary: {filing_summary}\n"
        f"Page URL: {document.final_url}\n"
        f"Page title: {document.title}\n"
        f"Fetched at UTC: {document.fetched_at.isoformat()}\n"
        f"Page text:\n{document.text_excerpt}"
    )
    result = provider.complete_json(
        system=(
            "You classify public news/issuer pages for a research-only SEC filing "
            "test harness. Use only the supplied page text. If a timestamp is not "
            "explicitly present, set public_at_utc to null. Do not infer that a "
            "page proves first-public timing; appears_first_public only means the "
            "page looks like a direct announcement or wire item."
        ),
        user=payload,
        schema=NEWS_RELEVANCE_SCHEMA,
        temperature=0.0,
    )
    return LocalNewsEvidence(
        url=document.final_url,
        raw_sha256=document.raw_sha256,
        model=provider.model,
        is_relevant=bool(result["is_relevant"]),
        mentions_target=bool(result["mentions_target"]),
        appears_first_public=bool(result["appears_first_public"]),
        public_at_utc=result["public_at_utc"],
        confidence=float(result["confidence"]),
        summary=str(result["summary"]),
        reasons=tuple(str(reason) for reason in result["reasons"]),
    )


def collect_local_news_evidence(
    urls: Sequence[str],
    config: Config,
    provider: LLMProvider,
    *,
    ticker: str,
    filing_summary: str,
    allowed_domains: Iterable[str] = DEFAULT_ALLOWED_NEWS_DOMAINS,
    now: datetime | None = None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[LocalNewsEvidence, ...]:
    """Scrape public pages and classify them with local Ollama.

    A model failure on one page should not kill the batch; the failed page is
    represented as low-confidence, non-relevant evidence with the source hash.
    """
    documents = fetch_public_news_documents(
        urls,
        config,
        allowed_domains=allowed_domains,
        now=now,
        transport=transport,
    )
    evidence: list[LocalNewsEvidence] = []
    for document in documents:
        try:
            evidence.append(
                classify_news_document(
                    document,
                    provider,
                    ticker=ticker,
                    filing_summary=filing_summary,
                )
            )
        except LLMError as exc:
            evidence.append(
                LocalNewsEvidence(
                    url=document.final_url,
                    raw_sha256=document.raw_sha256,
                    model=provider.model,
                    is_relevant=False,
                    mentions_target=False,
                    appears_first_public=False,
                    public_at_utc=None,
                    confidence=0.0,
                    summary="local LLM classification failed",
                    reasons=(str(exc),),
                )
            )
    return tuple(evidence)


def source_observation_from_scrape(document: ScrapedNewsDocument) -> SourceObservation:
    """Convert a scrape into a provenance observation without trusted timing."""
    return SourceObservation(
        source=ObservedSource.NEWSWIRE,
        url=document.final_url,
        observed_at=document.fetched_at,
        raw_sha256=document.raw_sha256,
        source_record_id=document.text_sha256,
    )


__all__ = [
    "DEFAULT_ALLOWED_NEWS_DOMAINS",
    "NEWS_RELEVANCE_SCHEMA",
    "LocalNewsEvidence",
    "ScrapedNewsDocument",
    "classify_news_document",
    "collect_local_news_evidence",
    "fetch_public_news_documents",
    "source_observation_from_scrape",
]
