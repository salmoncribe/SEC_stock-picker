from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from market_intelligence.collectors.public_news import (
    NEWS_RELEVANCE_SCHEMA,
    classify_news_document,
    collect_local_news_evidence,
    fetch_public_news_documents,
    source_observation_from_scrape,
)
from market_intelligence.llm.base import LLMError
from market_intelligence.schemas.provenance import ObservedSource

HTML = b"""
<html>
  <head><title>Acme announces supply deal</title><script>secret()</script></head>
  <body>
    <nav>navigation noise</nav>
    <main>
      <h1>Acme announces relationship with NVDA</h1>
      <time datetime="2024-05-07T14:00:00Z">May 7, 2024 14:00 UTC</time>
      <p>Acme Corp announced a strategic supply relationship with NVDA.</p>
    </main>
  </body>
</html>
"""


class FakeProvider:
    model = "local-test-model"

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "temperature": temperature}
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_fetch_public_news_documents_scrapes_and_hashes(tmp_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"]
        return httpx.Response(200, content=HTML, headers={"Content-Type": "text/html"})

    docs = fetch_public_news_documents(
        ["https://www.prnewswire.com/news-releases/acme.html"],
        tmp_config,
        now=datetime(2024, 5, 7, 14, 5, tzinfo=UTC),
        transport=httpx.MockTransport(handler),
    )

    assert len(docs) == 1
    assert docs[0].title == "Acme announces supply deal"
    assert "strategic supply relationship with NVDA" in docs[0].text_excerpt
    assert "secret()" not in docs[0].text_excerpt
    assert len(docs[0].raw_sha256) == 64
    assert len(docs[0].text_sha256) == 64


def test_fetch_rejects_urls_outside_allowed_domains(tmp_config) -> None:
    with pytest.raises(ValueError, match="outside allowed domains"):
        fetch_public_news_documents(["https://example.com/release"], tmp_config)


def test_classify_news_document_uses_schema_constrained_local_provider(tmp_config) -> None:
    doc = fetch_public_news_documents(
        ["https://www.globenewswire.com/newsroom/release"],
        tmp_config,
        now=datetime(2024, 5, 7, 14, 5, tzinfo=UTC),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=HTML)),
    )[0]
    provider = FakeProvider(
        {
            "is_relevant": True,
            "mentions_target": True,
            "appears_first_public": True,
            "public_at_utc": "2024-05-07T14:00:00Z",
            "confidence": 0.82,
            "summary": "Direct announcement mentioning NVDA.",
            "reasons": ["explicit timestamp", "target mentioned"],
        }
    )

    evidence = classify_news_document(
        doc,
        provider,
        ticker="NVDA",
        filing_summary="Relationship disclosure involving Acme and NVDA.",
    )

    assert provider.calls[0]["schema"] == NEWS_RELEVANCE_SCHEMA
    assert provider.calls[0]["temperature"] == 0.0
    assert "Use only the supplied page text" in provider.calls[0]["system"]
    assert evidence.is_relevant is True
    assert evidence.model == "local-test-model"
    assert evidence.public_at_utc == "2024-05-07T14:00:00Z"


def test_collect_local_news_evidence_degrades_when_ollama_fails(tmp_config) -> None:
    evidence = collect_local_news_evidence(
        ["https://www.businesswire.com/news/home/test"],
        tmp_config,
        FakeProvider(LLMError("ollama down")),
        ticker="NVDA",
        filing_summary="Relationship disclosure involving Acme and NVDA.",
        now=datetime(2024, 5, 7, 14, 5, tzinfo=UTC),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=HTML)),
    )

    assert len(evidence) == 1
    assert evidence[0].is_relevant is False
    assert evidence[0].confidence == 0.0
    assert evidence[0].reasons == ("ollama down",)


def test_source_observation_from_scrape_preserves_hash_without_trusted_timestamp(
    tmp_config,
) -> None:
    doc = fetch_public_news_documents(
        ["https://www.accesswire.com/releases/test"],
        tmp_config,
        now=datetime(2024, 5, 7, 14, 5, tzinfo=UTC),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=HTML)),
    )[0]

    observation = source_observation_from_scrape(doc)

    assert observation.source == ObservedSource.NEWSWIRE
    assert observation.raw_sha256 == doc.raw_sha256
    assert observation.asserted_public_at is None
    assert observation.source_record_id == doc.text_sha256
