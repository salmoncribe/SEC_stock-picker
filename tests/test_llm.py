"""Offline tests for the ollama provider: every failure funnels into LLMError.

The collector guards extraction with a single ``except LLMError`` and moves on,
so the provider's contract is that nothing else escapes -- a down server, a bad
status, empty output, or non-JSON content must all surface as LLMError, never as
a raw httpx or json exception that would abort a batch of thousands.
"""

from __future__ import annotations

import json

import httpx
import pytest

from market_intelligence.llm.base import LLMError
from market_intelligence.llm.ollama import OllamaProvider

SCHEMA = {"type": "object", "properties": {"edges": {"type": "array"}}, "required": ["edges"]}


def _provider(handler) -> OllamaProvider:
    return OllamaProvider(model="test-model", transport=httpx.MockTransport(handler))


def test_returns_parsed_object_on_success():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["format"] == SCHEMA  # schema-constrained decoding, not json mode
        assert body["options"]["temperature"] == 0.0
        return httpx.Response(200, json={"message": {"content": '{"edges": [{"target": "MSFT"}]}'}})

    result = _provider(handle).complete_json(system="s", user="u", schema=SCHEMA)

    assert result == {"edges": [{"target": "MSFT"}]}


def test_non_2xx_becomes_llm_error():
    provider = _provider(lambda _r: httpx.Response(500, text="boom"))

    with pytest.raises(LLMError):
        provider.complete_json(system="s", user="u", schema=SCHEMA)


def test_transport_error_becomes_llm_error():
    def handle(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        _provider(handle).complete_json(system="s", user="u", schema=SCHEMA)


def test_empty_content_becomes_llm_error():
    provider = _provider(lambda _r: httpx.Response(200, json={"message": {"content": ""}}))

    with pytest.raises(LLMError):
        provider.complete_json(system="s", user="u", schema=SCHEMA)


def test_non_json_content_becomes_llm_error():
    provider = _provider(
        lambda _r: httpx.Response(200, json={"message": {"content": "not json at all"}})
    )

    with pytest.raises(LLMError):
        provider.complete_json(system="s", user="u", schema=SCHEMA)


def test_non_object_json_becomes_llm_error():
    """A bare array is valid JSON but not the object contract the caller needs."""
    provider = _provider(lambda _r: httpx.Response(200, json={"message": {"content": "[1, 2, 3]"}}))

    with pytest.raises(LLMError):
        provider.complete_json(system="s", user="u", schema=SCHEMA)
