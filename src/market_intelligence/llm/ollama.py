"""Ollama backend for the LLM provider interface.

Talks to a local ``ollama`` server over HTTP with ``httpx`` (already a
dependency -- no ``ollama`` python package needed). The one detail that matters
for reliability: extraction uses ollama's **schema-constrained** decoding
(``format`` set to a JSON Schema), not its looser "json mode". The spike proved
the difference -- under bare json mode a model would return ``{}`` and drop the
expected ``edges`` key; under a schema it is forced to emit the declared shape,
so downstream parsing never has to guess.

A transport is injectable so the collector's tests run fully offline against a
mocked server, the same pattern the market-data client uses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from market_intelligence.llm.base import LLMError

if TYPE_CHECKING:
    from market_intelligence.config import Config


class OllamaProvider:
    """Structured JSON completion via a local ollama ``/api/chat`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout_seconds: float = 180.0,
        num_ctx: int = 16384,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._num_ctx = num_ctx
        self._transport = transport

    @classmethod
    def from_config(
        cls, config: Config, *, transport: httpx.BaseTransport | None = None
    ) -> OllamaProvider:
        llm = config.settings.llm
        return cls(
            base_url=llm.base_url,
            model=llm.model,
            timeout_seconds=llm.timeout_seconds,
            num_ctx=llm.num_ctx,
            transport=transport,
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Post one chat request and return the parsed JSON object.

        Every failure mode -- transport error, non-2xx, missing content, or
        content that is not the promised JSON -- is funnelled into
        :class:`LLMError` so the caller has exactly one exception to guard.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
        }
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc

        content = (body.get("message") or {}).get("content")
        if not content:
            raise LLMError("ollama returned an empty message")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"ollama returned non-JSON content: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"ollama returned a {type(parsed).__name__}, expected an object")
        return parsed


__all__ = ["OllamaProvider"]
