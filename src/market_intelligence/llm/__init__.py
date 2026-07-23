"""Local LLM inference for the connection engine.

A thin provider interface over whatever model runs the relationship extraction,
so the backend (ollama today) can be swapped without touching the collectors
that use it. The LLM only proposes edges; the event-study harness judges them.
"""

from market_intelligence.llm.base import LLMError, LLMProvider
from market_intelligence.llm.ollama import OllamaProvider

__all__ = ["LLMError", "LLMProvider", "OllamaProvider"]
