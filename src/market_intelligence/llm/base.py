"""The LLM provider interface: structured JSON completion, nothing more.

The connection engine needs exactly one thing from a model -- given a system
instruction, a user payload, and a JSON schema, return an object matching that
schema. Keeping the interface this narrow is what lets the backend be swapped
(ollama now, a hosted API or a different local model later) without any
collector knowing which model answered. The provider is a hypothesis generator;
correctness is decided downstream by the validation gate, not here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when the model call fails or returns unparseable output.

    Extraction must degrade, never crash a run: a collector catches this per
    section and moves on, so one bad response cannot end a batch of thousands.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """A model that returns schema-constrained JSON.

    ``model`` names the concrete model for provenance, so an edge records which
    model proposed it and a later re-run with a better model is distinguishable.
    """

    model: str

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return an object matching ``schema``, or raise :class:`LLMError`.

        ``temperature`` defaults to 0 because extraction wants determinism, not
        creativity: the same filing should yield the same edges every run.
        """
        ...


__all__ = ["LLMError", "LLMProvider"]
