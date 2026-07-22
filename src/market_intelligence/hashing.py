"""Deterministic content hashing and record-id generation.

Everything here is pure and side-effect free. Hashes are hex SHA-256.
Used to (a) fingerprint raw payloads for the raw store, and (b) derive
stable record ids from natural keys so re-collection is idempotent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators).

    ``default=str`` lets us hash objects containing dates/Decimals etc.
    without raising.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_text(canonical_json(obj))


def content_hash(*parts: Any) -> str:
    """Stable id from an ordered set of natural-key parts.

    ``None`` is rendered as the empty string so key shape is preserved.
    """
    joined = "|".join("" if p is None else str(p) for p in parts)
    return sha256_text(joined)


def short_hash(digest: str, length: int = 12) -> str:
    """First ``length`` characters of a hex digest (for filenames)."""
    return digest[:length]
