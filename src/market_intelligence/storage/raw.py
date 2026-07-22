"""Raw payload store — preserves source responses byte-for-byte.

Deterministic layout::

    <raw_dir>/<source>/<data_type>/<identifier>/<YYYY-MM-DD>/<hash12>.<ext>

The content hash is part of the filename, so:

* identical bytes collected again on the same day resolve to the same path
  and are skipped (idempotent);
* different bytes get a different filename and never silently overwrite an
  existing raw file.

Writes are atomic (temp file in the same directory + ``os.replace``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from market_intelligence.hashing import sha256_bytes, short_hash


@dataclass(frozen=True)
class RawSaveResult:
    path: str
    content_hash: str
    size: int
    was_new: bool


def _sanitize(part: str) -> str:
    cleaned = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in str(part))
    return cleaned or "unknown"


def save_raw(
    raw_dir: str | Path,
    source: str,
    data_type: str,
    identifier: str,
    content: bytes,
    *,
    collected_date: date | None = None,
    ext: str = "json",
) -> RawSaveResult:
    """Persist ``content`` under a deterministic, hash-named path."""
    if not isinstance(content, bytes):
        raise TypeError("raw content must be bytes")

    when = collected_date or date.today()
    digest = sha256_bytes(content)
    filename = f"{short_hash(digest)}.{ext.lstrip('.')}"

    directory = (
        Path(raw_dir)
        / _sanitize(source)
        / _sanitize(data_type)
        / _sanitize(identifier)
        / when.isoformat()
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename

    if target.exists():
        return RawSaveResult(str(target), digest, len(content), was_new=False)

    fd, tmp_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return RawSaveResult(str(target), digest, len(content), was_new=True)
