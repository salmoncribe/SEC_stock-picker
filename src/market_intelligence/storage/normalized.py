"""Normalized text store — the cleaned, derived rendering of source documents.

Deterministic layout::

    <normalized_dir>/sections/<cik>/<accession_nodash>/<item_code>.txt

This store differs from ``storage.raw`` on purpose:

* the **raw** store is immutable evidence — files are hash-named, so identical
  bytes resolve to the same path and differing bytes never overwrite;
* the **normalized** store is a *derived* rendering addressed by its logical
  key ``(accession, item_code)``. Re-extracting with an improved parser is
  expected to replace the file in place, which is why the record carries
  ``text_sha256`` — the hash travels with the row rather than the filename.

Writes are atomic (temp file in the same directory + ``os.replace``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from market_intelligence.hashing import sha256_text


@dataclass(frozen=True)
class SectionWriteResult:
    path: str
    text_sha256: str
    char_count: int
    word_count: int
    changed: bool


def _sanitize(part: str) -> str:
    cleaned = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in str(part))
    return cleaned or "unknown"


def write_section_text(
    normalized_dir: str | Path,
    *,
    cik: str,
    accession_number: str,
    item_code: str,
    text: str,
) -> SectionWriteResult:
    """Persist one extracted section's text and report its hash and size.

    ``changed`` is False when the file already held byte-identical text, which
    lets a re-run distinguish "nothing moved" from "re-extracted differently".
    """
    directory = (
        Path(normalized_dir)
        / "sections"
        / _sanitize(cik)
        / _sanitize(accession_number.replace("-", ""))
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_sanitize(item_code)}.txt"

    digest = sha256_text(text)
    encoded = text.encode("utf-8")

    if target.exists() and target.read_bytes() == encoded:
        return SectionWriteResult(str(target), digest, len(text), len(text.split()), changed=False)

    fd, tmp_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return SectionWriteResult(str(target), digest, len(text), len(text.split()), changed=True)


__all__ = ["SectionWriteResult", "write_section_text"]
