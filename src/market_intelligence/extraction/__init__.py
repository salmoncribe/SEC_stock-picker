"""Deterministic text extraction from raw filing documents.

Pure, offline parsing only — nothing here fetches, writes, or depends on wall-clock
time, so collectors can re-derive sections from archived raw bytes at any time and get
byte-identical output.
"""

from __future__ import annotations

from market_intelligence.extraction.sections import (
    EXTRACTION_METHOD,
    ITEMS_10K,
    ITEMS_10Q,
    ExtractedSection,
    extract_sections,
    html_to_text,
)

__all__ = [
    "EXTRACTION_METHOD",
    "ITEMS_10K",
    "ITEMS_10Q",
    "ExtractedSection",
    "extract_sections",
    "html_to_text",
]
