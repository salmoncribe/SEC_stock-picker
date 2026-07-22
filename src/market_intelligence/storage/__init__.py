"""Storage layer: raw payloads, Parquet datasets, DuckDB tables.

Import the submodules explicitly, e.g.::

    from market_intelligence.storage import raw, parquet
    from market_intelligence.storage import duckdb as duckdb_store
"""

from __future__ import annotations

from market_intelligence.storage.normalized import SectionWriteResult, write_section_text
from market_intelligence.storage.raw import RawSaveResult, save_raw

__all__ = ["RawSaveResult", "SectionWriteResult", "save_raw", "write_section_text"]
