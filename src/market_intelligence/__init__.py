"""Local market-intelligence data platform.

Baseline phase: durable ingestion of free financial & economic data
(SEC EDGAR, FRED) into raw JSON + Parquet + DuckDB, with validation,
provenance tracking, structured logging, and idempotent writes.

No trading logic. This package is a data foundation only.
"""

__version__ = "0.1.0"
