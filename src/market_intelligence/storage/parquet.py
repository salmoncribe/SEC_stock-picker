"""Durable Parquet datasets with idempotent merge-on-write.

Each write reads any existing partition file, concatenates the new rows,
drops duplicates on the natural key (last wins), and atomically replaces the
file. This mirrors the DuckDB idempotency guarantee so the two analytical
stores stay consistent.

Layout::

    <parquet_dir>/<dataset>/<dataset>.parquet                 # unpartitioned
    <parquet_dir>/<dataset>/<partition_col>=<value>/data.parquet  # partitioned
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _sanitize(value: Any) -> str:
    cleaned = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in str(value))
    return cleaned or "null"


def _merge_write(path: Path, frame: pd.DataFrame, key_cols: Sequence[str]) -> str:
    if path.exists():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, frame], ignore_index=True)

    subset = [col for col in key_cols if col in frame.columns]
    if subset:
        frame = frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return str(path)


def write_records(
    parquet_dir: str | Path,
    dataset: str,
    records: Iterable[dict[str, Any]],
    key_cols: Sequence[str],
    *,
    partition_col: str | None = None,
) -> list[str]:
    """Merge ``records`` into the ``dataset`` Parquet store. Returns file paths."""
    rows = list(records)
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    base = Path(parquet_dir) / dataset
    base.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    if partition_col and partition_col in frame.columns:
        for value, group in frame.groupby(partition_col, dropna=False):
            part_dir = base / f"{partition_col}={_sanitize(value)}"
            part_dir.mkdir(parents=True, exist_ok=True)
            written.append(
                _merge_write(part_dir / "data.parquet", group.reset_index(drop=True), key_cols)
            )
    else:
        written.append(_merge_write(base / f"{dataset}.parquet", frame, key_cols))

    return written
