"""FRED collector: fetch series + observations -> validate -> raw/Parquet/DuckDB.

``sync`` iterates the configured (or explicitly requested) series, fetching
each series' metadata and observation history, persisting the raw payloads,
validating the normalized records, and upserting the non-rejected rows into
DuckDB and the Parquet mirror. All bookkeeping (the ``pipeline_runs`` row) is
handled by ``pipeline_run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from market_intelligence import hashing
from market_intelligence.clients.fred import FREDClient
from market_intelligence.collectors import RunSummary, pipeline_run
from market_intelligence.schemas.common import utcnow
from market_intelligence.schemas.fred import (
    ObservationRecord,
    SeriesRecord,
    parse_observations,
    parse_series_metadata,
)
from market_intelligence.storage import duckdb as duckdb_store
from market_intelligence.storage import parquet, raw
from market_intelligence.validators.fred import validate_observation, validate_series

if TYPE_CHECKING:
    import httpx

    from market_intelligence.config import Config


def _clean_date(value: Any) -> Any:
    """Normalize an optional date field: empty strings become ``None``.

    Pydantic coerces ISO date strings to ``datetime.date`` on assignment; this
    only guards against FRED returning ``""`` where a date is absent.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


def sync(
    config: Config,
    *,
    series_ids: list[str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> RunSummary:
    """Collect FRED series metadata + observations for the configured series."""
    with pipeline_run(config, "fred.sync") as (con, summary):
        client = FREDClient.from_config(config, transport=transport)
        with client:
            targets = series_ids or [s.id for s in config.fred_series.series]
            start = config.fred_series.defaults.observation_start
            end = config.fred_series.defaults.observation_end
            schema_version = config.settings.app.schema_version

            series_rows: list[dict[str, Any]] = []
            obs_rows: list[dict[str, Any]] = []
            collected = 0
            rejected = 0

            for sid in targets:
                # --- series metadata ------------------------------------- #
                series_fetch = client.fetch_series(sid)
                raw.save_raw(config.paths.raw_dir, "fred", "series", sid, series_fetch.raw)
                meta = parse_series_metadata(series_fetch.data)
                series_record = SeriesRecord(
                    series_id=meta.get("id") or sid,
                    title=meta.get("title"),
                    units=meta.get("units"),
                    units_short=meta.get("units_short"),
                    frequency=meta.get("frequency"),
                    frequency_short=meta.get("frequency_short"),
                    seasonal_adjustment=meta.get("seasonal_adjustment"),
                    seasonal_adjustment_short=meta.get("seasonal_adjustment_short"),
                    observation_start=_clean_date(meta.get("observation_start")),
                    observation_end=_clean_date(meta.get("observation_end")),
                    last_updated=meta.get("last_updated"),
                    popularity=meta.get("popularity"),
                    notes=meta.get("notes"),
                    source_url=series_fetch.url,
                    content_hash=hashing.sha256_bytes(series_fetch.raw),
                    schema_version=schema_version,
                    collected_time=utcnow(),
                )
                validate_series(series_record)
                collected += 1
                if series_record.is_rejected:
                    rejected += 1
                else:
                    series_rows.append(series_record.to_row())

                # --- observations ---------------------------------------- #
                obs_fetch = client.fetch_observations(
                    sid, observation_start=start, observation_end=end
                )
                raw_result = raw.save_raw(
                    config.paths.raw_dir, "fred", "observations", sid, obs_fetch.raw
                )
                observations = parse_observations(obs_fetch.data)
                obs_count = 0
                for obs in observations:
                    observation_date = obs["observation_date"]
                    if observation_date is None:
                        continue  # cannot form a valid key without a date
                    value = obs["value"]
                    realtime_start = obs["realtime_start"]
                    realtime_end = obs["realtime_end"]
                    obs_record = ObservationRecord(
                        observation_id=hashing.content_hash(
                            "obs",
                            sid,
                            observation_date.isoformat(),
                            realtime_start,
                            realtime_end,
                        ),
                        series_id=sid,
                        observation_date=observation_date,
                        value=value,
                        realtime_start=realtime_start,
                        realtime_end=realtime_end,
                        raw_file_path=raw_result.path,
                        source_url=obs_fetch.url,
                        content_hash=hashing.content_hash(sid, observation_date.isoformat(), value),
                        schema_version=schema_version,
                        collected_time=utcnow(),
                    )
                    validate_observation(obs_record)
                    collected += 1
                    obs_count += 1
                    if obs_record.is_rejected:
                        rejected += 1
                    else:
                        obs_rows.append(obs_record.to_row())

                summary.note(f"{sid}: {obs_count} observations")

            # --- persist ------------------------------------------------- #
            series_result = duckdb_store.upsert_series(con, series_rows)
            obs_result = duckdb_store.upsert_observations(con, obs_rows)

            parquet.write_records(
                config.paths.parquet_dir, "economic_series", series_rows, ["series_id"]
            )
            parquet.write_records(
                config.paths.parquet_dir,
                "economic_observations",
                obs_rows,
                ["observation_id"],
                partition_col="series_id",
            )

            summary.collected = collected
            summary.inserted = series_result.inserted + obs_result.inserted
            summary.updated = series_result.updated + obs_result.updated
            summary.rejected = rejected

    return summary


__all__ = ["sync"]
