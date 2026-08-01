"""DuckDB connection management and schema definition (DDL).

``init_db`` creates the analytical tables (idempotent — ``CREATE TABLE IF NOT
EXISTS``). Per-table write/upsert logic lives in ``storage/duckdb.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

TABLES: tuple[str, ...] = (
    "companies",
    "filings",
    "filing_documents",
    "filing_sections",
    "economic_series",
    "economic_observations",
    "index_constituents",
    "daily_prices",
    "daily_returns",
    "events",
    "event_samples",
    "impact_stats",
    "signal_status",
    "people",
    "role_memberships",
    "company_edges",
    "processed_relationship_sections",
    "filing_packages",
    "source_observations",
    "market_snapshots",
    "strategy_versions",
    "experiment_registry",
    "relationship_opportunities",
    "opportunity_decisions",
    "paper_orders",
    "paper_fills",
    "paper_account_marks",
    "kill_switch_events",
    "system_health",
    "trade_alerts",
    "gap_calibration_stats",
    "dataset_build_watermarks",
    "pipeline_runs",
)

# One statement per table so we can execute them individually.
SCHEMA_STATEMENTS: dict[str, str] = {
    "companies": """
        CREATE TABLE IF NOT EXISTS companies (
            company_id        TEXT PRIMARY KEY,
            ticker            TEXT,
            company_name      TEXT,
            cik               TEXT NOT NULL UNIQUE,
            exchange          TEXT,
            is_active         BOOLEAN,
            first_seen_time   TIMESTAMPTZ,
            last_seen_time    TIMESTAMPTZ,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ
        )
    """,
    "filings": """
        CREATE TABLE IF NOT EXISTS filings (
            filing_id         TEXT PRIMARY KEY,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL UNIQUE,
            form              TEXT,
            filing_date       DATE,
            report_date       DATE,
            acceptance_time   TIMESTAMPTZ,
            primary_document  TEXT,
            filing_url        TEXT,
            raw_file_path     TEXT,
            content_hash      TEXT,
            collected_time    TIMESTAMPTZ,
            validation_status TEXT,
            validation_errors TEXT,
            source            TEXT,
            source_url        TEXT,
            schema_version    TEXT
        )
    """,
    "filing_documents": """
        CREATE TABLE IF NOT EXISTS filing_documents (
            document_id       TEXT PRIMARY KEY,
            filing_id         TEXT,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL,
            form              TEXT,
            document_name     TEXT NOT NULL,
            document_url      TEXT,
            document_type     TEXT,
            byte_size         BIGINT,
            declared_size     BIGINT,
            sha256            TEXT,
            raw_file_path     TEXT,
            content_type      TEXT,
            downloaded_time   TIMESTAMPTZ,
            integrity_status  TEXT,
            section_count     INTEGER,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (accession_number, document_name)
        )
    """,
    "filing_sections": """
        CREATE TABLE IF NOT EXISTS filing_sections (
            section_id        TEXT PRIMARY KEY,
            document_id       TEXT,
            filing_id         TEXT,
            company_id        TEXT,
            cik               TEXT,
            accession_number  TEXT NOT NULL,
            form              TEXT,
            report_date       DATE,
            item_code         TEXT NOT NULL,
            item_title        TEXT,
            section_order     INTEGER,
            char_count        INTEGER,
            word_count        INTEGER,
            text_path         TEXT,
            text_sha256       TEXT,
            preview           TEXT,
            extraction_method TEXT,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (accession_number, item_code)
        )
    """,
    "economic_series": """
        CREATE TABLE IF NOT EXISTS economic_series (
            series_id                 TEXT PRIMARY KEY,
            title                     TEXT,
            units                     TEXT,
            units_short               TEXT,
            frequency                 TEXT,
            frequency_short           TEXT,
            seasonal_adjustment       TEXT,
            seasonal_adjustment_short TEXT,
            observation_start         DATE,
            observation_end           DATE,
            last_updated              TEXT,
            popularity                INTEGER,
            notes                     TEXT,
            source                    TEXT,
            source_url                TEXT,
            content_hash              TEXT,
            schema_version            TEXT,
            validation_status         TEXT,
            validation_errors         TEXT,
            collected_time            TIMESTAMPTZ
        )
    """,
    "economic_observations": """
        CREATE TABLE IF NOT EXISTS economic_observations (
            observation_id    TEXT PRIMARY KEY,
            series_id         TEXT NOT NULL,
            observation_date  DATE NOT NULL,
            value             DOUBLE,
            realtime_start    DATE,
            realtime_end      DATE,
            collected_time    TIMESTAMPTZ,
            raw_file_path     TEXT,
            content_hash      TEXT,
            source            TEXT,
            source_url        TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            UNIQUE (series_id, observation_date, realtime_start, realtime_end)
        )
    """,
    # Point-in-time index membership. `removed_date IS NULL` means "still a
    # member". Storing the window (rather than a flat current list) is what lets
    # a backtest ask "who was in the index on 2019-03-14?" instead of "who is in
    # it now?" -- the difference between a real result and survivorship bias.
    "index_constituents": """
        CREATE TABLE IF NOT EXISTS index_constituents (
            constituent_id    TEXT PRIMARY KEY,
            index_id          TEXT NOT NULL,
            company_id        TEXT,
            cik               TEXT,
            ticker            TEXT NOT NULL,
            company_name      TEXT,
            added_date        DATE NOT NULL,
            removed_date      DATE,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (index_id, ticker, added_date)
        )
    """,
    "daily_prices": """
        CREATE TABLE IF NOT EXISTS daily_prices (
            price_id          TEXT PRIMARY KEY,
            symbol            TEXT NOT NULL,
            price_date        DATE NOT NULL,
            open              DOUBLE,
            high              DOUBLE,
            low               DOUBLE,
            close             DOUBLE,
            adj_close         DOUBLE,
            volume            BIGINT,
            provider          TEXT,
            is_delisted_gap   BOOLEAN,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (symbol, price_date)
        )
    """,
    # `abnormal_return` is the label the whole signal layer is graded against.
    # `estimation_window_start` records which trailing window produced `beta`,
    # so a stored row can be re-derived and audited for lookahead.
    "daily_returns": """
        CREATE TABLE IF NOT EXISTS daily_returns (
            return_id               TEXT PRIMARY KEY,
            symbol                  TEXT NOT NULL,
            price_date              DATE NOT NULL,
            total_return            DOUBLE,
            market_return           DOUBLE,
            sector_return           DOUBLE,
            abnormal_return         DOUBLE,
            beta                    DOUBLE,
            alpha                   DOUBLE,
            method                  TEXT,
            estimation_window_start DATE,
            source                  TEXT,
            source_url              TEXT,
            content_hash            TEXT,
            schema_version          TEXT,
            validation_status       TEXT,
            validation_errors       TEXT,
            collected_time          TIMESTAMPTZ,
            UNIQUE (symbol, price_date)
        )
    """,
    # One typed, timestamped thing that happened at one company.
    #
    # Two clocks, deliberately separate, because conflating them is the single
    # easiest way to fabricate a backtest result:
    #
    #   event_time     -- when it happened in the world (e.g. the trade date on
    #                     a Form 4). Unknowable to the market at the time.
    #   available_time -- when the public could first have known. This is the
    #                     only clock a feature or a label may key on.
    #
    # For an insider trade those differ by up to two business days, and using
    # event_time as t=0 would be trading on information nobody had yet.
    #
    # `event_type` is a plain string, not an enum, so a new kind of event is a
    # config entry and a re-run rather than a schema migration.
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            event_id              TEXT PRIMARY KEY,
            company_id            TEXT,
            cik                   TEXT,
            ticker                TEXT,
            event_type            TEXT NOT NULL,
            event_subtype         TEXT,
            event_key             TEXT NOT NULL,
            accession_number      TEXT,
            filing_id             TEXT,
            event_time            TIMESTAMPTZ,
            available_time        TIMESTAMPTZ,
            magnitude             DOUBLE,
            direction             INTEGER,
            payload               TEXT,
            extraction_method     TEXT,
            extraction_confidence DOUBLE,
            source                TEXT,
            source_url            TEXT,
            content_hash          TEXT,
            schema_version        TEXT,
            validation_status     TEXT,
            validation_errors     TEXT,
            collected_time        TIMESTAMPTZ,
            UNIQUE (event_type, event_key)
        )
    """,
    # One (event, target company, horizon) observation: what happened to the
    # target over the N trading days after the event became actionable.
    #
    # This is the single contract every statistic downstream is computed from,
    # so it stores the *derivation* alongside the answer: `available_on` (when
    # the public could know), `t0` (the first tradeable day after that), and
    # `window_end`. Keeping all three means a stored row can be re-derived and
    # audited for leakage rather than trusted.
    #
    # `edge_id` is 'self' for a single-company event such as an insider trade,
    # and a graph edge id once propagation events arrive -- so the same table
    # serves the positive control and the real hypothesis without a second
    # schema.
    "event_samples": """
        CREATE TABLE IF NOT EXISTS event_samples (
            sample_id               TEXT PRIMARY KEY,
            event_id                TEXT NOT NULL,
            edge_id                 TEXT NOT NULL,
            event_type              TEXT,
            event_subtype           TEXT,
            source_ticker           TEXT,
            target_ticker           TEXT,
            horizon_days            INTEGER NOT NULL,
            available_on            DATE,
            t0                      DATE,
            window_end              DATE,
            forward_abnormal_return DOUBLE,
            magnitude               DOUBLE,
            direction               INTEGER,
            split                   TEXT,
            features                TEXT,
            source                  TEXT,
            source_url              TEXT,
            content_hash            TEXT,
            schema_version          TEXT,
            validation_status       TEXT,
            validation_errors       TEXT,
            collected_time          TIMESTAMPTZ,
            UNIQUE (event_id, edge_id, horizon_days, target_ticker)
        )
    """,
    # Measured behaviour of one (event kind, edge kind, horizon) combination on
    # one split, plus the verdict on whether it may fire alerts.
    #
    # n_samples and n_clusters are both stored and they are not redundant:
    # n_clusters is the number of independent observations (one per company-day)
    # and is what every statistic here is computed from, while n_samples is the
    # raw row count. Keeping both makes the clustering visible rather than
    # implicit -- a large gap between them is exactly the condition under which
    # an unclustered statistic would have been badly overconfident.
    #
    # The verdict is a property of the cell, not of a split: it is decided on
    # discovery and stamped on both rows so either can be read alone.
    "impact_stats": """
        CREATE TABLE IF NOT EXISTS impact_stats (
            stat_id           TEXT PRIMARY KEY,
            event_type        TEXT NOT NULL,
            event_subtype     TEXT,
            edge_type         TEXT NOT NULL,
            horizon_days      INTEGER NOT NULL,
            split             TEXT NOT NULL,
            n_samples         INTEGER,
            n_clusters        INTEGER,
            mean_car          DOUBLE,
            median_car        DOUBLE,
            std_car           DOUBLE,
            hit_rate          DOUBLE,
            t_stat            DOUBLE,
            verdict           TEXT,
            verdict_reason    TEXT,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (event_type, event_subtype, edge_type, horizon_days, split)
        )
    """,
    "signal_status": """
        CREATE TABLE IF NOT EXISTS signal_status (
            signal_id           TEXT PRIMARY KEY,
            event_type          TEXT NOT NULL,
            event_subtype       TEXT,
            edge_type           TEXT NOT NULL,
            horizon_days        INTEGER NOT NULL,
            status              TEXT NOT NULL,
            confirm_streak      INTEGER NOT NULL,
            fail_streak         INTEGER NOT NULL,
            holdout_clusters    INTEGER,
            last_verdict        TEXT,
            last_reason         TEXT,
            mean_car            DOUBLE,
            hit_rate            DOUBLE,
            n_clusters          INTEGER,
            direction           INTEGER,
            first_seen_time     TIMESTAMPTZ,
            became_active_time  TIMESTAMPTZ,
            last_evaluated_time TIMESTAMPTZ,
            schema_version      TEXT,
            UNIQUE (event_type, event_subtype, edge_type, horizon_days)
        )
    """,
    # One insider's persistent identity, keyed on their SEC reporting-owner
    # CIK -- see docs/specs/2026-07-25-people-insider-graph-design.md Decision
    # A. Re-projected from REPORTINGOWNER rows the insider collector already
    # fetched; no new network fetch backs this table.
    "people": """
        CREATE TABLE IF NOT EXISTS people (
            person_id              TEXT PRIMARY KEY,
            reporting_owner_cik    TEXT NOT NULL UNIQUE,
            canonical_name         TEXT,
            name_variants          TEXT,
            first_seen_filing_date DATE,
            last_seen_filing_date  DATE,
            source                 TEXT,
            source_url             TEXT,
            content_hash           TEXT,
            schema_version         TEXT,
            validation_status      TEXT,
            validation_errors      TEXT,
            collected_time         TIMESTAMPTZ
        )
    """,
    # One person's relationship to one company. Flags are OR'd across every
    # qualifying filing seen for that (person, company) pair -- a person who
    # files once as director and later as an officer keeps both flags true.
    "role_memberships": """
        CREATE TABLE IF NOT EXISTS role_memberships (
            role_id              TEXT PRIMARY KEY,
            person_id            TEXT NOT NULL,
            company_id           TEXT NOT NULL,
            is_officer           BOOLEAN,
            is_director          BOOLEAN,
            is_ten_pct_owner     BOOLEAN,
            latest_officer_title TEXT,
            first_seen           DATE,
            last_seen            DATE,
            source_filing_count  INTEGER,
            source               TEXT,
            source_url           TEXT,
            content_hash         TEXT,
            schema_version       TEXT,
            validation_status    TEXT,
            validation_errors    TEXT,
            collected_time       TIMESTAMPTZ,
            UNIQUE (person_id, company_id)
        )
    """,
    "company_edges": """
        CREATE TABLE IF NOT EXISTS company_edges (
            edge_id               TEXT PRIMARY KEY,
            edge_key              TEXT NOT NULL,
            source_cik            TEXT NOT NULL,
            source_ticker         TEXT,
            source_company_id     TEXT,
            target                TEXT NOT NULL,
            target_name           TEXT NOT NULL,
            target_cik            TEXT,
            target_ticker         TEXT,
            edge_type             TEXT NOT NULL,
            resolution_status     TEXT,
            resolution_confidence DOUBLE,
            evidence              TEXT,
            extraction_confidence DOUBLE,
            extraction_method     TEXT,
            extraction_model      TEXT,
            accession_number      TEXT,
            filing_id             TEXT,
            report_date           DATE,
            times_asserted        INTEGER,
            first_seen_time       TIMESTAMPTZ,
            last_seen_time        TIMESTAMPTZ,
            source                TEXT,
            source_url            TEXT,
            source_record_id      TEXT,
            event_time            TIMESTAMPTZ,
            published_time        TIMESTAMPTZ,
            content_hash          TEXT,
            schema_version        TEXT,
            validation_status     TEXT,
            validation_errors     TEXT,
            collected_time        TIMESTAMPTZ,
            UNIQUE (edge_key)
        )
    """,
    "processed_relationship_sections": """
        CREATE TABLE IF NOT EXISTS processed_relationship_sections (
            accession_number   TEXT NOT NULL,
            item_code          TEXT NOT NULL,
            source_ticker      TEXT,
            processed_time     TIMESTAMPTZ,
            edge_count         INTEGER,
            schema_version     TEXT,
            PRIMARY KEY (accession_number, item_code)
        )
    """,
    # Immutable SEC disclosure package.  `first_public_state` is intentionally
    # distinct from `edgar_accepted_at`: EDGAR acceptance is regulatory timing,
    # not proof that the market broadly received a disclosure.
    "filing_packages": """
        CREATE TABLE IF NOT EXISTS filing_packages (
            package_id                  TEXT PRIMARY KEY,
            accession_number             TEXT NOT NULL UNIQUE,
            filing_id                    TEXT,
            cik                          TEXT,
            form                         TEXT,
            is_amendment                 BOOLEAN,
            amends_accession_number      TEXT,
            supersedes_package_id        TEXT,
            complete_submission_sha256   TEXT NOT NULL,
            primary_document_sha256      TEXT,
            exhibit_hashes               TEXT,
            event_occurred_at            TIMESTAMPTZ,
            issuer_claimed_release_at    TIMESTAMPTZ,
            edgar_accepted_at            TIMESTAMPTZ,
            sec_dissemination_observed_at TIMESTAMPTZ,
            first_public_at              TIMESTAMPTZ,
            first_public_state           TEXT NOT NULL,
            parser_version               TEXT NOT NULL,
            raw_file_path                TEXT,
            created_at                   TIMESTAMPTZ NOT NULL,
            UNIQUE (accession_number, complete_submission_sha256)
        )
    """,
    # A source may be observed repeatedly. This table never overwrites a
    # receipt timestamp, which preserves reconciliation gaps and late feeds.
    "source_observations": """
        CREATE TABLE IF NOT EXISTS source_observations (
            observation_id               TEXT PRIMARY KEY,
            package_id                   TEXT,
            source_name                  TEXT NOT NULL,
            source_url                   TEXT,
            source_record_id             TEXT,
            raw_sha256                   TEXT NOT NULL,
            observed_at                  TIMESTAMPTZ NOT NULL,
            our_fetch_at                 TIMESTAMPTZ NOT NULL,
            parse_complete_at            TIMESTAMPTZ,
            observation_kind             TEXT NOT NULL,
            status                       TEXT NOT NULL,
            details                      TEXT,
            UNIQUE (source_name, source_record_id, raw_sha256, observed_at)
        )
    """,
    "market_snapshots": """
        CREATE TABLE IF NOT EXISTS market_snapshots (
            snapshot_id                  TEXT PRIMARY KEY,
            ticker                       TEXT NOT NULL,
            provider                     TEXT NOT NULL,
            exchange_event_at            TIMESTAMPTZ,
            local_receipt_at             TIMESTAMPTZ NOT NULL,
            feed_sequence                BIGINT,
            bid                          DOUBLE,
            ask                          DOUBLE,
            bid_size                     BIGINT,
            ask_size                     BIGINT,
            bid_venue                    TEXT,
            ask_venue                    TEXT,
            midpoint                     DOUBLE,
            market_status                TEXT,
            session                      TEXT,
            halt_state                   TEXT,
            luld_state                   TEXT,
            ssr_active                   BOOLEAN,
            one_minute_volume            BIGINT,
            adv                          DOUBLE,
            volatility                   DOUBLE,
            market_return                DOUBLE,
            sector_return                DOUBLE,
            p50_spread                   DOUBLE,
            p95_spread                   DOUBLE,
            p95_slippage                 DOUBLE,
            p95_impact                   DOUBLE,
            gap_r_p95                    DOUBLE,
            borrow_payload               TEXT,
            source_hash                  TEXT NOT NULL,
            validity                     TEXT NOT NULL,
            suppression_reasons          TEXT NOT NULL,
            created_at                   TIMESTAMPTZ NOT NULL
        )
    """,
    "strategy_versions": """
        CREATE TABLE IF NOT EXISTS strategy_versions (
            strategy_version_id          TEXT PRIMARY KEY,
            strategy_name                TEXT NOT NULL,
            policy_hash                  TEXT NOT NULL UNIQUE,
            policy_json                  TEXT NOT NULL,
            code_version                 TEXT,
            data_version                 TEXT,
            prompt_version               TEXT,
            cost_model_version           TEXT,
            hypothesis                   TEXT,
            created_at                   TIMESTAMPTZ NOT NULL,
            retired_at                   TIMESTAMPTZ
        )
    """,
    "experiment_registry": """
        CREATE TABLE IF NOT EXISTS experiment_registry (
            experiment_id                TEXT PRIMARY KEY,
            strategy_version_id          TEXT NOT NULL,
            hypothesis                   TEXT NOT NULL,
            discovery_window             TEXT NOT NULL,
            calibration_window           TEXT NOT NULL,
            sealed_test_window           TEXT NOT NULL,
            feature_hash                 TEXT NOT NULL,
            result_json                  TEXT,
            registered_at                TIMESTAMPTZ NOT NULL,
            completed_at                 TIMESTAMPTZ,
            UNIQUE (strategy_version_id, feature_hash)
        )
    """,
    "relationship_opportunities": """
        CREATE TABLE IF NOT EXISTS relationship_opportunities (
            opportunity_id               TEXT PRIMARY KEY,
            event_id                     TEXT NOT NULL,
            edge_id                      TEXT NOT NULL,
            target_ticker                TEXT NOT NULL,
            horizon_days                 INTEGER NOT NULL,
            strategy_version             TEXT NOT NULL,
            evidence_score               DOUBLE NOT NULL,
            evidence_components          TEXT NOT NULL,
            evidence_snapshot_hash       TEXT NOT NULL,
            market_snapshot_id           TEXT,
            tradeability_json            TEXT NOT NULL,
            status                       TEXT NOT NULL,
            suppression_reasons          TEXT NOT NULL,
            candidate_scored_at          TIMESTAMPTZ NOT NULL,
            decision_at                  TIMESTAMPTZ,
            alert_sent_at                TIMESTAMPTZ,
            created_at                   TIMESTAMPTZ NOT NULL,
            updated_at                   TIMESTAMPTZ NOT NULL,
            UNIQUE (event_id, edge_id, target_ticker, horizon_days, strategy_version)
        )
    """,
    # Attempts are append-only.  No INSERT OR REPLACE/UPSERT must be used for
    # a decision, otherwise a timeout or model drift could erase the audit.
    "opportunity_decisions": """
        CREATE TABLE IF NOT EXISTS opportunity_decisions (
            decision_id                  TEXT PRIMARY KEY,
            opportunity_id               TEXT NOT NULL,
            input_hash                   TEXT NOT NULL,
            prompt_version               TEXT NOT NULL,
            model_version                TEXT NOT NULL,
            attempt_number               INTEGER NOT NULL,
            verdict                      TEXT NOT NULL,
            decision_json                TEXT,
            verification_status          TEXT NOT NULL,
            verification_reasons         TEXT NOT NULL,
            started_at                   TIMESTAMPTZ NOT NULL,
            completed_at                 TIMESTAMPTZ,
            UNIQUE (opportunity_id, input_hash, prompt_version, model_version, attempt_number)
        )
    """,
    # ``quantity`` is BIGINT and predates the decision to trade fractional
    # shares. It is kept (NOT NULL, and dropping a column is not something
    # migrate_db can do) as a rounded convenience index; ``quantity_exact`` is
    # the authoritative quantity the account folds. ``symbol`` and ``reason``
    # close the other two gaps: neither table could name the instrument it
    # traded, and an order could not record why it existed.
    "paper_orders": """
        CREATE TABLE IF NOT EXISTS paper_orders (
            paper_order_id               TEXT PRIMARY KEY,
            opportunity_id               TEXT NOT NULL,
            symbol                       TEXT,
            side                         TEXT NOT NULL,
            quantity                     BIGINT NOT NULL,
            quantity_exact               DOUBLE,
            reason                       TEXT,
            limit_price                  DOUBLE,
            submitted_at                 TIMESTAMPTZ NOT NULL,
            status                       TEXT NOT NULL,
            simulation_version           TEXT NOT NULL,
            UNIQUE (opportunity_id, simulation_version)
        )
    """,
    "paper_fills": """
        CREATE TABLE IF NOT EXISTS paper_fills (
            paper_fill_id                TEXT PRIMARY KEY,
            paper_order_id               TEXT NOT NULL,
            symbol                       TEXT,
            fill_price                   DOUBLE NOT NULL,
            quantity                     BIGINT NOT NULL,
            quantity_exact               DOUBLE,
            filled_at                    TIMESTAMPTZ NOT NULL,
            fees                         DOUBLE NOT NULL,
            borrow_cost                  DOUBLE NOT NULL,
            slippage                     DOUBLE NOT NULL,
            fill_assumptions             TEXT NOT NULL,
            UNIQUE (paper_order_id, filled_at)
        )
    """,
    # The daily equity mark, and the only place the high-water mark is durable.
    #
    # The account of record is a fold over ``paper_fills``, but ``apply_fill``
    # cannot maintain the high-water mark -- the ratchet lives in
    # ``account.mark``, which is a function of *marked* equity, and a fill log
    # carries no prices. So a rebuilt account's mark was always the opening
    # deposit, every subsequent peak lost, and the drawdown governor on the live
    # ``portfolio step`` path measured every drawdown from $10,000 and could
    # never fire.
    #
    # A column on ``paper_fills`` would not fix it: the mark moves on *marking*
    # days, and fills only happen on rebalance days (weekly by default). Every
    # peak struck on a session in between would still be lost. Hence one row per
    # session, written whether or not anything traded.
    #
    # ``UNIQUE (simulation_version, as_of)`` is the natural key -- one mark per
    # session per experiment -- and ``mark_id`` is derived from exactly those two
    # values, so a re-run rewrites its row instead of duplicating it.
    "paper_account_marks": """
        CREATE TABLE IF NOT EXISTS paper_account_marks (
            mark_id                      TEXT PRIMARY KEY,
            simulation_version           TEXT NOT NULL,
            as_of                        DATE NOT NULL,
            equity                       DOUBLE NOT NULL,
            cash                         DOUBLE NOT NULL,
            high_water_mark              DOUBLE NOT NULL,
            drawdown                     DOUBLE NOT NULL,
            marked_at                    TIMESTAMPTZ NOT NULL,
            UNIQUE (simulation_version, as_of)
        )
    """,
    "kill_switch_events": """
        CREATE TABLE IF NOT EXISTS kill_switch_events (
            kill_switch_event_id         TEXT PRIMARY KEY,
            switch_name                  TEXT NOT NULL,
            state                        TEXT NOT NULL,
            reason                       TEXT NOT NULL,
            actor                        TEXT,
            occurred_at                  TIMESTAMPTZ NOT NULL,
            review_required              BOOLEAN NOT NULL,
            resolved_at                  TIMESTAMPTZ,
            UNIQUE (switch_name, occurred_at)
        )
    """,
    "system_health": """
        CREATE TABLE IF NOT EXISTS system_health (
            health_id                    TEXT PRIMARY KEY,
            component                    TEXT NOT NULL,
            status                       TEXT NOT NULL,
            observed_at                  TIMESTAMPTZ NOT NULL,
            details                      TEXT,
            source_hash                  TEXT,
            UNIQUE (component, observed_at)
        )
    """,
    # The trade-alert ledger: every fired alert persisted as a discrete graded
    # prediction, the same way `signal_status` gives the signal layer its own
    # track record.
    #
    # `UNIQUE (kind, ticker, trigger_key)` is the natural key that makes
    # re-firing a no-op -- the daily briefing's lookback window re-surfaces the
    # same event on consecutive runs, and the gap scanner can be re-run within
    # a morning, so the insert helper must skip rows whose key already exists
    # rather than upsert over them (see `insert_new_trade_alerts` in
    # `storage/duckdb.py`).
    #
    # This table supersedes the `alerts` table designed but never built in the
    # signal-graph spec: same purpose (persist fired predictions and backfill
    # outcomes), now carrying the trade plan as well. That table will not be
    # built separately.
    "trade_alerts": """
        CREATE TABLE IF NOT EXISTS trade_alerts (
            alert_id          TEXT PRIMARY KEY,
            kind              TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            trigger_key       TEXT NOT NULL,
            fired_at          TIMESTAMPTZ,
            direction         INTEGER,
            entry_ref         DOUBLE,
            stop              DOUBLE,
            target            DOUBLE,
            shares            INTEGER,
            notional          DOUBLE,
            risk_amount       DOUBLE,
            time_exit_date    DATE,
            confidence        INTEGER,
            evidence          TEXT,
            event_id          TEXT,
            edge_id           TEXT,
            delivered         BOOLEAN,
            delivery_note     TEXT,
            outcome           TEXT,
            outcome_return    DOUBLE,
            graded_at         TIMESTAMPTZ,
            unsizeable        BOOLEAN,
            source            TEXT,
            source_url        TEXT,
            content_hash      TEXT,
            schema_version    TEXT,
            validation_status TEXT,
            validation_errors TEXT,
            collected_time    TIMESTAMPTZ,
            UNIQUE (kind, ticker, trigger_key)
        )
    """,
    # The gap scanner's own track record, one row per (direction, gap size
    # bucket, catalyst present) combination -- see
    # ``signals/gap_calibration.py``. Deliberately separate from
    # ``impact_stats``/``signal_status``: those are keyed on
    # ``event_samples`` rows (event_type/edge_type/horizon), and a
    # ``price_gap`` alert has no ``event_id`` to join on, so it cannot
    # populate that table. This one is rebuilt from scratch (delete +
    # reinsert) every run rather than upserted -- the bucket count is small
    # and an incremental update would need to distinguish "recompute this
    # bucket's aggregate" from "this row is new", which is more moving parts
    # than the win is worth. ``bucket_id`` is the deterministic key
    # (``f"{direction}:{gap_size_bucket}:{catalyst_present}"``), not a
    # surrogate, so a bucket's identity never depends on insert order.
    "gap_calibration_stats": """
        CREATE TABLE IF NOT EXISTS gap_calibration_stats (
            bucket_id         TEXT PRIMARY KEY,
            direction         INTEGER,
            gap_size_bucket   TEXT,
            catalyst_present  BOOLEAN,
            n_decisive        INTEGER,
            n_expired         INTEGER,
            hit_rate          DOUBLE,
            mean_return       DOUBLE,
            last_updated      TIMESTAMPTZ
        )
    """,
    # Per-symbol "did anything this symbol's build depends on actually change"
    # marker for signals.dataset.build_incremental. Keyed on (symbol,
    # build_fingerprint) rather than symbol alone because horizons, the split
    # date, the event_type/subtype filter, and schema_version all change what
    # a symbol's built rows should look like -- a watermark recorded under one
    # parameter set must never be read as valid under a different one.
    #
    # returns_checksum / events_checksum are content-hash-based, not
    # collected_time-based: returns.compute() recomputes and re-upserts a
    # symbol's *entire* return history on every run (unscoped by default),
    # which stamps a fresh collected_time on every row whether or not its
    # value changed. A checksum over each row's content_hash only moves when
    # the data actually does.
    #
    # Checksums are HUGEINT, not BIGINT: SUM() over many UBIGINT hash()
    # outputs promotes to DuckDB's 128-bit INT128 to avoid overflowing
    # mid-sum, and that value can legitimately exceed BIGINT's 64-bit signed
    # range for even a few dozen rows.
    "dataset_build_watermarks": """
        CREATE TABLE IF NOT EXISTS dataset_build_watermarks (
            symbol             TEXT NOT NULL,
            build_fingerprint  TEXT NOT NULL,
            returns_row_count  BIGINT,
            returns_checksum   HUGEINT,
            events_row_count   BIGINT,
            events_checksum    HUGEINT,
            last_built_time    TIMESTAMPTZ,
            PRIMARY KEY (symbol, build_fingerprint)
        )
    """,
    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id            TEXT PRIMARY KEY,
            pipeline_name     TEXT,
            started_time      TIMESTAMPTZ,
            completed_time    TIMESTAMPTZ,
            status            TEXT,
            records_collected INTEGER,
            records_inserted  INTEGER,
            records_updated   INTEGER,
            records_rejected  INTEGER,
            error_message     TEXT,
            config_hash       TEXT
        )
    """,
}


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (creating parent dirs) a DuckDB connection to ``db_path``.

    The session timezone is pinned to UTC so ``TIMESTAMPTZ`` columns round-trip
    as UTC (the platform's internal clock) rather than the host's local zone.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone='UTC'")
    return con


@contextmanager
def connection(db_path: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-managed DuckDB connection."""
    con = connect(db_path)
    try:
        yield con
    finally:
        con.close()


#: Default bounded retry for ``with_db_retry`` -- matches what
#: ``collectors/gaps.py`` has used since Task 8 (3 attempts, 30s apart).
#: Callers whose lock-holder is known to run longer (e.g. the daily autopilot
#: loop racing the relationship-extraction watchdog) pass their own, larger
#: budget rather than changing this shared default.
DB_BUSY_MAX_ATTEMPTS = 3
DB_BUSY_RETRY_SECONDS = 30.0


def with_db_retry[T](
    fn: Callable[[], T],
    *,
    sleep: Callable[[float], None],
    max_attempts: int = DB_BUSY_MAX_ATTEMPTS,
    retry_seconds: float = DB_BUSY_RETRY_SECONDS,
) -> T:
    """Call ``fn()``, retrying on ``duckdb.IOException`` (the lock is busy).

    DuckDB is single-writer: a second ``connect()`` while another process
    holds the file fails immediately rather than queuing. ``fn`` is expected
    to own a full connect-use-close cycle (typically via ``connection()``)
    and return a plain value, so a retry is a clean do-over, not a resume.
    Re-raises the last ``IOException`` once ``max_attempts`` is exhausted --
    the caller decides what "give up" means.
    """
    last_error: duckdb.IOException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except duckdb.IOException as exc:
            last_error = exc
            if attempt < max_attempts:
                sleep(retry_seconds)
    assert last_error is not None  # loop always sets it before falling through
    raise last_error


# Columns added to existing tables after their first release. ``CREATE TABLE IF
# NOT EXISTS`` will not add them to a database that already exists, so they are
# applied additively by ``migrate_db``.
COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "pipeline_runs": {
        "records_downloaded": "INTEGER",
        "records_stored": "INTEGER",
        "records_skipped": "INTEGER",
        "records_deduped": "INTEGER",
        # Free-form per-stage counters as JSON, so a run stays reconcilable
        # after the process that produced it is gone.
        "stage_counts": "TEXT",
    },
    "signal_status": {
        # Holdout cluster count at the last evaluation. The promotion ladder
        # advances a confirmation streak only when this grows -- i.e. when new
        # out-of-sample evidence actually arrived -- so re-running the gate on
        # unchanged data cannot manufacture a streak.
        "holdout_clusters": "INTEGER",
    },
    # The paper ledger's DDL was written before fractional shares were chosen
    # and before the account dataclasses existed, so a live table created from
    # it cannot name its instrument, hold a fractional quantity, or say why an
    # order was placed. These are additive, which is the one kind of schema
    # change migrate_db can apply to an existing table -- unlike a UNIQUE
    # constraint, which it cannot rewrite (see event_samples).
    "paper_orders": {
        "symbol": "TEXT",
        # Authoritative. ``quantity`` stays BIGINT and rounded; nothing reads it
        # back, so its loss of precision is an audit note, not a correctness bug.
        "quantity_exact": "DOUBLE",
        "reason": "TEXT",
    },
    "paper_fills": {
        "symbol": "TEXT",
        # The account rebuild folds THIS column. A BIGINT cannot carry 12.3456
        # shares, and a $10,000 account trading whole shares measures rounding
        # error rather than edge -- which is why fractional is the default mode.
        "quantity_exact": "DOUBLE",
    },
}


def migrate_db(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Additively add any missing columns. Idempotent; returns what it added."""
    applied: list[str] = []
    existing_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for table, columns in COLUMN_MIGRATIONS.items():
        if table not in existing_tables:
            continue
        present = {row[1] for row in con.execute(f'PRAGMA table_info("{table}")').fetchall()}
        for column, ddl in columns.items():
            if column not in present:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}')
                applied.append(f"{table}.{column}")
    return applied


def init_db(con: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    """Create all tables if they do not exist, then apply column migrations."""
    for statement in SCHEMA_STATEMENTS.values():
        con.execute(statement)
    migrate_db(con)
    return TABLES


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row count per table. Missing tables report -1."""
    counts: dict[str, int] = {}
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for table in TABLES:
        if table in existing:
            result = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            counts[table] = int(result[0]) if result else 0
        else:
            counts[table] = -1
    return counts
