# market-intelligence

A **local** market-intelligence data platform for macOS. It downloads free,
public financial and economic data, validates and normalizes it, preserves the
untouched raw responses, and stores structured records in a local
[DuckDB](https://duckdb.org) database plus partitioned Parquet files.

---

## The decision rule (read this before any judgement call)

> **Make the decision that gives the best chance of making the most money.**
> Not the easiest, not the fastest, not the one that demos soonest.

This is the tiebreaker for every design choice in this repo. It is a hard rule,
not a preference. Where it conflicts with convenience, convenience loses.

It has one non-obvious consequence that matters more than the rule itself:

**The money-maximizing choice is almost always the more rigorous one, not the
more optimistic one.** A backtest is trivially easy to make look spectacular —
skip the point-in-time discipline, keep today's index members, let one
unadjusted reverse split into the return series, tune the thresholds until the
curve bends up. Every one of those makes the *reported* number bigger and the
*real* number smaller. A system that says "+40% annually" and delivers −5% is
worth far less than one that honestly says "no edge found here," because the
first one loses money with confidence.

So in practice this rule reads:

| Choice | Rule says |
|---|---|
| Full historical universe vs. today's survivors | **Full universe.** Survivorship bias inflates results and the inflation is invisible. |
| Ship the signal vs. validate it on held-out data first | **Validate.** An unvalidated edge is indistinguishable from noise, and you cannot tell which you have by looking. |
| Free data vs. paid data | **Whichever is more correct where correctness drives the answer.** Cheap infrastructure, yes. Cheap *labels*, no — the label is what everything is graded against. |
| More event types vs. fewer, better-measured ones | **Measured.** Untested breadth multiplies false positives; it does not multiply edge. |
| Fast collection run vs. slow, complete one | **Complete.** Compute time is cheap and one-off; a hole in the history is permanent and silent. |

The rule does **not** license spending money by default — the cheap-to-run
principle still governs infrastructure. It licenses spending *effort and time*
freely on anything that determines whether a signal is real, and spending money
only where a free source would corrupt the answer rather than merely slow it
down.

---

## Scope and honesty

This repo produces **model scores with measured track records**. It contains no
order execution, no broker integration, no position sizing, and no investment
advice. Every prediction it emits carries the historical hit rate of the exact
pattern that produced it, so it can be judged rather than trusted. What is done
with that output is the operator's decision and responsibility.

---

## What it does today

- **SEC EDGAR**
  - Downloads the company **ticker → CIK** map and stores the company universe.
  - Retrieves a company's **submission history** by CIK and normalizes filing
    metadata (10-K, 10-Q, 8-K, S-1, S-1/A, F-1, F-1/A, and more).
  - Filters filings by **form type**.
  - Saves the **untouched JSON** response to the raw store.
  - Writes normalized filing metadata to **DuckDB + Parquet**.
- **SEC filing documents** (`sec ingest-documents`)
  - Downloads the actual **primary document** for each 10-K / 10-Q and preserves
    the bytes **verbatim** in the raw store.
  - **Validates source integrity** by cross-checking the received byte count
    against the size declared in the filing's `index.json` manifest.
  - Splits each document into its official **Item sections** deterministically
    (no LLM, no heuristics that drift between runs) and writes each section's
    text to the normalized store.
  - Exposes clean **section records** — addressing, sizes, hashes, previews —
    ready for a later AI-extraction phase to consume.
- **FRED** (Federal Reserve Economic Data)
  - Downloads one or more configured economic **series** (metadata) and their
    **observations** (with realtime windows).
  - Handles FRED's `"."` missing-value marker explicitly (recorded, not dropped).
- **Index membership** (`market sync-constituents`)
  - Reconstructs **point-in-time** S&P 500 membership as `added_date` /
    `removed_date` windows, so a backtest sees the index as it was on each date
    rather than as it is today. Without this, every company dropped for
    collapsing is invisible and results are inflated by survivorship.
  - Handles tickers reused by unrelated companies, same-day company swaps under
    one symbol, and departures the source never logged. Windows whose dates are
    approximated are stored as `warning` with the reason, so a strict query can
    exclude them.
- **Market prices** (`market sync-prices`)
  - Daily OHLCV from a live provider (`yfinance`) behind the
    `MarketDataProvider` interface. **Resumable**: it fetches only what is newer
    than what is stored and flushes per symbol, so an interrupted backfill costs
    one symbol rather than all of them.
  - A dead symbol is a counted, named failure — never an empty series that
    reads as "did not trade".
- **Returns** (`market compute-returns`)
  - Derives daily and **abnormal** returns (market-model, market-adjusted, or
    sector-adjusted) from stored prices. This is the label every later signal is
    graded against.
  - Betas are fitted on a trailing window ending **strictly before** the day
    they price, and a test asserts it by perturbing the final observation and
    requiring every earlier point to be unchanged.
  - Price pairs spanning an unadjusted reverse split are dropped before fitting.
    Left in, one such artifact corrupts every beta for the following 252 days.
- **Cross-cutting**
  - Deterministic raw-file storage with content hashing and **atomic writes**.
  - **Idempotent** DuckDB + Parquet writes (re-running never duplicates rows).
  - Per-record **validation** (`valid` / `warning` / `rejected`) with provenance.
  - Structured **JSON logging** and **pipeline-run** bookkeeping.
  - A **Typer** CLI with useful output and proper exit codes.

## What it intentionally does NOT do yet

- No trading logic, order execution, or position sizing.
- No AI/LLM extraction, Obsidian notes, or relationship graphs yet — these are
  Stage 1 of
  [the signal-graph design](docs/specs/2026-07-22-signal-graph-design.md).
  Section records are *prepared* for extraction; nothing interprets their
  meaning yet.
- **No validated signal.** The price spine exists and is measurable, but no
  event has been shown to predict anything. Per the decision rule above, a
  measured "no edge here" is a legitimate and useful outcome of Stage 3.
- **Deterministic sectioning only for 10-K / 10-Q.** These have a stable,
  official Item taxonomy. 8-K and S-1 documents are not sectioned, because a
  reliable split for them needs judgement this phase deliberately avoids.
- **The price source is free, and that is a known limitation.** `yfinance` is an
  unofficial endpoint that can change without notice, and it will not serve
  many delisted symbols: of 874 tickers that were ever S&P 500 members, 210
  returned no data. Survivorship bias is therefore fixed in the *universe* but
  partly returns at the *price* layer. Isolated behind `MarketDataProvider`, so
  a licensed EOD feed is a one-file swap — and per the decision rule, that swap
  is worth paying for once a signal is real enough to trade.
- **No global "all recent IPOs" endpoint.** The SEC does not expose one in the
  form this baseline needs. `collect-ipos` scans the *configured companies'*
  submissions for S-1/F-1-family forms. The documented extension point for a true
  IPO feed is the SEC EDGAR **full-text search API**
  (`https://efts.sec.gov/LATEST/search-index?...`), left for a later phase.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python **3.12+** (the project is developed on 3.14; `uv` will manage the
  interpreter for you)
- [`uv`](https://docs.astral.sh/uv/) for packaging and virtual environments

### Install `uv` on macOS

```bash
# Homebrew (recommended on this machine)
brew install uv

# …or the official standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

## Setup

### 1. Initialize the project (create the venv + install deps)

From the project root:

```bash
uv sync --extra dev
```

This creates `.venv/`, installs runtime and dev dependencies, and installs the
`market-intelligence` command into the environment. Run everything through
`uv run` (no manual `activate` needed).

### 2. Create your `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
SEC_USER_AGENT=MarketIntelligence/0.1 (you@yourdomain.com)
FRED_API_KEY=your_fred_key_here
MARKET_INTELLIGENCE_HOME=
LOG_LEVEL=INFO
```

`.env` is git-ignored. Never commit real secrets.

### 3. Configure the SEC User-Agent

SEC EDGAR **requires** a descriptive `User-Agent` that identifies your
application and a **real contact email** (see the
[SEC developer FAQ](https://www.sec.gov/os/webmaster-faq#developers)). Set
`SEC_USER_AGENT` in `.env` using the format:

```
AppName/version (contact@example.com)
```

The app **fails clearly** if `SEC_USER_AGENT` is unset or still uses the example
`you@example.com` placeholder — it will not send SEC requests without a real one.
Requests use polite pacing and exponential-backoff retries; nothing here attempts
to bypass rate limits.

### 4. Obtain a FRED API key

1. Create a free account at <https://fredaccount.stlouisfed.org/login/secure/>.
2. Request an API key at
   <https://fred.stlouisfed.org/docs/api/api_key.html>.
3. Put it in `.env` as `FRED_API_KEY`.

### 5. Initialize DuckDB

```bash
uv run market-intelligence init-db
```

Creates `data/database/market_intelligence.duckdb` and all tables. Idempotent —
safe to run again.

---

## Commands

Run `uv run market-intelligence --help` for the full list. Every command returns
a proper exit code: **0** success, **1** pipeline/runtime failure, **2**
configuration error (e.g. missing credentials).

```bash
# One-time database setup
uv run market-intelligence init-db

# SEC: download the ticker->CIK map and store the company universe
uv run market-intelligence sec sync-companies

# SEC: collect filings for a ticker (all supported forms)
uv run market-intelligence sec collect-filings --ticker NVDA

# SEC: collect only specific forms
uv run market-intelligence sec collect-filings --ticker NVDA --forms 10-K,10-Q,8-K

# SEC: scan configured companies for IPO / registration filings
uv run market-intelligence sec collect-ipos --forms S-1,S-1/A,F-1,F-1/A

# SEC: download the actual 10-K/10-Q documents and extract their Item sections.
# Operates on filings already collected above. Safe to re-run.
uv run market-intelligence sec ingest-documents
uv run market-intelligence sec ingest-documents --ticker NVDA --limit 5

# FRED: download the configured economic series (config/fred_series.yaml)
uv run market-intelligence fred sync
# …or specific series
uv run market-intelligence fred sync --series DGS10,UNRATE

# Audit stored records: validation-status counts + anomalies
uv run market-intelligence validate

# Reconcile pipeline counts and explain any differences
uv run market-intelligence reconcile
uv run market-intelligence reconcile --no-verify-hashes   # skip re-hashing text files

# Snapshot: config readiness, table row counts, recent pipeline runs
uv run market-intelligence status
```

### Configuration files (`config/`)

| File | Purpose |
|------|---------|
| `settings.yaml`     | HTTP timeouts, retry behavior, per-source request pacing, SEC/FRED endpoints |
| `companies.yaml`    | Configured companies (ticker, optional CIK, name) |
| `fred_series.yaml`  | FRED series to collect + default observation window |
| `forms.yaml`        | Supported SEC forms (validation allow-list) and the IPO form subset |

Configuration is loaded and validated with Pydantic; malformed config fails
fast with a clear message.

---

## Where data is stored

All paths are under `MARKET_INTELLIGENCE_HOME` (defaults to the project root).

```
data/
├── raw/            # untouched API responses, never modified
│   ├── sec/<data_type>/<identifier>/<YYYY-MM-DD>/<hash>.json
│   ├── sec/filing_document/<accession>/<YYYY-MM-DD>/<hash>.htm
│   ├── fred/<data_type>/<series_id>/<YYYY-MM-DD>/<hash>.json
│   └── market/
├── normalized/     # derived, cleaned renderings of source documents
│   └── sections/<cik>/<accession>/<item_code>.txt
├── parquet/        # durable, partitioned structured datasets
│   ├── companies/…
│   ├── filings/…
│   ├── filing_documents/…
│   ├── filing_sections/…
│   ├── economic_series/…
│   └── economic_observations/series_id=<id>/data.parquet
└── database/
    └── market_intelligence.duckdb   # analytical tables
logs/
└── market_intelligence.log          # structured JSON logs
```

### Database tables

| Table | Natural key (unique) | Notes |
|-------|----------------------|-------|
| `companies`             | `cik` | `first_seen_time` is set once; `last_seen_time` updates |
| `filings`               | `accession_number` | links to a company via `company_id` |
| `filing_documents`      | `(accession_number, document_name)` | one downloaded document; carries `sha256`, `declared_size`, `integrity_status` |
| `filing_sections`       | `(accession_number, item_code)` | one extracted Item section; text lives on disk at `text_path`, verified by `text_sha256` |
| `economic_series`       | `series_id` | series metadata |
| `economic_observations` | `(series_id, observation_date, realtime_start, realtime_end)` | one row per observation/realtime window |
| `pipeline_runs`         | `run_id` | per-run counts, status, and a config hash |

Every normalized record carries provenance fields: `source`, `source_url`,
`source_record_id`, `event_time`, `published_time`, `collected_time`,
`content_hash`, `schema_version`, `validation_status`, `validation_errors`. All
timestamps are stored in **UTC**.

---

## How idempotency & validation work

**Idempotency** (re-running a command never duplicates data) comes from two
cooperating mechanisms:

- **Raw store** — the SHA-256 content hash is embedded in the filename. Identical
  bytes resolve to the same path and are skipped; different bytes get a different
  filename, so a re-download never silently overwrites a prior raw response.
  Writes are atomic (temp file + `os.replace`).
- **Analytical store** — before writing, the DuckDB layer reads the existing
  natural keys for the target table and splits the batch into *inserts* vs.
  *updates*. Re-running reports `0 inserted, N updated` and the row count stays
  constant. Record ids are deterministic hashes of the natural key. Parquet
  writes merge-and-dedupe on the same key.

**Validation** classifies each record as `valid`, `warning`, or `rejected` and
records the reasons on the record itself (never silently discarded):

- missing required fields, malformed CIKs, malformed accession numbers,
  unsupported forms, invalid/missing dates, missing source URLs, empty API
  responses, unexpected value types, and FRED `"."` missing-value markers.

Rejected records are kept out of the analytical tables but the failure is logged
and counted in the pipeline run. The `validate` command aggregates stored
statuses and flags known anomalies.

---

## Reading the pipeline counts

Each counter answers a different question, so they are tracked separately rather
than derived from one another. `reconcile` asserts the identities below and
prints the residual whenever one fails.

| Counter | Meaning |
|---------|---------|
| `candidates` | Filings selected for ingestion |
| `downloaded` | Payloads successfully fetched over the network |
| `stored` | **New** raw files written to disk (first sighting of these bytes) |
| `skipped` | Fetched bytes already byte-identical in the raw store — the idempotency signal, **not** an error |
| `collected` | Records built and offered to storage |
| `inserted` / `updated` | DuckDB rows created vs. modified |
| `deduped` | Rows collapsed within one batch that shared a natural key |
| `rejected` | Failed a hard invariant; kept for audit, not persisted |

```
candidates          = downloaded + fetch_failed + skipped_non_textual
downloaded          = stored + skipped
downloaded          = collected + rejected
collected           = inserted + updated + deduped
sections_extracted  = sections_offered + sections_rejected
sections_offered    = sections_inserted + sections_updated + sections_deduped
```

Why `stored` and `skipped` diverge from `downloaded` on a second run: the raw
store is content-addressed, so re-fetching an unchanged document resolves to the
existing file and writes nothing. A second identical run therefore reports
`stored=0, skipped=N` while still reporting `downloaded=N` — the network work
happened, the disk work did not.

### Document integrity

`filing_documents.integrity_status` records how well a download is corroborated:

| Status | Meaning | Outcome |
|--------|---------|---------|
| `verified` | Listed in `index.json` and byte count matches the declared size | stored |
| `size_mismatch` | Listed, but byte counts disagree | **rejected** |
| `not_in_index` | Manifest readable, document absent from it | **rejected** |
| `unverified` | Manifest could not be fetched — absence of evidence, not evidence of corruption | stored with a warning |

---

## Development: tests, lint, format, types

```bash
# Run the test suite (offline — no network access required)
uv run pytest

# Lint
uv run ruff check .

# Format (check-only, or apply)
uv run ruff format --check .
uv run ruff format .

# Type-check
uv run mypy src/market_intelligence
```

Tests use saved fixtures and mocked HTTP (`httpx.MockTransport`); none of them
touch the live internet or your real `data/` directory.

---

## Future architecture

The baseline is structured so these can be added without rework:

- **XBRL company facts** — `SECClient.fetch_company_facts` is a stubbed
  extension point for structured financial statements.
- **LLM structured fact extraction** — runs over the `filing_sections` records
  this phase produces (each row addresses one Item's text plus its hash), and
  writes new normalized record types under the same provenance/validation
  contract. Route cheap passes to a local model; reserve paid APIs for hard
  reasoning.
- **Obsidian Markdown notes** — a renderer that turns normalized records into
  vault notes.
- **Company relationship graphs** — supplier/customer edges as a new table +
  schema, populated from extracted facts.
- **Earnings ingestion & daily market features** — new collectors following the
  `clients → collectors → validators → storage` pattern.
- **Market prices** — implement `MarketDataProvider` against a licensed EOD API
  and register it.
- **Signal scoring, backtesting, model training** — downstream consumers that
  read DuckDB/Parquet; **not** part of this phase.
- **Scheduled daily runs** — wrap the existing CLI commands in a scheduler
  (launchd/cron); each command already records a `pipeline_runs` row.

---

## Project layout

```
market-intelligence/
├── config/                     # YAML configuration
├── data/                       # raw / normalized / parquet / database (git-ignored contents)
├── logs/                       # structured JSON logs
├── src/market_intelligence/
│   ├── cli.py                  # Typer CLI
│   ├── config.py               # typed .env + YAML config
│   ├── logging_config.py       # structlog JSON logging
│   ├── database.py             # DuckDB connection + schema DDL
│   ├── hashing.py              # content hashing / record ids
│   ├── reconciliation.py       # count identities + storage agreement checks
│   ├── schemas/                # Pydantic records (common, sec, fred)
│   ├── clients/                # base HTTP client + sec, fred, market
│   ├── collectors/             # orchestration + pipeline_run bookkeeping
│   ├── extraction/             # deterministic 10-K/10-Q Item sectioning
│   ├── validators/             # sec, fred validation rules
│   └── storage/                # raw, normalized, parquet, duckdb writers
└── tests/                      # offline tests + fixtures
```

## Safety & scope

This project ingests and stores public data only. It provides **no** investment
advice, trade execution, or price predictions. It is a data foundation.
