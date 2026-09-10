# SEC Continuous Downloader & Database Service

A bare-bones, dead-simple, single-purpose service for downloading financial data from **SEC EDGAR** and storing it in a local **DuckDB database** and raw disk archive.

---

## What This Service Does

1. **Syncs SEC Ticker Map**: Downloads `company_tickers.json` from SEC to map tickers to CIKs.
2. **Collects Submission History**: Fetches company filing histories (10-K, 10-Q, 8-K, S-1) and logs metadata in DuckDB.
3. **Downloads Primary Documents**: Preserves raw HTML/text filing documents in `data/raw/sec/` and logs document sizes and locations in DuckDB.
4. **Runs Continuously**: Can run continuously in the background, polling SEC for new filings and updating DuckDB automatically.

---

## File Structure

```text
quant/
├── .env                       # Local environment variables (SEC_USER_AGENT)
├── main.py                    # Main service entry point (status / sync / run)
├── pyproject.toml             # Python dependencies (httpx, duckdb, python-dotenv)
│
├── src/sec_service/           # SEC Downloader Core Package
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration & path management
│   ├── database.py            # DuckDB tables (companies, filings, filing_documents, sync_log)
│   ├── sec_client.py          # Rate-limited SEC EDGAR HTTP client (httpx)
│   └── downloader.py          # Downloader pipeline logic
│
├── data/                      # Preserved Local Data Archive
│   ├── database/
│   │   └── market_intelligence.duckdb  # Local DuckDB database file
│   └── raw/sec/               # Raw untouched filing documents
│
└── tests/
    └── test_sec_service.py    # Unit tests for database & client
```

---

## Commands

### 1. Check Database Status & Row Counts
```bash
python main.py status
```

### 2. Run a Single Download Pass
```bash
python main.py sync
```

### 3. Run Continuous Downloading Loop
```bash
python main.py run
```
*(Runs continuously in the background, syncing SEC data every 60 seconds).*

---

## Running Tests

```bash
pytest
```
