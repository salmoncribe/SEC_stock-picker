"""Shared test fixtures.

All tests run offline. Any test that writes to disk uses ``tmp_path``; any DB
test uses an in-memory DuckDB connection. Nothing touches the project ``data/``
directory or the network.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from market_intelligence import database
from market_intelligence.config import Config, EnvSettings, load_config, reset_config_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real ``.env`` out of every test.

    The project's ``.env`` reaches config by two independent routes, and both
    have to be closed or a "credential is missing" test passes or fails
    depending on whose machine runs it:

    1. ``load_config`` calls ``load_dotenv``, which repopulates ``os.environ``
       and silently undoes ``monkeypatch.delenv``;
    2. ``EnvSettings`` declares ``env_file=".env"``, so pydantic-settings reads
       the file itself even when ``os.environ`` is clean.

    Tests that need a credential set it explicitly via ``monkeypatch.setenv``.
    """
    monkeypatch.setattr("market_intelligence.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setitem(EnvSettings.model_config, "env_file", None)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Config whose data/log home is a throwaway tmp dir, with test creds set."""
    monkeypatch.setenv("SEC_USER_AGENT", "TestSuite/0.1 (tests@market-intel.test)")
    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    reset_config_cache()
    config = load_config(project_root=PROJECT_ROOT, home=tmp_path)
    config.paths.ensure()
    return config


@pytest.fixture
def memory_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the schema initialized."""
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'")
    database.init_db(con)
    try:
        yield con
    finally:
        con.close()
