"""Simple configuration for SEC service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    sec_user_agent: str
    db_path: Path
    raw_dir: Path
    requests_per_second: float = 5.0

    @classmethod
    def load(cls, root: Path | None = None) -> Config:
        root_dir = root or PROJECT_ROOT
        load_dotenv(root_dir / ".env")

        user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not user_agent or "example.com" in user_agent:
            # Fallback user agent if not set
            user_agent = "SECService/1.0 (contact@company.org)"

        db_dir = root_dir / "data" / "database"
        db_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = root_dir / "data" / "raw" / "sec"
        raw_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            sec_user_agent=user_agent,
            db_path=db_dir / "market_intelligence.duckdb",
            raw_dir=raw_dir,
            requests_per_second=5.0,
        )
