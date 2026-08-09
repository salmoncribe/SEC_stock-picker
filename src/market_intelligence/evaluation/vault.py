"""DEV/VAULT split for out-of-sample discipline.

The time-based holdout (2023-2026) is spent — it has been read repeatedly. This
module is its replacement: a permanent, deterministic split of *companies*, so a
genuinely unseen sample survives for one final read.

Keyed on CIK, never ticker. Tickers are reused and reassigned, so a ticker-keyed
split would migrate a company across the boundary on a rename and quietly leak
VAULT names into DEV.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

DEV_PERCENTILE = 70
VAULT_READ_LOG = Path("docs/preregistration/vault-reads.log")


class Split(enum.Enum):
    DEV = "dev"
    VAULT = "vault"


def normalize_cik(cik: str) -> str:
    """Zero-pad to EDGAR's 10-digit form so ``320193`` and ``0000320193`` agree.

    Without this the same company lands in different splits depending on which
    source spelled its CIK, and would sit on both sides of the holdout.
    """
    if not cik.isdigit():
        raise ValueError(f"expected an all-digit CIK, got {cik!r}")
    return cik.zfill(10)


def split_of(cik: str) -> Split:
    """Assign a company to DEV or VAULT. Deterministic and permanent."""
    digest = sha256(normalize_cik(cik).encode()).hexdigest()
    return Split.DEV if int(digest, 16) % 100 < DEV_PERCENTILE else Split.VAULT


def record_vault_read(log_path: Path, *, factor: str, git_sha: str, note: str) -> None:
    """Append a VAULT read to the audit log. Never rewrites existing entries.

    The log is git-tracked, so a second read of the vault cannot happen without
    a visible commit. This makes the mistake structurally impossible rather than
    merely discouraged — the same principle as the existing sealed-clock guard.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{factor}\t{git_sha}\t{note}\n")
