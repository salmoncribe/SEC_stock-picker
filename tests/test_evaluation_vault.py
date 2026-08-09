"""Vault split: deterministic, CIK-keyed, auditable.

The time-based holdout (2023-2026) is spent. This split is its replacement, so
its correctness is load-bearing for every out-of-sample claim that follows.
"""

from __future__ import annotations

import pytest

from market_intelligence.evaluation.vault import Split, record_vault_read, split_of


def test_split_is_deterministic() -> None:
    assert split_of("0000320193") == split_of("0000320193")


def test_split_is_keyed_on_cik_not_ticker() -> None:
    # A ticker-keyed split migrates a company across the boundary on a rename,
    # leaking VAULT names into DEV. CIKs are all digits; reject anything else.
    with pytest.raises(ValueError, match="CIK"):
        split_of("AAPL")


def test_split_is_roughly_seventy_thirty() -> None:
    ciks = [str(i).zfill(10) for i in range(10_000)]
    dev = sum(1 for cik in ciks if split_of(cik) is Split.DEV)
    assert 6_700 <= dev <= 7_300, f"expected ~70% DEV, got {dev / 100:.1f}%"


def test_normalizes_cik_padding() -> None:
    # EDGAR writes CIKs both zero-padded and bare. If those landed in different
    # splits the same company would sit on both sides of the holdout.
    assert split_of("320193") == split_of("0000320193")


def test_vault_read_is_logged(tmp_path) -> None:
    log = tmp_path / "vault-reads.log"
    record_vault_read(log, factor="net_issuance", git_sha="abc123", note="round 1")
    record_vault_read(log, factor="pead", git_sha="abc123", note="round 1")

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "net_issuance" in lines[0]
    assert "abc123" in lines[0]


def test_vault_read_log_is_append_only(tmp_path) -> None:
    log = tmp_path / "vault-reads.log"
    record_vault_read(log, factor="a", git_sha="s", note="")
    first = log.read_text()
    record_vault_read(log, factor="b", git_sha="s", note="")

    # The earlier entry must survive verbatim — erasing it is how a second
    # peek at the vault would get hidden.
    assert log.read_text().startswith(first)
