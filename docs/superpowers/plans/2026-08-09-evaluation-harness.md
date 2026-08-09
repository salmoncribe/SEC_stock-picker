# Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-sectional factor evaluation harness that produces a monthly long-short spread and honest statistics for it, and prove the harness is trustworthy with two self-tests before any real factor touches it.

**Architecture:** A new `evaluation/` package with five independent modules — `portfolio_sort` (monthly rank → decile spread), `fama_macbeth` (Newey-West t-stat on monthly coefficients), `cpcv` (purged combinatorial CV → Sharpe distribution), `deflated_sharpe` (Bailey & López de Prado, N from pre-registration), and `vault` (CIK-hashed DEV/VAULT split with a git-tracked read log). Nothing here knows what a factor *means*; it consumes a `(date, cik) -> float` panel and returns statistics. That isolation is what lets the canaries test it.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, DuckDB, pytest. No new dependencies beyond what `pyproject.toml` already declares — verify before adding.

**Spec:** `docs/superpowers/specs/2026-08-09-sec-factor-signals-design.md` §6, §8
**Pre-registration:** `docs/preregistration/2026-08-09-factor-slate.md`

---

## Conventions (read before starting)

This repo has strong existing conventions. Follow them, not the root `CLAUDE.md`
defaults, where they differ:

- **Type annotations ARE used** here (`py.typed`, mypy configured), even though
  root `CLAUDE.md` says otherwise. Repo-local convention wins.
- `from __future__ import annotations` at the top of every module.
- Pure logic goes in `schemas/`-style modules with no I/O; I/O lives in
  `clients/`, orchestration in `collectors/`, writes in `storage/duckdb.py`.
- **All tests run offline.** No network, no touching `data/`. DB tests use an
  in-memory DuckDB connection; file tests use `tmp_path`. See `tests/conftest.py`.
- `ruff` line-length 100, target py312.
- Run tests with `.venv/bin/pytest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/market_intelligence/evaluation/__init__.py` | package marker, public exports |
| `src/market_intelligence/evaluation/vault.py` | CIK→DEV/VAULT split, read logging |
| `src/market_intelligence/evaluation/portfolio_sort.py` | monthly rank → bucket → long-short spread series |
| `src/market_intelligence/evaluation/fama_macbeth.py` | monthly cross-sectional regression, Newey-West t-stat |
| `src/market_intelligence/evaluation/cpcv.py` | purged + embargoed combinatorial CV paths |
| `src/market_intelligence/evaluation/deflated_sharpe.py` | DSR given SR, T, skew, kurtosis, N |
| `src/market_intelligence/evaluation/report.py` | assembles the five required statistics + survivorship label |
| `tests/test_evaluation_vault.py` | split determinism, CIK-keying, read log |
| `tests/test_evaluation_sort.py` | bucket rules, winsorization, compounding, delisting returns |
| `tests/test_evaluation_stats.py` | Fama-MacBeth, CPCV path count/purging, DSR against known values |
| `tests/test_evaluation_canaries.py` | **the two harness self-tests** |

---

## Task 1: Vault split

**Files:**
- Create: `src/market_intelligence/evaluation/__init__.py`
- Create: `src/market_intelligence/evaluation/vault.py`
- Test: `tests/test_evaluation_vault.py`

- [ ] **Step 1: Write the failing test**

```python
"""Vault split: deterministic, CIK-keyed, auditable."""

from __future__ import annotations

import pytest

from market_intelligence.evaluation.vault import Split, split_of


def test_split_is_deterministic():
    assert split_of("0000320193") == split_of("0000320193")


def test_split_is_keyed_on_cik_not_ticker():
    # A company that renames keeps its split. Passing a ticker is a bug;
    # CIKs are all-digit strings, so reject anything else loudly.
    with pytest.raises(ValueError, match="CIK"):
        split_of("AAPL")


def test_split_is_roughly_seventy_thirty():
    ciks = [str(i).zfill(10) for i in range(10_000)]
    dev = sum(1 for c in ciks if split_of(c) is Split.DEV)
    assert 6_700 <= dev <= 7_300, f"expected ~70% DEV, got {dev / 100:.1f}%"


def test_normalizes_cik_padding():
    # EDGAR writes CIKs both zero-padded and bare; they must not land in
    # different splits or the same company is in both.
    assert split_of("320193") == split_of("0000320193")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_vault.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'market_intelligence.evaluation'`

- [ ] **Step 3: Write minimal implementation**

```python
"""DEV/VAULT split for out-of-sample discipline.

The time-based holdout (2023-2026) is spent — it has been read repeatedly. This
module provides the replacement: a permanent, deterministic split of *companies*
so a genuinely unseen sample survives.

Keyed on CIK, never ticker: tickers are reused and reassigned, so a ticker-keyed
split would migrate a company across the boundary on a rename and leak VAULT
names into DEV.
"""

from __future__ import annotations

import enum
from hashlib import sha256

DEV_PERCENTILE = 70


class Split(enum.Enum):
    DEV = "dev"
    VAULT = "vault"


def normalize_cik(cik: str) -> str:
    """Zero-pad to EDGAR's 10-digit form so `320193` and `0000320193` agree."""
    if not cik.isdigit():
        raise ValueError(f"expected an all-digit CIK, got {cik!r}")
    return cik.zfill(10)


def split_of(cik: str) -> Split:
    """Assign a company to DEV or VAULT. Deterministic and permanent."""
    digest = sha256(normalize_cik(cik).encode()).hexdigest()
    return Split.DEV if int(digest, 16) % 100 < DEV_PERCENTILE else Split.VAULT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_vault.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/ tests/test_evaluation_vault.py
git commit -m "feat(evaluation): CIK-keyed DEV/VAULT split"
```

---

## Task 2: Vault read log

The split is worthless if it can be peeked at silently. Every VAULT read appends
to a git-tracked log.

**Files:**
- Modify: `src/market_intelligence/evaluation/vault.py`
- Test: `tests/test_evaluation_vault.py`

- [ ] **Step 1: Write the failing test**

```python
def test_vault_read_is_logged(tmp_path):
    from market_intelligence.evaluation.vault import record_vault_read

    log = tmp_path / "vault-reads.log"
    record_vault_read(log, factor="net_issuance", git_sha="abc123", note="round 1")
    record_vault_read(log, factor="pead", git_sha="abc123", note="round 1")

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "net_issuance" in lines[0]
    assert "abc123" in lines[0]


def test_vault_read_log_is_append_only(tmp_path):
    from market_intelligence.evaluation.vault import record_vault_read

    log = tmp_path / "vault-reads.log"
    record_vault_read(log, factor="a", git_sha="s", note="")
    first = log.read_text()
    record_vault_read(log, factor="b", git_sha="s", note="")
    # The earlier entry must still be present verbatim — a read can never be
    # erased, because erasing it is how a second peek gets hidden.
    assert log.read_text().startswith(first)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_vault.py -k vault_read -v`
Expected: FAIL — `ImportError: cannot import name 'record_vault_read'`

- [ ] **Step 3: Write minimal implementation**

Append to `vault.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

VAULT_READ_LOG = Path("docs/preregistration/vault-reads.log")


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_vault.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/vault.py tests/test_evaluation_vault.py
git commit -m "feat(evaluation): append-only vault read log"
```

---

## Task 3: Portfolio sort — bucketing and the spread series

**Files:**
- Create: `src/market_intelligence/evaluation/portfolio_sort.py`
- Test: `tests/test_evaluation_sort.py`

The three rules that must hold, each of which corresponds to a documented past
failure:

1. Returns **compound**, never sum (`2026-07-29-backtest-findings.md` Finding 2).
2. A bucket below `MIN_NAMES_PER_BUCKET` yields `NaN`, not a partial sort.
3. Bucket count is chosen by breadth and **recorded**, never assumed.

- [ ] **Step 1: Write the failing test**

```python
"""Portfolio sort: bucketing, compounding, breadth rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_intelligence.evaluation.portfolio_sort import (
    MIN_NAMES_PER_BUCKET,
    SortResult,
    compound,
    monthly_spread,
)


def test_returns_compound_not_sum():
    # +10% then -10% is -1%, not 0%. The old harness summed and reported 0%.
    assert compound(pd.Series([0.10, -0.10])) == pytest.approx(-0.01)


def test_spread_is_long_top_minus_short_bottom():
    # factor value == next-month return, so the spread must be strongly positive
    n = MIN_NAMES_PER_BUCKET * 10
    ciks = [str(i).zfill(10) for i in range(n)]
    factor = pd.Series(np.linspace(0, 1, n), index=ciks)
    fwd = factor.copy()

    result = monthly_spread(factor=factor, forward_returns=fwd, n_buckets=10)

    assert isinstance(result, SortResult)
    assert result.spread > 0.5
    assert result.n_buckets == 10


def test_thin_month_yields_nan_not_partial_sort():
    # 10 names cannot fill 10 buckets at 20/bucket. Must refuse, not average.
    ciks = [str(i).zfill(10) for i in range(10)]
    factor = pd.Series(np.arange(10.0), index=ciks)

    result = monthly_spread(factor=factor, forward_returns=factor, n_buckets=10)

    assert np.isnan(result.spread)
    assert result.reason == "insufficient_breadth"


def test_winsorization_caps_outliers():
    n = MIN_NAMES_PER_BUCKET * 10
    ciks = [str(i).zfill(10) for i in range(n)]
    factor = pd.Series(np.linspace(0, 1, n), index=ciks)
    factor.iloc[-1] = 1e9  # one absurd value

    result = monthly_spread(factor=factor, forward_returns=factor, n_buckets=10)

    # The outlier must not drag the long bucket's mean to ~1e9/20.
    assert result.long_return < 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_sort.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Cross-sectional portfolio sort.

This is the statistic the event-study harness should have been computing. An
event study averages forward returns across events, so the ticker that emits the
most events dominates the result — measured at 76% for TPL in the insider
corpus. A portfolio sort ranks the universe each month, so one company
contributes at most one name to one bucket in one month, and concentration
cannot inflate the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_NAMES_PER_BUCKET = 20
WINSOR_LIMITS = (0.01, 0.99)


@dataclass(frozen=True)
class SortResult:
    spread: float
    long_return: float
    short_return: float
    n_buckets: int
    n_names: int
    reason: str = ""


def compound(returns: pd.Series) -> float:
    """Geometric compounding. NEVER sum returns.

    `forward_abnormal_return` in the existing schema is a cumulative *sum* of
    daily returns. A real position compounds, and the gap is ~1/2 sigma^2 H —
    for 20 days at 5% daily vol that is ~2.5%, the same order as the entire
    claimed edge it was used to measure.
    """
    return float(np.prod(1.0 + returns.to_numpy()) - 1.0)


def _winsorize(values: pd.Series) -> pd.Series:
    low, high = values.quantile(WINSOR_LIMITS[0]), values.quantile(WINSOR_LIMITS[1])
    return values.clip(lower=low, upper=high)


def monthly_spread(
    *,
    factor: pd.Series,
    forward_returns: pd.Series,
    n_buckets: int,
) -> SortResult:
    """One month: rank by factor, bucket, return long-top-minus-short-bottom."""
    paired = pd.DataFrame({"factor": factor, "fwd": forward_returns}).dropna()
    n = len(paired)

    if n < MIN_NAMES_PER_BUCKET * n_buckets:
        return SortResult(np.nan, np.nan, np.nan, n_buckets, n, "insufficient_breadth")

    paired["factor"] = _winsorize(paired["factor"])
    ranks = paired["factor"].rank(method="first")
    buckets = pd.qcut(ranks, n_buckets, labels=False)

    long_return = float(paired.loc[buckets == n_buckets - 1, "fwd"].mean())
    short_return = float(paired.loc[buckets == 0, "fwd"].mean())
    return SortResult(long_return - short_return, long_return, short_return, n_buckets, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_sort.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/portfolio_sort.py tests/test_evaluation_sort.py
git commit -m "feat(evaluation): monthly cross-sectional portfolio sort"
```

---

## Task 4: Fama-MacBeth with Newey-West

**Files:**
- Create: `src/market_intelligence/evaluation/fama_macbeth.py`
- Test: `tests/test_evaluation_stats.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fama_macbeth_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    months = []
    for _ in range(120):
        x = pd.Series(rng.normal(size=200))
        y = 0.02 * x + pd.Series(rng.normal(scale=0.05, size=200))
        months.append((x, y))

    result = fama_macbeth(months, lags=3)

    assert result.mean_coefficient == pytest.approx(0.02, abs=0.005)
    assert result.t_stat > 3


def test_fama_macbeth_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    months = [
        (pd.Series(rng.normal(size=200)), pd.Series(rng.normal(size=200)))
        for _ in range(120)
    ]

    result = fama_macbeth(months, lags=3)

    assert abs(result.t_stat) < 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k fama -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Fama-MacBeth cross-sectional regression with Newey-West standard errors.

This replaces the per-event t-statistics used previously. On this corpus a
per-event t-stat is invalid: events cluster in a handful of tickers, so the
effective sample size is the number of *companies*, not the number of events.
That is how a t=8.85 result later collapsed inside a ticker-clustered CI
containing zero.

Here the regression runs cross-sectionally within each month, and the t-stat is
computed on the time series of monthly coefficients — so each month contributes
one observation regardless of how many events it contained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FamaMacBethResult:
    mean_coefficient: float
    t_stat: float
    n_months: int


def _newey_west_se(coefficients: np.ndarray, lags: int) -> float:
    demeaned = coefficients - coefficients.mean()
    n = len(demeaned)
    variance = float(demeaned @ demeaned / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(demeaned[lag:] @ demeaned[:-lag] / n)
        variance += 2.0 * weight * cov
    return float(np.sqrt(max(variance, 0.0) / n))


def fama_macbeth(
    monthly: list[tuple[pd.Series, pd.Series]],
    *,
    lags: int = 3,
) -> FamaMacBethResult:
    """`monthly` is one (factor, forward_return) pair per month."""
    coefficients = []
    for factor, forward in monthly:
        paired = pd.DataFrame({"x": factor, "y": forward}).dropna()
        if len(paired) < 2:
            continue
        design = np.column_stack([np.ones(len(paired)), paired["x"].to_numpy()])
        beta, *_ = np.linalg.lstsq(design, paired["y"].to_numpy(), rcond=None)
        coefficients.append(beta[1])

    values = np.asarray(coefficients, dtype=float)
    se = _newey_west_se(values, lags)
    t_stat = float(values.mean() / se) if se > 0 else 0.0
    return FamaMacBethResult(float(values.mean()), t_stat, len(values))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k fama -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/fama_macbeth.py tests/test_evaluation_stats.py
git commit -m "feat(evaluation): Fama-MacBeth with Newey-West errors"
```

---

## Task 5: Deflated Sharpe Ratio

**Files:**
- Create: `src/market_intelligence/evaluation/deflated_sharpe.py`
- Test: `tests/test_evaluation_stats.py`

- [ ] **Step 1: Write the failing test**

```python
def test_more_trials_lowers_the_deflated_sharpe():
    kwargs = dict(sharpe=1.0, n_obs=120, skew=0.0, kurtosis=3.0, sr_variance=0.25)
    one = deflated_sharpe(n_trials=1, **kwargs)
    fifty = deflated_sharpe(n_trials=50, **kwargs)
    assert one > fifty, "testing 50 strategies must raise the bar, not lower it"


def test_a_strong_single_trial_result_survives():
    assert deflated_sharpe(
        sharpe=2.0, n_obs=240, skew=0.0, kurtosis=3.0, sr_variance=0.1, n_trials=1
    ) > 0.95


def test_dsr_is_a_probability():
    value = deflated_sharpe(
        sharpe=0.5, n_obs=60, skew=-1.0, kurtosis=6.0, sr_variance=0.3, n_trials=20
    )
    assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k deflated -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Answers: given that N strategies were tried, what is the probability this
Sharpe is real? Testing ~50 variants at a 5% threshold manufactures ~2.5
"discoveries" by construction, which is roughly how many this project produced
before anyone counted the trials.

`n_trials` MUST come from docs/preregistration/, not from recollection.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(*, sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe across `n_trials` independent null strategies."""
    if n_trials <= 1:
        return 0.0
    gamma = EULER_MASCHERONI
    term = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    return float(np.sqrt(sr_variance) * term)


def deflated_sharpe(
    *,
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_variance: float,
    n_trials: int,
) -> float:
    """Probability the true Sharpe exceeds the trial-adjusted benchmark."""
    benchmark = expected_max_sharpe(sr_variance=sr_variance, n_trials=n_trials)
    denominator = np.sqrt(1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe**2)
    if denominator <= 0:
        return 0.0
    return float(norm.cdf((sharpe - benchmark) * np.sqrt(n_obs - 1) / denominator))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k deflated -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/deflated_sharpe.py tests/test_evaluation_stats.py
git commit -m "feat(evaluation): deflated Sharpe ratio"
```

---

## Task 6: Purged combinatorial cross-validation

**Files:**
- Create: `src/market_intelligence/evaluation/cpcv.py`
- Test: `tests/test_evaluation_stats.py`

- [ ] **Step 1: Write the failing test**

```python
def test_path_count_matches_the_formula():
    # phi = k/N * C(N,k); N=6, k=2 -> 2/6 * 15 = 5
    splits = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0))
    assert len(splits) == 15
    assert n_backtest_paths(n_groups=6, n_test_groups=2) == 5


def test_train_and_test_never_overlap():
    for train, test in cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0):
        assert not (set(train) & set(test))


def test_embargo_removes_observations_after_each_test_block():
    no_embargo = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=0))
    embargoed = list(cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5))
    # Embargo can only shrink the training set, never grow it.
    for (train_a, _), (train_b, _) in zip(no_embargo, embargoed, strict=True):
        assert len(train_b) <= len(train_a)
    assert sum(len(t) for t, _ in embargoed) < sum(len(t) for t, _ in no_embargo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k cpcv -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Combinatorial Purged Cross-Validation (Lopez de Prado, AFML ch. 12).

A single backtest is one path through history and reports one Sharpe, which
says nothing about how lucky that path was. CPCV withholds k of N contiguous
groups at a time, giving C(N,k) splits and k/N * C(N,k) complete backtest paths
— so the output is a *distribution* of Sharpes.

Purging drops training observations whose label window overlaps the test window;
the embargo drops a further span after each test block to kill serial-
correlation leakage across the boundary. With monthly rebalancing and one-month
labels this is nearly a no-op, but it becomes essential for the longer holding
periods this project has tested extensively (120 days).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import comb

import numpy as np


def n_backtest_paths(*, n_groups: int, n_test_groups: int) -> int:
    return n_test_groups * comb(n_groups, n_test_groups) // n_groups


def cpcv_splits(
    *,
    n_obs: int,
    n_groups: int,
    n_test_groups: int,
    embargo: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for every combination of withheld groups."""
    groups = np.array_split(np.arange(n_obs), n_groups)

    for held in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in held])
        blocked = set(test_idx.tolist())
        for g in held:
            end = int(groups[g][-1])
            blocked.update(range(end + 1, min(end + 1 + embargo, n_obs)))
        train_idx = np.array([i for i in range(n_obs) if i not in blocked])
        yield train_idx, test_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k cpcv -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/cpcv.py tests/test_evaluation_stats.py
git commit -m "feat(evaluation): purged combinatorial cross-validation"
```

---

## Task 7: THE CANARIES — the gate for everything downstream

**Files:**
- Create: `tests/test_evaluation_canaries.py`

These two tests are the highest-value work in this plan. Every other test checks
that a component does what it claims. These check that **the harness as a whole
cannot manufacture an edge that isn't there** — which is the exact failure that
consumed the previous three months.

**Do not proceed to `fundamentals/` or `factors/` until both pass.**

- [ ] **Step 1: Write the lookahead canary**

```python
"""Canaries: tests of the harness itself, not of any factor.

If these fail, every result the harness produces is void.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_intelligence.evaluation.fama_macbeth import fama_macbeth
from market_intelligence.evaluation.portfolio_sort import MIN_NAMES_PER_BUCKET, monthly_spread

N_NAMES = MIN_NAMES_PER_BUCKET * 10


def test_lookahead_canary_screams_when_given_the_future():
    """A factor built FROM future returns must post an enormous spread.

    This proves the harness can detect an edge at all. A harness that reports
    nothing here is broken in the direction that hides real signal; a harness
    that reports nothing here would also have reported nothing for a genuine
    factor, and we would never have known why.
    """
    rng = np.random.default_rng(0)
    ciks = [str(i).zfill(10) for i in range(N_NAMES)]
    forward = pd.Series(rng.normal(scale=0.08, size=N_NAMES), index=ciks)
    cheating_factor = forward.copy()  # perfect foreknowledge

    result = monthly_spread(factor=cheating_factor, forward_returns=forward, n_buckets=10)

    assert result.spread > 0.10, (
        "the harness failed to detect perfect foreknowledge — it cannot detect "
        "anything, and no result from it means anything"
    )


def test_lookahead_canary_goes_quiet_when_the_future_is_removed():
    """The same factor, shifted so it can no longer see its own month."""
    rng = np.random.default_rng(0)
    ciks = [str(i).zfill(10) for i in range(N_NAMES)]
    forward = pd.Series(rng.normal(scale=0.08, size=N_NAMES), index=ciks)
    lagged_factor = pd.Series(rng.normal(size=N_NAMES), index=ciks)  # independent

    result = monthly_spread(factor=lagged_factor, forward_returns=forward, n_buckets=10)

    assert abs(result.spread) < 0.05, (
        "the harness reported an edge from a factor independent of returns"
    )
```

- [ ] **Step 2: Write the null canary**

```python
def test_null_canary_t_stats_are_standard_normal():
    """100 random factors must produce t-stats distributed ~N(0,1).

    If random noise scores well here, the harness has a bug and would have
    scored the six real factors the same way. This is the test that would have
    caught the t=8.85 concentration result before it was believed.
    """
    rng = np.random.default_rng(42)
    t_stats = []

    for _ in range(100):
        months = [
            (
                pd.Series(rng.normal(size=200)),
                pd.Series(rng.normal(scale=0.05, size=200)),
            )
            for _ in range(120)
        ]
        t_stats.append(fama_macbeth(months, lags=3).t_stat)

    values = np.asarray(t_stats)

    assert abs(values.mean()) < 0.3, f"t-stats are biased: mean={values.mean():.3f}"
    assert 0.7 < values.std() < 1.4, f"t-stats are misscaled: sd={values.std():.3f}"

    false_positive_rate = float(np.mean(np.abs(values) > 1.96))
    assert false_positive_rate < 0.12, (
        f"{false_positive_rate:.0%} of PURE NOISE factors cleared p<0.05 "
        f"(expected ~5%). The harness manufactures edges."
    )
```

- [ ] **Step 3: Run the canaries**

Run: `.venv/bin/pytest tests/test_evaluation_canaries.py -v`
Expected: PASS (3 passed)

If the null canary fails, **stop**. Do not tune the threshold to make it pass.
Find the bug in the harness. A failing null canary means every downstream number
is unreliable, and adjusting the threshold only hides that.

- [ ] **Step 4: Commit**

```bash
git add tests/test_evaluation_canaries.py
git commit -m "test(evaluation): lookahead and null canaries — the harness gate"
```

---

## Task 8: Report assembly

**Files:**
- Create: `src/market_intelligence/evaluation/report.py`
- Test: `tests/test_evaluation_stats.py`

The pre-registration requires five statistics reported **together**; no subset
may be quoted alone. Enforce that in the type, so partial reporting is a
compile-time-ish error rather than a habit.

- [ ] **Step 1: Write the failing test**

```python
def test_report_requires_every_statistic():
    import dataclasses

    from market_intelligence.evaluation.report import FactorReport

    required = {
        "sharpe", "mean_monthly_spread", "fama_macbeth_t", "cpcv_sharpes",
        "deflated_sharpe", "survivorship", "n_trials", "bucket_counts",
    }
    fields = {f.name for f in dataclasses.fields(FactorReport)}
    assert required <= fields, f"missing: {required - fields}"


def test_promotion_requires_all_four_conditions():
    from market_intelligence.evaluation.report import FactorReport

    passing = FactorReport(
        factor="x", sharpe=1.2, mean_monthly_spread=0.01, fama_macbeth_t=3.1,
        cpcv_sharpes=[0.9, 1.1, 1.0, 1.3, 0.8], deflated_sharpe=0.97,
        survivorship="clean", n_trials=6, bucket_counts={10: 120},
        value_weighted_spread=0.008,
    )
    assert passing.promotes()

    # DSR below 0.95 alone must block promotion.
    assert not dataclasses.replace(passing, deflated_sharpe=0.80).promotes()
    # Sign disagreement between EW and VW must block promotion.
    assert not dataclasses.replace(passing, value_weighted_spread=-0.004).promotes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -k report -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `FactorReport` with a `promotes()` method** encoding the
four pre-registered conditions from `docs/preregistration/2026-08-09-factor-slate.md` §4:
DSR > 0.95, ≥4 of 5 CPCV paths positive, EW and VW spreads same sign, and
`survivorship` in `{clean, bounded}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_evaluation_stats.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/market_intelligence/evaluation/report.py tests/test_evaluation_stats.py
git commit -m "feat(evaluation): factor report with pre-registered promotion rule"
```

---

## Definition of done

- [ ] `.venv/bin/pytest tests/test_evaluation_*.py -v` — all pass
- [ ] `.venv/bin/ruff check src/market_intelligence/evaluation tests/test_evaluation_*.py` — clean
- [ ] **Both canaries pass** — this is the gate for `fundamentals/` and `factors/`
- [ ] `promotes()` encodes exactly the four conditions in the pre-registration, with no fifth condition invented during implementation

## Next plans (not this one)

2. `fundamentals/` — companyfacts.zip ingest, as-of accessor, split adjustment
3. `factors/` — the six pre-registered factors
4. `universe/` — filing-cessation delisting detection and the hole report
