"""Cross-sectional factor evaluation.

This package answers one question honestly: *is this signal real?*

It replaces the event-study statistic used previously. An event study averages
forward returns across events, so the ticker that emits the most events
dominates the result — measured at 76% for a single name in the insider corpus.
These modules rank the whole universe each month instead, so one company
contributes at most one name to one bucket in one month.

Nothing here knows what a factor *means*. Every module consumes a
``(date, cik) -> float`` panel and returns statistics. That isolation is what
lets ``tests/test_evaluation_canaries.py`` test the harness itself.
"""

from __future__ import annotations

from market_intelligence.evaluation.cpcv import cpcv_splits, n_backtest_paths
from market_intelligence.evaluation.deflated_sharpe import deflated_sharpe
from market_intelligence.evaluation.fama_macbeth import FamaMacBethResult, fama_macbeth
from market_intelligence.evaluation.portfolio_sort import SortResult, compound, monthly_spread
from market_intelligence.evaluation.report import FactorReport
from market_intelligence.evaluation.vault import Split, record_vault_read, split_of

__all__ = [
    "FactorReport",
    "FamaMacBethResult",
    "SortResult",
    "Split",
    "compound",
    "cpcv_splits",
    "deflated_sharpe",
    "fama_macbeth",
    "monthly_spread",
    "n_backtest_paths",
    "record_vault_read",
    "split_of",
]
