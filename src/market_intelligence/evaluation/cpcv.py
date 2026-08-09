"""Combinatorial Purged Cross-Validation (López de Prado, AFML ch. 12).

A single backtest is one path through history and reports one Sharpe, which
says nothing about how lucky that path was. CPCV withholds ``k`` of ``N``
contiguous groups at a time, giving ``C(N,k)`` splits and ``k/N · C(N,k)``
complete backtest paths — so the output is a *distribution* of Sharpes rather
than a single number that may or may not have been fortunate.

**Purging** drops training observations whose label window overlaps the test
window. **Embargo** drops a further span immediately after each test block, to
kill serial-correlation leakage across the boundary. With monthly rebalancing
and one-month labels this is nearly a no-op — but it becomes essential for the
longer holding periods this project has tested extensively (120 days), where
overlapping labels make train and test share information outright.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import comb

import numpy as np


def n_backtest_paths(*, n_groups: int, n_test_groups: int) -> int:
    """Number of complete backtest paths: ``k/N · C(N,k)``.

    N=6, k=2 gives 2/6 · 15 = 5 paths from 15 train/test splits.
    """
    return n_test_groups * comb(n_groups, n_test_groups) // n_groups


def cpcv_splits(
    *,
    n_obs: int,
    n_groups: int,
    n_test_groups: int,
    embargo: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for every combination of withheld groups."""
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be smaller than n_groups")

    groups = np.array_split(np.arange(n_obs), n_groups)

    for held in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in held])

        blocked = set(test_idx.tolist())
        for g in held:
            block_end = int(groups[g][-1])
            blocked.update(range(block_end + 1, min(block_end + 1 + embargo, n_obs)))

        train_idx = np.fromiter(
            (i for i in range(n_obs) if i not in blocked), dtype=int, count=-1
        )
        yield train_idx, test_idx
