"""Per-ticker concentration: is a cell's edge a population, or one stock?

Every admission t-statistic in this codebase is computed **per event**, and
events are not independent -- the same ticker recurs dozens to hundreds of
times. Measured 2026-08-04, reweighting each ticker once instead of each event
once collapses every currently-admitted cell:

    P/60d   discovery   per-event t=8.01   ->  per-ticker t=3.09   (TPL  56.4%)
    P/60d   holdout     per-event t=2.84   ->  per-ticker t=0.53   (TPL 120.2%)
    C/120d  discovery   per-event t=11.44  ->  per-ticker t=0.26   (CRWD 37.4%)
    D/20d   discovery   per-event t=-2.50  ->  per-ticker t=-0.17

TPL contributing *more than 100%* of P/60d's holdout edge means the cell is
negative without that one name. C is 44 tickers, not 1,447 events.

The existing placebo control cannot catch this: ``impact._placebo_samples``
randomises the *control* ticker, so it measures whether the panel drifts up --
not whether the treatment group is one stock wearing a trenchcoat. The K=2
promotion ladder cannot catch it either, because a contaminated cell keeps
confirming on the same contaminating ticker.

Consumes ``impact._load_samples``' return shape directly: ``(symbol, t0,
market_adjusted_return)``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConcentrationReport:
    """One cell's independence profile, returned whether or not it passes.

    Kept as a real value rather than a bool so a rejected cell can be audited
    for *why* -- too few names, or one name dominating -- matching the
    convention in ``portfolio.stops.StopCheck``.
    """

    n_events: int
    n_tickers: int
    #: Mean of per-ticker mean returns. The honest central estimate.
    per_ticker_mean: float
    #: t-statistic of those per-ticker means. The honest significance.
    per_ticker_t: float
    #: Largest single ticker's share of the summed edge. May exceed 1.0 (or go
    #: negative) when the ticker's contribution outweighs the total -- that is
    #: not a bug, it means the cell is negative without that name.
    top_ticker: str | None
    top_ticker_share: float
    passed: bool
    reason: str


def per_ticker_stats(
    samples: Sequence[tuple[str, Any, float]],
) -> tuple[int, float, float, str | None, float]:
    """Collapse per-event samples to per-ticker means and score them.

    Returns ``(n_tickers, mean, t_stat, top_ticker, top_share)``. A cell with
    fewer than two tickers has no computable t-statistic and returns ``nan``
    rather than raising -- an unmeasurable cell must read as "cannot admit",
    never as "admitted by default".
    """
    by_ticker: dict[str, list[float]] = defaultdict(list)
    for symbol, _t0, ret in samples:
        by_ticker[symbol.upper()].append(ret)

    if not by_ticker:
        return 0, float("nan"), float("nan"), None, float("nan")

    means = {sym: sum(rs) / len(rs) for sym, rs in by_ticker.items()}
    n = len(means)

    total = sum(sum(rs) for rs in by_ticker.values())
    sums = {sym: sum(rs) for sym, rs in by_ticker.items()}
    top_ticker = max(sums, key=lambda s: sums[s])
    top_share = sums[top_ticker] / total if total else float("nan")

    values = list(means.values())
    mean = sum(values) / n
    if n < 2:
        return n, mean, float("nan"), top_ticker, top_share
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(var / n)
    t_stat = mean / stderr if stderr > 0 else float("nan")
    return n, mean, t_stat, top_ticker, top_share


def check_concentration(
    samples: Sequence[tuple[str, Any, float]],
    *,
    min_tickers: int,
    max_top_ticker_share: float,
    min_per_ticker_t: float,
) -> ConcentrationReport:
    """Decide whether a cell's edge is a population effect or a few names.

    TODO(michael): implement the admission policy below.

    You have the measured inputs from ``per_ticker_stats``: ``n_tickers``,
    ``per_ticker_mean``, ``per_ticker_t``, ``top_ticker``, ``top_share``.
    Decide when a cell is admissible, set ``passed``, and put a short
    human-readable explanation in ``reason`` (it lands in ``signal_status`` and
    is the only thing future-you will see explaining a rejection).

    The judgement calls, and why each is genuinely yours to make:

    * **How strict on `top_share`?** 10% is aggressive -- it would reject a
      cell where one name is merely the biggest of a healthy hundred. 25% is
      permissive enough that C/120d (CRWD at 37.4%) still fails but P/60d
      discovery (TPL at 56.4%) is the only /P casualty. Note the share can
      exceed 1.0 or go negative; decide whether those are automatic rejects.
    * **Is `min_tickers` a hard floor or a sliding scale?** C has 44 tickers
      with a per-ticker t near zero. A cell with 44 tickers and t=6 might be
      genuinely real and narrow (a sector effect). Does breadth substitute for
      significance, or is neither sufficient alone?
    * **Should a *negative* cell (like D/20d, a short signal) be judged on the
      same thresholds?** Its edge sums negative, which flips the sign of every
      share calculation.
    * **Does one failing criterion reject, or do you want a scoring rule?**
      A hard AND is simpler to reason about and harder to rationalise around
      later -- which matters, because the whole reason this file exists is that
      a softer check let three bad cells through.

    Be conservative: the cost of rejecting a real cell is delay, and the cost
    of admitting a fake one is real money on a signal that is one stock.
    """
    n_tickers, mean, t_stat, top_ticker, top_share = per_ticker_stats(samples)

    raise NotImplementedError(
        "admission policy not yet implemented -- see the TODO above"
    )


__all__ = ["ConcentrationReport", "check_concentration", "per_ticker_stats"]
