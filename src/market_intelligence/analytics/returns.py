"""Daily return and abnormal-return computation. Pure and deterministic.

The output of this module is the label the entire signal layer is graded
against, so three properties matter more than anything else here:

**Adjusted prices only.** Returns are computed from ``adj_close``. A raw-close
series shows a 2-for-1 split as a -50% day, which would read as a catastrophic
event next to a filing that merely happened to be nearby.

**No lookahead.** The market-model ``beta`` for day *t* is fitted on a trailing
window that ends strictly before *t*. A beta fitted on a window containing *t*
leaks the very move being measured, which inflates backtest results in a way
that is invisible unless you go looking for it. ``estimation_window_start`` is
returned alongside each point so a stored row can be re-derived and audited.

**Gaps are not returns.** A price series with a hole in it (a halt, a missing
provider day, a delisting) would otherwise produce a single "daily" return
spanning weeks. Anything spanning more than ``max_gap_days`` is dropped rather
than reported as if it were a normal day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from itertools import pairwise

# A daily return may span a long weekend or a holiday closure. Beyond this many
# calendar days the observation is a gap, not a daily move.
DEFAULT_MAX_GAP_DAYS = 7

# Trailing window used to fit alpha/beta, and the minimum usable observations
# within it. One year of trading days is the conventional estimation window;
# below ~60 points a beta is too noisy to subtract meaningfully.
DEFAULT_ESTIMATION_WINDOW = 252
DEFAULT_MIN_OBSERVATIONS = 60

# Below this ratio of (residual variance / scale of x), a benchmark window is
# treated as constant. Real return windows sit near 1.0; floating-point residue
# from a genuinely flat series lands around 1e-32, so this sits far from both.
_DEGENERATE_VARIANCE_RATIO = 1e-12


class AbnormalReturnMethod(StrEnum):
    """How the "expected" part of a return is modelled before subtracting it.

    ``MARKET_MODEL``    -- r - (alpha + beta * r_benchmark). Accounts for the
                           fact that a high-beta name is *expected* to move more
                           on a big market day.
    ``MARKET_ADJUSTED`` -- r - r_benchmark. Assumes beta == 1. Cruder, but needs
                           no estimation window, so it works early in a series.
    ``SECTOR_ADJUSTED`` -- r - r_sector. Strips out sector-wide shocks, which
                           otherwise masquerade as propagation: if a chip tariff
                           moves every semiconductor name, that is not company A
                           transmitting information to company B.
    """

    MARKET_MODEL = "market_model"
    MARKET_ADJUSTED = "market_adjusted"
    SECTOR_ADJUSTED = "sector_adjusted"


@dataclass(frozen=True)
class ReturnPoint:
    """One symbol-day return, with the components that produced it."""

    symbol: str
    price_date: date
    total_return: float
    market_return: float | None = None
    sector_return: float | None = None
    abnormal_return: float | None = None
    beta: float | None = None
    alpha: float | None = None
    method: str | None = None
    estimation_window_start: date | None = None


@dataclass(frozen=True)
class OLSFit:
    """Result of fitting ``y = alpha + beta * x`` over a window."""

    alpha: float
    beta: float
    observations: int
    window_start: date | None = None


def simple_returns(
    bars: list[tuple[date, float]],
    *,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> list[tuple[date, float]]:
    """Convert an adjusted-close series into simple daily returns.

    ``bars`` is ``(date, adj_close)`` in any order; it is sorted here. Bars with
    a non-positive price are unusable (you cannot take a return against zero)
    and are dropped, as are returns spanning more than ``max_gap_days`` calendar
    days -- those are gaps in the series, not daily moves.

    The first bar produces no return, since a return needs a prior price.
    """
    usable = sorted((d, p) for d, p in bars if p is not None and p > 0)

    returns: list[tuple[date, float]] = []
    for (prev_date, prev_price), (curr_date, curr_price) in pairwise(usable):
        if (curr_date - prev_date).days > max_gap_days:
            continue
        returns.append((curr_date, curr_price / prev_price - 1.0))
    return returns


def fit_ols(points: list[tuple[float, float]]) -> OLSFit | None:
    """Fit ``y = alpha + beta * x`` by ordinary least squares.

    ``points`` is a list of ``(y, x)`` pairs. Returns ``None`` when the fit is
    undefined: fewer than two points, or an ``x`` series with no variance (a
    flat benchmark yields no information about beta, and dividing by its zero
    variance would produce a meaningless number rather than an error).
    """
    n = len(points)
    if n < 2:
        return None

    mean_y = sum(y for y, _ in points) / n
    mean_x = sum(x for _, x in points) / n

    covariance = sum((y - mean_y) * (x - mean_x) for y, x in points)
    variance = sum((x - mean_x) ** 2 for _, x in points)

    # A constant benchmark carries no information about beta. Testing
    # `variance == 0.0` is not sufficient: the mean of a perfectly constant
    # series is not exactly that constant in floating point, so subtracting it
    # leaves ~1e-34 of residue. Dividing a similarly tiny covariance by that
    # residue yields a large, entirely fabricated beta. The comparison must
    # therefore be relative to the scale of x, not against exact zero.
    scale = sum(x * x for _, x in points)
    if variance <= _DEGENERATE_VARIANCE_RATIO * scale:
        return None

    beta = covariance / variance
    return OLSFit(alpha=mean_y - beta * mean_x, beta=beta, observations=n)


def _as_map(series: list[tuple[date, float]] | None) -> dict[date, float]:
    return dict(series) if series else {}


def compute_abnormal_returns(
    symbol: str,
    asset_returns: list[tuple[date, float]],
    *,
    market_returns: list[tuple[date, float]] | None = None,
    sector_returns: list[tuple[date, float]] | None = None,
    method: AbnormalReturnMethod = AbnormalReturnMethod.MARKET_MODEL,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> list[ReturnPoint]:
    """Compute per-day abnormal returns for one symbol.

    For each day the expected return is modelled per ``method`` and subtracted
    from the realized return. Under ``MARKET_MODEL`` the alpha/beta for day *t*
    are fitted on the ``estimation_window`` observations immediately *before*
    *t* -- never including *t* itself.

    ``abnormal_return`` is ``None`` whenever the day's expectation cannot be
    modelled (no benchmark value for that day, or too few prior observations to
    fit). A ``None`` is deliberate and honest: a missing expectation is not the
    same as a zero abnormal return, and collapsing the two would let unmeasured
    days quietly count as "no surprise" in downstream statistics.
    """
    market_by_date = _as_map(market_returns)
    sector_by_date = _as_map(sector_returns)

    # The benchmark that defines "expected" depends on the method; sector and
    # market are both still *recorded* on every point regardless.
    benchmark_by_date = (
        sector_by_date if method is AbnormalReturnMethod.SECTOR_ADJUSTED else market_by_date
    )

    ordered = sorted(asset_returns)
    # History usable for fitting: days where both the asset and the benchmark
    # have a value. Built incrementally so day t only ever sees days before it.
    history: list[tuple[date, float, float]] = []

    points: list[ReturnPoint] = []
    for day, asset_return in ordered:
        benchmark_return = benchmark_by_date.get(day)

        abnormal: float | None = None
        fit: OLSFit | None = None

        if benchmark_return is not None:
            if method is AbnormalReturnMethod.MARKET_MODEL:
                window = history[-estimation_window:]
                if len(window) >= min_observations:
                    fit = fit_ols([(a, b) for _, a, b in window])
                    if fit is not None:
                        fit = OLSFit(
                            alpha=fit.alpha,
                            beta=fit.beta,
                            observations=fit.observations,
                            window_start=window[0][0],
                        )
                        abnormal = asset_return - (fit.alpha + fit.beta * benchmark_return)
            else:
                abnormal = asset_return - benchmark_return

        points.append(
            ReturnPoint(
                symbol=symbol,
                price_date=day,
                total_return=asset_return,
                market_return=market_by_date.get(day),
                sector_return=sector_by_date.get(day),
                abnormal_return=abnormal,
                beta=fit.beta if fit else None,
                alpha=fit.alpha if fit else None,
                method=method.value,
                estimation_window_start=fit.window_start if fit else None,
            )
        )

        # Append only after emitting, so day t is never part of its own window.
        if benchmark_return is not None:
            history.append((day, asset_return, benchmark_return))

    return points


def cumulative_abnormal_return(
    points: list[ReturnPoint],
    start: date,
    horizon_days: int,
) -> float | None:
    """Sum abnormal returns over the ``horizon_days`` trading days after ``start``.

    This is the forward label: "what happened to B in the N days after the event
    at A". Days at or before ``start`` are excluded, so an event never gets
    credit for a move that already happened.

    Returns ``None`` if any day in the window lacks an abnormal return, rather
    than summing a partial window -- a 5-day CAR built from 2 usable days is not
    a 5-day CAR, and treating it as one would understate the noise.
    """
    if horizon_days < 1:
        return None

    forward = [p for p in sorted(points, key=lambda p: p.price_date) if p.price_date > start]
    window = forward[:horizon_days]
    if len(window) < horizon_days:
        return None
    if any(p.abnormal_return is None for p in window):
        return None

    return sum(p.abnormal_return for p in window if p.abnormal_return is not None)


__all__ = [
    "DEFAULT_ESTIMATION_WINDOW",
    "DEFAULT_MAX_GAP_DAYS",
    "DEFAULT_MIN_OBSERVATIONS",
    "AbnormalReturnMethod",
    "OLSFit",
    "ReturnPoint",
    "compute_abnormal_returns",
    "cumulative_abnormal_return",
    "fit_ols",
    "simple_returns",
]
