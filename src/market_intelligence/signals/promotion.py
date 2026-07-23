"""The promotion ladder: how a signal earns, keeps, and loses the right to fire.

Re-running the validation gate on a schedule is repeated hypothesis testing --
run it often enough and noise clears any fixed threshold by luck. Two mechanisms
contain that, and this module is the second one.

The first lives in the gate itself: the discovery/holdout boundary is frozen, so
admission is decided once per cell against frozen data and new data only ever
grows the holdout. Re-running therefore cannot mint fresh admissions from a
growing pool; it can only accumulate out-of-sample confirmations.

The second is here. A cell must confirm on holdout for **K consecutive
evaluations** before it is trusted to fire alerts, and a single out-of-sample
sign reversal disqualifies it outright. A cell that flickers admitted-on-noise
never assembles a streak, so it never earns trust -- which is exactly the
failure a scheduled re-runner would otherwise walk into.

The transition is a pure function of ``(previous status, this run's verdict)``.
Keeping it free of I/O means the whole state machine is testable as a table of
cases; reading yesterday's status and writing today's are thin wrappers around
it, in :mod:`market_intelligence.signals.impact`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from market_intelligence.analytics.impact import Verdict


class LadderStatus(StrEnum):
    """What a cell is currently allowed to do.

    ``CANDIDATE`` cleared frozen discovery but has not confirmed K times yet;
    it does not fire. ``ACTIVE`` has confirmed K times running and is the only
    status that may fire an alert. ``DORMANT`` was active and then weakened
    below the economic floor without reversing -- it stops firing but is still
    watched, because a strong effect can dip and recover. ``RETIRED`` reversed
    sign out of sample, or failed too many times running; it is archived and
    must re-earn candidacy from scratch.
    """

    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    RETIRED = "retired"


@dataclass(frozen=True)
class SignalStatus:
    """A cell's position on the ladder: its status and the two streaks.

    ``confirm_streak`` counts consecutive confirmations (resets on any failure);
    ``fail_streak`` counts consecutive failures (resets on any confirmation).
    Only one is ever non-zero, but both are stored so a demotion knows how close
    the cell is to retirement.
    """

    status: LadderStatus
    confirm_streak: int
    fail_streak: int


#: Substring the gate writes into a DEMOTED reason when holdout flipped sign.
#: A reversal is disqualifying, so it is matched explicitly rather than folded
#: in with an ordinary below-floor collapse.
_REVERSAL_MARKER = "reversed sign"


def advance(
    previous: SignalStatus | None,
    verdict: Verdict,
    reason: str,
    *,
    promotion_streak: int,
    retire_after: int,
    new_evidence: bool = True,
) -> SignalStatus | None:
    """Compute a cell's new ladder position from its verdict on this run.

    Returns ``None`` when the cell should not be tracked at all -- a cell that
    has never cleared discovery (``REJECTED``/``INSUFFICIENT`` with no history)
    is not a candidate and gets no status row. Every other case returns the new
    :class:`SignalStatus`.

    The gate's four verdicts map onto ladder inputs as follows. ``ADMITTED`` is
    a confirmation (discovery cleared *and* holdout held). ``DEMOTED`` is a
    failure -- a reversal if the reason names one, otherwise a collapse.
    ``REJECTED`` means discovery itself failed; with a frozen boundary that is
    stable, so it only matters for a cell already being tracked, where it counts
    as a failure toward retirement. ``INSUFFICIENT`` means "could not look" --
    distinct from failure, so it holds the current status and moves no streak.

    ``new_evidence`` is the guard against the ladder counting *runs* instead of
    *evidence*. A confirmation only advances the streak when the holdout grew
    since the last evaluation; re-running the gate on unchanged data (a manual
    re-run, or a day when nothing new matured) is not a second independent
    confirmation and must not move a cell toward firing. When there is no new
    evidence, an already-tracked cell is held exactly as it was.
    """
    if not new_evidence and previous is not None:
        return previous

    if verdict is Verdict.ADMITTED:
        return _confirm(previous, promotion_streak)
    if verdict is Verdict.DEMOTED:
        return _fail(previous, reason, retire_after)
    if verdict is Verdict.REJECTED:
        # Failed discovery. Untracked cells stay untracked; a tracked cell that
        # now fails discovery is decaying and counts toward retirement.
        if previous is None:
            return None
        return _fail(previous, reason, retire_after)
    # INSUFFICIENT: not enough independent observations to judge either way.
    # Hold whatever we had; never promote or fail on absence of evidence.
    return previous


def _confirm(previous: SignalStatus | None, promotion_streak: int) -> SignalStatus:
    """Apply a confirmation: build the confirm streak, promote at K.

    A previously-active cell stays active. Any other cell (new, dormant, or
    re-earning after retirement) becomes active only once its streak reaches K;
    until then it is a candidate, so nothing fires on a single lucky run.
    """
    confirm_streak = (previous.confirm_streak if previous else 0) + 1
    already_active = previous is not None and previous.status is LadderStatus.ACTIVE

    if already_active or confirm_streak >= promotion_streak:
        status = LadderStatus.ACTIVE
    else:
        status = LadderStatus.CANDIDATE

    return SignalStatus(status=status, confirm_streak=confirm_streak, fail_streak=0)


def _fail(previous: SignalStatus | None, reason: str, retire_after: int) -> SignalStatus:
    """Apply a failure: break the confirm streak, demote or retire.

    A sign reversal retires immediately -- an effect that flips direction out of
    sample was never real. Otherwise the failure streak builds: an active cell
    goes dormant (watched, not fired), a dormant or long-failing cell retires
    once the streak reaches ``retire_after``, and a cell that never confirmed
    simply stays a candidate until it, too, exhausts its patience.
    """
    fail_streak = (previous.fail_streak if previous else 0) + 1
    prior_status = previous.status if previous else LadderStatus.CANDIDATE

    was_trusted = prior_status in (LadderStatus.ACTIVE, LadderStatus.DORMANT)
    if _REVERSAL_MARKER in reason or (was_trusted and fail_streak >= retire_after):
        # A sign flip, or a trusted cell out of patience: archived.
        status = LadderStatus.RETIRED
    elif was_trusted:
        # Active or already dormant, still within patience: hold in dormant,
        # watched but not firing.
        status = LadderStatus.DORMANT
    else:
        # A candidate that never earned trust; it simply stays a candidate.
        status = LadderStatus.CANDIDATE

    return SignalStatus(status=status, confirm_streak=0, fail_streak=fail_streak)


__all__ = [
    "LadderStatus",
    "SignalStatus",
    "advance",
]
