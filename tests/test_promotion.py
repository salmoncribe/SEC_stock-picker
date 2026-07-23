"""The promotion ladder as a transition table.

The whole point of the ladder is that noise cannot earn trust by being tested
repeatedly. These tests pin every transition, and two of them are the load-
bearing guarantees: a signal needs K *consecutive* confirmations to fire, and a
single out-of-sample sign reversal disqualifies it no matter how strong it was.
"""

from __future__ import annotations

from market_intelligence.analytics.impact import Verdict
from market_intelligence.signals.promotion import (
    LadderStatus,
    SignalStatus,
    advance,
)

K = 2  # promotion_streak used throughout
RETIRE = 3  # retire_after used throughout


def step(previous: SignalStatus | None, verdict: Verdict, reason: str = "") -> SignalStatus | None:
    return advance(previous, verdict, reason, promotion_streak=K, retire_after=RETIRE)


ADMIT = Verdict.ADMITTED
DEMOTE = Verdict.DEMOTED
REJECT = Verdict.REJECTED
INSUFF = Verdict.INSUFFICIENT
REVERSAL = "holdout reversed sign: discovery +0.90%, holdout -0.40%"
COLLAPSE = "holdout mean CAR +0.01% collapsed below the economic floor"


# --------------------------------------------------------------------------- #
# the two guarantees                                                          #
# --------------------------------------------------------------------------- #
def test_one_confirmation_is_not_enough_to_fire():
    """A single admitted run must not activate; that is the anti-noise rule."""
    after_one = step(None, ADMIT)

    assert after_one is not None
    assert after_one.status is LadderStatus.CANDIDATE
    assert after_one.confirm_streak == 1


def test_k_consecutive_confirmations_activate():
    status = None
    for _ in range(K):
        status = step(status, ADMIT)

    assert status is not None
    assert status.status is LadderStatus.ACTIVE
    assert status.confirm_streak == K


def test_a_broken_streak_must_start_over():
    """Confirm, fail, confirm is not two confirmations -- it is one."""
    s = step(None, ADMIT)  # candidate, streak 1
    s = step(s, DEMOTE, COLLAPSE)  # streak reset
    s = step(s, ADMIT)  # streak 1 again, still candidate

    assert s is not None
    assert s.status is LadderStatus.CANDIDATE
    assert s.confirm_streak == 1


def test_sign_reversal_retires_even_an_active_signal():
    active = SignalStatus(LadderStatus.ACTIVE, confirm_streak=5, fail_streak=0)

    after = step(active, DEMOTE, REVERSAL)

    assert after is not None
    assert after.status is LadderStatus.RETIRED


# --------------------------------------------------------------------------- #
# active -> dormant -> recover / retire                                       #
# --------------------------------------------------------------------------- #
def test_active_collapse_goes_dormant_not_retired():
    """A dip below the floor without a reversal is recoverable, so: dormant."""
    active = SignalStatus(LadderStatus.ACTIVE, confirm_streak=4, fail_streak=0)

    after = step(active, DEMOTE, COLLAPSE)

    assert after is not None
    assert after.status is LadderStatus.DORMANT
    assert after.fail_streak == 1


def test_dormant_retires_after_enough_consecutive_failures():
    s: SignalStatus | None = SignalStatus(LadderStatus.ACTIVE, confirm_streak=4, fail_streak=0)
    s = step(s, DEMOTE, COLLAPSE)  # dormant, fail 1
    assert s is not None and s.status is LadderStatus.DORMANT
    s = step(s, DEMOTE, COLLAPSE)  # dormant, fail 2
    assert s is not None and s.status is LadderStatus.DORMANT
    s = step(s, DEMOTE, COLLAPSE)  # fail 3 == retire_after -> retired

    assert s is not None
    assert s.status is LadderStatus.RETIRED
    assert s.fail_streak == RETIRE


def test_dormant_recovers_but_must_re_earn_activation():
    """A dormant cell that confirms again does not fire on the first re-confirm."""
    dormant = SignalStatus(LadderStatus.DORMANT, confirm_streak=0, fail_streak=1)

    once = step(dormant, ADMIT)
    assert once is not None
    assert once.status is LadderStatus.CANDIDATE  # re-earning, not yet firing

    twice = step(once, ADMIT)
    assert twice is not None
    assert twice.status is LadderStatus.ACTIVE


# --------------------------------------------------------------------------- #
# tracking boundaries                                                         #
# --------------------------------------------------------------------------- #
def test_never_confirmed_cell_is_untracked_on_rejection():
    """A cell that failed discovery and has no history is not a candidate."""
    assert step(None, REJECT) is None


def test_insufficient_with_no_history_stays_untracked():
    assert step(None, INSUFF) is None


def test_insufficient_holds_an_existing_status_unchanged():
    """'Could not look' must not move a streak in either direction."""
    active = SignalStatus(LadderStatus.ACTIVE, confirm_streak=3, fail_streak=0)

    after = step(active, INSUFF)

    assert after == active


def test_rejection_of_a_tracked_cell_counts_toward_retirement():
    """If a tracked cell later fails discovery, it decays rather than vanishing."""
    s: SignalStatus | None = SignalStatus(LadderStatus.ACTIVE, confirm_streak=4, fail_streak=0)
    s = step(s, REJECT)  # dormant, fail 1
    assert s is not None and s.status is LadderStatus.DORMANT
    s = step(s, REJECT)  # fail 2
    s = step(s, REJECT)  # fail 3 -> retired

    assert s is not None
    assert s.status is LadderStatus.RETIRED


def test_retired_cell_re_earns_candidacy_from_scratch():
    retired = SignalStatus(LadderStatus.RETIRED, confirm_streak=0, fail_streak=5)

    once = step(retired, ADMIT)
    assert once is not None
    assert once.status is LadderStatus.CANDIDATE
    assert once.confirm_streak == 1  # streak restarts, no credit for the past


def test_no_new_evidence_holds_a_cell_unchanged():
    """Re-running on the same holdout must not advance a confirmation streak.

    The ladder counts fresh out-of-sample evidence, not runs. Without this a
    manual re-run -- or a day when nothing new matured -- would fabricate a
    streak and promote on stale data, the exact self-deception the ladder
    exists to prevent.
    """
    candidate = SignalStatus(LadderStatus.CANDIDATE, confirm_streak=1, fail_streak=0)

    held = advance(
        candidate, ADMIT, "", promotion_streak=K, retire_after=RETIRE, new_evidence=False
    )

    assert held == candidate  # streak did not advance, so no promotion


def test_new_evidence_lets_a_confirmation_promote():
    candidate = SignalStatus(LadderStatus.CANDIDATE, confirm_streak=1, fail_streak=0)

    promoted = advance(
        candidate, ADMIT, "", promotion_streak=K, retire_after=RETIRE, new_evidence=True
    )

    assert promoted is not None
    assert promoted.status is LadderStatus.ACTIVE


def test_first_sighting_counts_even_as_new_evidence_default():
    """A never-seen cell (previous is None) is always fresh evidence."""
    first = advance(None, ADMIT, "", promotion_streak=K, retire_after=RETIRE, new_evidence=False)

    assert first is not None
    assert first.confirm_streak == 1


def test_streaks_are_mutually_exclusive():
    """A confirmation zeroes the fail streak and vice versa."""
    s = step(None, ADMIT)
    assert s is not None and s.fail_streak == 0 and s.confirm_streak == 1

    s = step(s, DEMOTE, COLLAPSE)
    assert s is not None and s.confirm_streak == 0 and s.fail_streak == 1
