"""How many distinct insiders corroborate a signal on the same ticker.

Measured this session: a ticker where exactly 2 distinct insiders (by owner
CIK) each independently made a qualifying transaction -- a purchase, or a
10%-owner sale -- within a trailing window is at least as strong a signal as
one insider alone (120d edge 11.5%/21.2% search/confirm vs 7.9%/21.0% for
one). A third or more is NOT stronger: out-of-sample the edge for 3+
collapsed toward zero and failed year-robustness (confirm-window edge 2.1%
vs a search-window 9.2% that never repeated) -- this is exactly the failure
mode the old, blunt "2+ insiders buying" cluster detector had, just diluted
by lumping the good 2-person case in with the bad 3+ one. See
``signals.confidence`` for how the count turns into a score adjustment.

Only the same qualifying transaction TYPES corroborate each other -- this is
deliberately not "any Form 4 filing," since most insider_transaction codes
(grants, tax-withholding, mechanical option exercises) fire on a schedule for
every officer regardless of conviction, and counting those would just
reproduce the flat, undifferentiated result this session got from counting
any qualifying code together (see quant-good-signals-2026-08-04.md).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

#: Trailing window (calendar days) tonight's validation held roughly constant
#: across 10/20/40/60-day choices -- 20 is the one actually measured in the
#: headline numbers above.
CORROBORATION_WINDOW_DAYS = 20

#: (event_type, event_subtype) pairs that corroborate each other for a given
#: ticker. 'S' is intentionally split from the rest -- only a TenPercentOwner
#: sale carries the validated edge; a generic officer sale does not, and
#: mixing them back together would recreate the diluted result this session
#: specifically found and corrected.
_QUALIFYING_CODES = ("A", "F", "M", "P")


def count_corroborating_people(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str,
    t0: date,
    window_days: int = CORROBORATION_WINDOW_DAYS,
) -> int:
    """Distinct insiders (owner CIK) with a qualifying transaction on
    ``ticker`` in ``[t0 - window_days, t0]``, including whoever triggered
    this call. Always >= 1 when at least one qualifying event exists on the
    ticker at ``t0`` itself; a caller with no matching event should not call
    this and should leave the default of 1 (no known corroboration) in place.

    A plain 'S' (open-market sale) does not corroborate on its own -- only
    one filed by a 10%+ owner does, matched via ``events.payload`` through
    ``event_samples.event_id``. Everything else in ``_QUALIFYING_CODES``
    counts regardless of the filer's role.
    """
    codes_sql = ", ".join(f"'{code}'" for code in _QUALIFYING_CODES)
    row = con.execute(
        f"""
        SELECT count(DISTINCT json_extract_string(ev.payload, '$.owner_cik'))
        FROM event_samples es
        JOIN events ev ON ev.event_id = es.event_id
        WHERE upper(es.target_ticker) = upper(?)
          AND es.event_type = 'insider_transaction'
          AND es.t0 BETWEEN ? - CAST(? AS INTEGER) AND ?
          AND (
                es.event_subtype IN ({codes_sql})
             OR (es.event_subtype = 'S'
                 AND json_extract_string(ev.payload, '$.owner_relationship') = 'TenPercentOwner')
          )
          AND json_extract_string(ev.payload, '$.owner_cik') IS NOT NULL
        """,
        [ticker, t0, window_days, t0],
    ).fetchone()
    count = int(row[0]) if row and row[0] is not None else 0
    return max(count, 1)


__all__ = ["CORROBORATION_WINDOW_DAYS", "count_corroborating_people"]
