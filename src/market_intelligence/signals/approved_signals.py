"""The explicit, human-reviewed allowlist of cells permitted to fire a live alert.

The promotion ladder (``signal_status``, ``status='active'``) is a mechanical,
statistical test: discovery admission plus holdout confirmation against fixed
thresholds. That test is necessary but this module exists because it is not
sufficient on its own for a system whose output is meant to feed real trades
(and eventually an automated one): every cell that reaches production should
also have had a human look at the specific evidence, not just cleared a
threshold. ``autopilot.briefing.event_alerts`` checks both -- ladder ACTIVE
*and* :func:`is_approved` here -- before a cell can produce a live alert.

This is why it is a module-level constant, not a database table: adding a
cell here is a code change, reviewed the same way any other code change is.
The ladder itself cannot add to it, and neither can a scheduled run.

**Why this specific list, as of 2026-08-04.** Every ``insider_transaction``
cell was searched (owner role, stake-change size, dollar size, cluster
co-occurrence) and every 8-K item code. Several looked real inside an
in-discovery train/test split -- A, F, M -- and then reversed sign on the
real, previously-untouched 2023-2026 holdout once it was actually spent. That
reversal is exactly the failure mode this allowlist exists to catch even if
the mechanical ladder is ever fooled the same way again: only cells that
survived BOTH the in-discovery split AND the real holdout are here. Every
cross-company propagation edge (shared_board_member, competitor, supplier,
customer, partner) failed independently of that -- either the underlying
relationship graph has no usable dates (shared_board_member) or the
extraction pipeline only covers each company's single most recent filing
(supplier/customer/partner, all dated too late to be point-in-time valid
anywhere in the discovery period) -- so only ``edge_type='self'`` cells are
approved at all right now. Full evidence:
``quant-good-signals-2026-08-04.md`` in the user's memory.
"""

from __future__ import annotations

#: (event_type, event_subtype, edge_type, horizon_days) -- exactly how
#: signal_status keys a cell. A cell must match one of these AND be
#: status='active' in signal_status to ever produce a live alert.
APPROVED_CELLS: frozenset[tuple[str, str | None, str, int]] = frozenset(
    {
        ("insider_transaction", "P", "self", 60),
        ("insider_transaction", "C", "self", 40),
        ("insider_transaction", "C", "self", 60),
        ("insider_transaction", "C", "self", 90),
        ("insider_transaction", "C", "self", 120),
        ("insider_transaction", "D", "self", 20),
    }
)


def is_approved(
    event_type: str, event_subtype: str | None, edge_type: str, horizon_days: int
) -> bool:
    """Whether this exact cell may produce a live alert, independent of ladder status."""
    return (event_type, event_subtype, edge_type, horizon_days) in APPROVED_CELLS


__all__ = ["APPROVED_CELLS", "is_approved"]
