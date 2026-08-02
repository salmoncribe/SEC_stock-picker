"""Account-level portfolio construction for the paper-trading simulator.

The brain is a pure function of ``(panel, signals, edges, account, config) ->
(weights, orders)``. Historical replay and the Monday-forward step run the
identical code path, which is what makes the replay's evidence apply to the
account of record.

Layering, strictly downward -- no module imports one below it:

    account   costs   metrics   stops   (pure, no internal deps)
    feed                               (the one bounded DB read)
    pit_guard                          (feed; OPT-IN, imported by nothing below)
    risk                               (numpy; account, for the HWM decay)
    views                              (risk)
    optimizer                          (risk, views)
    simulator                          (everything above)
    store   report                     (persistence / rendering edges)

``pit_guard`` sits outside the replay path on purpose: it wraps ``feed``'s
``MarketPanel`` and signal dicts behind a declared "as of" clock that raises on
a lookahead read instead of silently returning one, for tests and in-progress
features that want that guarantee. ``simulator`` does not import it and never
will by default -- see ``pit_guard``'s own module docstring for what it does
and, as importantly, does not catch.

Domain math is numpy. **No pandas in domain code** -- the price panel is a
``[T, N]`` float64 array, not a DataFrame, because 633 symbols x 2520 days x 5
fields is ~64 MB as an array and roughly 1.5M Python objects as anything else,
and this machine has 16 GB shared with a running collector.

This package simulates and proposes. It never executes on a real brokerage
(repo design decision G, "Execution -- None, ever").
"""

from __future__ import annotations

__all__: list[str] = []
