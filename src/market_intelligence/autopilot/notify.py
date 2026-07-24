"""The Telegram projection: a briefing turned into a phone nudge that can fail.

The daily loop's whole point is to run unattended, so its notifier is held to a
stricter contract than the rest of the pipeline: *it must never raise*. A
missing token, a Telegram outage, a 500 from their API -- none of these are
reasons to fail a run whose real work (ingest, gating, the Obsidian note)
already landed. Every failure path here logs and returns ``False`` instead of
propagating, because a down notifier is a missed message, not a broken pipeline.

Two design choices follow from that and from testability:

* **Plain text, no ``parse_mode``.** Markdown/HTML modes make Telegram reject
  (HTTP 400) any message with an unescaped ``_``, ``*``, ``[`` -- all of which
  occur naturally in signal labels like ``insider_transaction/P/self/20d``.
  Sending plain text sidesteps a class of avoidable rejections entirely.
* **Injectable transport.** ``send`` builds its ``httpx.Client`` around an
  optional ``transport`` exactly as the API clients do, so the whole suite
  drives Telegram through ``httpx.MockTransport`` and never touches the network.

``render_message`` is split out as a pure ``Briefing -> str`` function: it does
no I/O, so the message body is unit-testable without a client, a config, or a
socket. Trade alerts (Task 8) get the same treatment: ``render_trade_alerts``
is a pure ``Sequence[TradeAlertRecord] -> str`` function, and both it and
``render_message`` share one I/O primitive, ``_post_text`` -- the
credential-check + POST + logging body that used to live inline in ``send``.
Trade alerts are deliberately their **own** Telegram message rather than
folded into the briefing nudge: the briefing stays a status glance, a trade
alert is an actionable ping that deserves its own notification.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from market_intelligence.autopilot.types import ChangeKind
from market_intelligence.logging_config import get_logger

if TYPE_CHECKING:
    from market_intelligence.autopilot.types import Briefing, EventAlert, SignalChange
    from market_intelligence.config import Config
    from market_intelligence.signals.trade_alerts import TradeAlertRecord
    from market_intelligence.signals.trade_plan import TradePlan

logger = get_logger(__name__)

#: Telegram's hard per-message ceiling. We render well under it and truncate
#: anything longer rather than let the API reject an oversized payload.
TELEGRAM_MAX_CHARS = 4096

#: Our own budget, comfortably below the ceiling so a truncation marker always
#: fits and the message reads as a nudge, not a wall of text.
_SAFE_CHARS = 3900

#: Only the first few alerts belong in a phone nudge; the note carries the rest.
_MAX_ALERTS = 5

#: Cap on change lines so a churny run still yields a short message.
_MAX_CHANGES = 6

_TRUNCATION_MARKER = "\n… (truncated)"


def _direction_arrow(direction: int) -> str:
    """Render an alert's sign as an arrow (neutral when unsigned)."""
    if direction > 0:
        return "↑"
    if direction < 0:
        return "↓"
    return "→"


def _ingest_line(ingest: dict[str, int]) -> str:
    """One compact 'what moved' line, or a plain note when nothing came in."""
    if not ingest:
        return "Ingest: nothing new"
    parts = [f"{name} {count}" for name, count in ingest.items()]
    return "Ingest: " + " · ".join(parts)


def _change_line(change: SignalChange) -> str:
    """A single change as ``KIND label (CAR, hit)`` -- names ACTIVATED cells."""
    return (
        f"{change.kind.upper()} {change.label} ({change.mean_car:+.2%}, hit {change.hit_rate:.0%})"
    )


def _alert_line(alert: EventAlert) -> str:
    """A single fired prediction as ``TICKER ↑ +CAR / Nd — basis``."""
    arrow = _direction_arrow(alert.direction)
    return (
        f"{alert.ticker} {arrow} {alert.predicted_car:+.2%} / {alert.horizon_days}d — {alert.basis}"
    )


def _changes_first_activated(changes: list[SignalChange]) -> list[SignalChange]:
    """Order changes so ACTIVATED cells are never the ones the cap drops."""
    return sorted(changes, key=lambda c: c.kind != ChangeKind.ACTIVATED)


def render_message(briefing: Briefing) -> str:
    """Turn a briefing into a short, plain-text Telegram nudge (pure, no I/O).

    Deliberately terse -- a phone glance, not the note: the run status, one
    ingest line, the change count with a few named changes (ACTIVATED first so
    the cap never hides a newly-trusted signal), and up to a few alerts. The
    result is truncated to stay well under Telegram's 4096-char ceiling.
    """
    lines: list[str] = [
        f"Market Intelligence — {briefing.as_of.isoformat()}",
        f"Run: {briefing.run_status}",
        "",
        _ingest_line(briefing.ingest),
    ]

    if briefing.changes:
        lines.append("")
        lines.append(f"Signals: {len(briefing.changes)} change(s)")
        ordered = _changes_first_activated(briefing.changes)
        lines.extend(f"  {_change_line(change)}" for change in ordered[:_MAX_CHANGES])
        if len(ordered) > _MAX_CHANGES:
            lines.append(f"  …and {len(ordered) - _MAX_CHANGES} more")

    if briefing.alerts:
        lines.append("")
        lines.append(f"Alerts: {len(briefing.alerts)}")
        lines.extend(f"  {_alert_line(alert)}" for alert in briefing.alerts[:_MAX_ALERTS])
        if len(briefing.alerts) > _MAX_ALERTS:
            lines.append(f"  …and {len(briefing.alerts) - _MAX_ALERTS} more")

    # On a partial/failed run the notes carry which steps broke; a single line
    # of them keeps the nudge honest about a degraded run without bloating it.
    if briefing.run_status != "success" and briefing.notes:
        lines.append("")
        lines.append("Notes: " + "; ".join(briefing.notes[:3]))

    message = "\n".join(lines)
    if len(message) > _SAFE_CHARS:
        message = message[: _SAFE_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return message


def _post_text(
    config: Config,
    text: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """POST already-rendered text to Telegram; return whether it was delivered.

    Never raises: a blank/absent token or chat id, a network error, or a
    non-2xx response each log and return ``False`` so the daily loop survives a
    down notifier. Returns ``True`` only on a 2xx from Telegram. ``transport``
    is injected into the client so tests run against ``httpx.MockTransport``.
    This is the shared I/O primitive behind both ``send`` (the briefing nudge)
    and ``send_trade_alerts`` (the trade-alert message) -- credential check,
    POST, and log event names live here exactly once.
    """
    token = (config.env.telegram_bot_token or "").strip()
    chat_id = (config.env.telegram_chat_id or "").strip()
    if not token or not chat_id:
        logger.warning("telegram_notify_skipped", reason="missing_credentials")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    timeout = httpx.Timeout(
        config.settings.http.timeout_seconds,
        connect=config.settings.http.connect_timeout_seconds,
    )

    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        logger.warning("telegram_notify_failed", error=str(exc))
        return False

    if response.is_success:
        logger.info("telegram_notify_sent", chat_id=chat_id, status=response.status_code)
        return True

    logger.warning(
        "telegram_notify_rejected",
        status=response.status_code,
        body=response.text[:500],
    )
    return False


def send(
    config: Config,
    briefing: Briefing,
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Push a briefing to Telegram; return whether it was delivered.

    Never raises: a blank/absent token or chat id, a network error, or a
    non-2xx response each log and return ``False`` so the daily loop survives a
    down notifier. Returns ``True`` only on a 2xx from Telegram. ``transport``
    is injected into the client so tests run against ``httpx.MockTransport``.
    """
    return _post_text(config, render_message(briefing), transport=transport)


def _kind_label(kind: str) -> str:
    """``"reaction_lag"`` -> ``"reaction lag"``; ``"price_gap"`` -> ``"price gap"``."""
    return kind.replace("_", " ")


def _why_line(evidence: dict[str, object]) -> str:
    """``"<event_type> event; <basis>"``, plus ``(n=<n_clusters>)`` when known.

    Every lookup goes through ``.get`` -- a record whose evidence dict is
    missing a key (a gap alert with no cell stats, say) must render a plainer
    line, never raise.
    """
    event_type = evidence.get("event_type") or "event"
    basis = evidence.get("basis") or ""
    line = f"{event_type} event; {basis}"
    n_clusters = evidence.get("n_clusters")
    if n_clusters:
        line += f" (n={n_clusters})"
    return line


def _plan_line(plan: TradePlan) -> str:
    """``"Plan: entry ~$X · stop $Y · target $Z (+P.P%)"``."""
    target_pct = (plan.target / plan.entry_ref - 1.0) if plan.entry_ref else 0.0
    return (
        f"Plan: entry ~${plan.entry_ref:.2f} · stop ${plan.stop:.2f} · "
        f"target ${plan.target:.2f} ({target_pct:+.1%})"
    )


def _size_line(plan: TradePlan) -> str:
    """The Size line, or the unsizeable variant when the account can't fund a share."""
    if plan.unsizeable:
        return (
            "Size: unsizeable at current risk settings · "
            f"exit by {plan.time_exit_date.isoformat()}"
        )
    return (
        f"Size: {plan.shares} sh (~${plan.notional:,.0f} → risks ${plan.risk_amount:,.0f}) "
        f"· exit by {plan.time_exit_date.isoformat()}"
    )


def _trade_alert_block(index: int, record: TradeAlertRecord) -> str:
    """One numbered alert block: headline, why, plan, size, the disclaimer."""
    arrow = _direction_arrow(record.direction)
    lines = [
        f"{index}) {record.ticker} {arrow} {_kind_label(record.kind)} "
        f"(confidence {record.confidence})",
        f"   Why: {_why_line(record.evidence)}",
        f"   {_plan_line(record.plan)}",
        f"   {_size_line(record.plan)}",
        "   Not auto-traded — your call.",
    ]
    return "\n".join(lines)


def render_trade_alerts(records: Sequence[TradeAlertRecord]) -> str:
    """Turn fired trade alerts into a plain-text Telegram message (pure, no I/O).

    Its own message, separate from the briefing nudge -- see the module
    docstring. Highest confidence first, one block per alert, truncated under
    the same ``_SAFE_CHARS`` budget as ``render_message`` so 30 records still
    fits comfortably under Telegram's ceiling.
    """
    ordered = sorted(records, key=lambda r: r.confidence, reverse=True)
    header = f"🎯 TRADE SIGNALS — {len(ordered)} alert(s)"
    blocks = [_trade_alert_block(i, record) for i, record in enumerate(ordered, start=1)]
    message = header + "\n\n" + "\n\n".join(blocks)
    if len(message) > _SAFE_CHARS:
        message = message[: _SAFE_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return message


def send_trade_alerts(
    config: Config,
    records: Sequence[TradeAlertRecord],
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Render and push trade alerts as their own Telegram message.

    An empty ``records`` sequence returns ``False`` without touching the
    network -- there is nothing to say, and a "0 alert(s)" ping would just be
    noise. Otherwise behaves exactly like ``send``: never raises, ``True``
    only on a 2xx.
    """
    if not records:
        return False
    return _post_text(config, render_trade_alerts(records), transport=transport)


__all__ = [
    "TELEGRAM_MAX_CHARS",
    "render_message",
    "render_trade_alerts",
    "send",
    "send_trade_alerts",
]
