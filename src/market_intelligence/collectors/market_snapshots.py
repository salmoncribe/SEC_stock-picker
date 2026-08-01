"""Capture and fail-close validation of immutable decision-time market snapshots.

The functions here perform only read operations through vendor-neutral
protocols.  They do not open a database connection, make an order, or fall
back to daily prices.  Integration code may persist the resulting frozen
snapshot later, after any slow provider call has completed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

from market_intelligence import hashing
from market_intelligence.clients.realtime_market import ReadOnlyBrokerClient, RealtimeMarketClient
from market_intelligence.schemas.market_snapshot import (
    AccountState,
    BorrowStatus,
    MarketDataHealth,
    MarketSession,
    MarketSnapshot,
    MarketSnapshotPolicy,
    MarketStatus,
    RealtimeQuote,
    TradeSide,
)


def _age_seconds(now: datetime, then: datetime | None) -> float | None:
    """Return age only when both clocks are timezone-aware and comparable."""
    if now.tzinfo is None or then is None or then.tzinfo is None:
        return None
    return (now - then).total_seconds()


def _is_aware(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _read_or_none[T](fn: Callable[[], T | None]) -> tuple[T | None, bool]:
    """Run one read-only adapter call, containing provider failures as invalid state."""
    try:
        return fn(), False
    except Exception:  # Provider outage must suppress, never escape as eligibility.
        return None, True


def evaluate_market_data_health(
    *,
    symbol: str,
    side: TradeSide,
    quantity: int,
    observed_at: datetime,
    quote: RealtimeQuote | None,
    market_status: MarketStatus | None,
    borrow: BorrowStatus | None,
    account: AccountState | None,
    policy: MarketSnapshotPolicy = MarketSnapshotPolicy(),
    previous_feed_sequence: int | None = None,
    market_read_failed: bool = False,
    broker_read_failed: bool = False,
) -> MarketDataHealth:
    """Return all hard suppression reasons for one candidate evaluation.

    The function is deterministic for supplied facts.  It never upgrades an
    incomplete condition to a warning: every unknown, stale, contradictory,
    or restricted condition makes ``healthy`` false.
    """
    reasons: list[str] = []
    if quantity <= 0:
        reasons.append("invalid_quantity")
    if not _is_aware(observed_at):
        reasons.append("observation_time_naive")

    quote_age: float | None = None
    status_age: float | None = None
    broker_age: float | None = None
    clock_drift: float | None = None

    if market_read_failed:
        reasons.append("market_data_unavailable")
    if quote is None:
        reasons.append("no_quote")
    else:
        if quote.symbol.upper() != symbol.upper():
            reasons.append("quote_symbol_mismatch")
        if quote.bid is None or quote.ask is None:
            reasons.append("quote_missing_bid_or_ask")
        elif quote.bid <= 0 or quote.ask <= 0:
            reasons.append("quote_non_positive")
        elif quote.bid > quote.ask:
            reasons.append("quote_crossed")
        elif quote.bid == quote.ask:
            reasons.append("quote_locked")
        if (
            quote.bid_size is None
            or quote.ask_size is None
            or quote.bid_size <= 0
            or quote.ask_size <= 0
        ):
            reasons.append("quote_missing_size")
        if not quote.is_firm:
            reasons.append("quote_not_firm")
        quote_age = _age_seconds(observed_at, quote.received_at)
        if quote_age is None:
            reasons.append("quote_receipt_time_unknown")
        elif quote_age < 0:
            reasons.append("quote_from_future")
        elif quote_age > policy.max_quote_age_seconds:
            reasons.append("quote_stale")
        exchange_time = quote.exchange_time
        receipt_time = quote.received_at
        if not _is_aware(exchange_time) or not _is_aware(receipt_time):
            reasons.append("quote_exchange_time_unknown")
        else:
            # _is_aware establishes the values are non-null; the assertions
            # make that narrowing explicit for static type checkers.
            assert exchange_time is not None
            assert receipt_time is not None
            clock_drift = abs((receipt_time - exchange_time).total_seconds())
            if clock_drift > policy.max_clock_drift_seconds:
                reasons.append("quote_clock_drift")
        if quote.sequence is None:
            reasons.append("feed_sequence_unknown")
        elif quote.sequence < 0:
            reasons.append("feed_sequence_invalid")
        elif previous_feed_sequence is not None:
            if quote.sequence <= previous_feed_sequence:
                reasons.append("feed_sequence_rewind")
            elif quote.sequence > previous_feed_sequence + 1:
                reasons.append("feed_sequence_gap")

    if market_status is None:
        reasons.append("no_market_status")
    else:
        if market_status.symbol.upper() != symbol.upper():
            reasons.append("status_symbol_mismatch")
        status_age = _age_seconds(observed_at, market_status.as_of)
        if status_age is None:
            reasons.append("status_time_unknown")
        elif status_age < 0:
            reasons.append("status_from_future")
        elif status_age > policy.max_status_age_seconds:
            reasons.append("status_stale")
        if not market_status.is_trading or market_status.is_halted:
            reasons.append("market_halt")
        if market_status.luld_active:
            reasons.append("luld_active")
        if market_status.ssr_active and not policy.allow_ssr:
            reasons.append("ssr_active")
        if market_status.session == MarketSession.UNKNOWN:
            reasons.append("market_session_unknown")
        elif (
            market_status.session == MarketSession.OPENING_AUCTION
            and not policy.allow_opening_auction
        ):
            reasons.append("opening_auction")
        elif (
            market_status.session == MarketSession.CLOSING_AUCTION
            and not policy.allow_closing_auction
        ):
            reasons.append("closing_auction")
        elif market_status.session != MarketSession.REGULAR and not policy.allow_extended_hours:
            reasons.append("outside_regular_session")

        opened_age = _age_seconds(observed_at, market_status.session_open_at)
        if (
            market_status.session == MarketSession.REGULAR
            and not policy.allow_first_minutes
            and (opened_age is None or opened_age < 0)
        ):
            reasons.append("session_open_time_unknown")
        elif (
            market_status.session == MarketSession.REGULAR
            and not policy.allow_first_minutes
            and opened_age is not None
            and opened_age < policy.first_minutes * 60
        ):
            reasons.append("first_minutes_restricted")
        closes_in = (
            _age_seconds(market_status.session_close_at, observed_at)
            if market_status.session_close_at
            else None
        )
        if (
            market_status.session == MarketSession.REGULAR
            and not policy.allow_last_minutes
            and (closes_in is None or closes_in < 0)
        ):
            reasons.append("session_close_time_unknown")
        elif (
            market_status.session == MarketSession.REGULAR
            and not policy.allow_last_minutes
            and closes_in is not None
            and closes_in <= policy.last_minutes * 60
        ):
            reasons.append("last_minutes_restricted")

    if broker_read_failed:
        reasons.append("broker_data_unavailable")
    if account is None:
        reasons.append("no_broker_account_state")
    else:
        broker_age = _age_seconds(observed_at, account.as_of)
        if broker_age is None:
            reasons.append("broker_account_time_unknown")
        elif broker_age < 0:
            reasons.append("broker_account_from_future")
        elif broker_age > policy.max_broker_age_seconds:
            reasons.append("broker_account_stale")
        ack_age = _age_seconds(observed_at, account.broker_acknowledged_at)
        if ack_age is None:
            reasons.append("broker_acknowledgment_unknown")
        elif ack_age < 0:
            reasons.append("broker_acknowledgment_from_future")
        elif ack_age > policy.max_broker_age_seconds:
            reasons.append("broker_acknowledgment_delayed")
        if account.account_restricted or account.restrictions:
            reasons.append("broker_account_restricted")

    # Borrow facts are material only for a short candidate.  Long candidates
    # still require current account restrictions above, but cannot be blocked
    # merely because a broker omitted irrelevant locate data.
    if side == TradeSide.SHORT:
        if borrow is None:
            reasons.append("no_borrow_status")
        else:
            if borrow.symbol.upper() != symbol.upper():
                reasons.append("borrow_symbol_mismatch")
            borrow_age = _age_seconds(observed_at, borrow.as_of)
            if borrow_age is None:
                reasons.append("borrow_time_unknown")
            elif borrow_age < 0:
                reasons.append("borrow_from_future")
            elif borrow_age > policy.max_broker_age_seconds:
                reasons.append("borrow_stale")
            borrow_ack_age = _age_seconds(observed_at, borrow.broker_acknowledged_at)
            if borrow_ack_age is None:
                reasons.append("borrow_acknowledgment_unknown")
            elif borrow_ack_age < 0:
                reasons.append("borrow_acknowledgment_from_future")
            elif borrow_ack_age > policy.max_broker_age_seconds:
                reasons.append("borrow_acknowledgment_delayed")
            if not borrow.is_borrowable:
                reasons.append("borrow_unavailable")
            if borrow.is_hard_to_borrow:
                reasons.append("hard_to_borrow")
            if borrow.is_recalled:
                reasons.append("borrow_recalled")
            if borrow.account_restricted or borrow.restrictions:
                reasons.append("broker_restriction")
            if not borrow.locate_id:
                reasons.append("borrow_locate_missing")
            if borrow.approved_quantity is None or borrow.approved_quantity < quantity:
                reasons.append("locate_quantity_exceeded")
            expires_at = borrow.expires_at
            if not _is_aware(expires_at):
                reasons.append("locate_expiration_unknown")
            else:
                assert expires_at is not None
                if observed_at >= expires_at:
                    reasons.append("locate_expired")

    unique_reasons = tuple(sorted(set(reasons)))
    return MarketDataHealth(
        healthy=not unique_reasons,
        suppression_reasons=unique_reasons,
        quote_age_seconds=quote_age,
        status_age_seconds=status_age,
        broker_age_seconds=broker_age,
        clock_drift_seconds=clock_drift,
    )


def capture_market_snapshot(
    market_client: RealtimeMarketClient,
    broker_client: ReadOnlyBrokerClient,
    *,
    symbol: str,
    side: TradeSide | str,
    quantity: int,
    observed_at: datetime,
    policy: MarketSnapshotPolicy = MarketSnapshotPolicy(),
    previous_feed_sequence: int | None = None,
) -> MarketSnapshot:
    """Read facts once and freeze an ineligible-or-healthy snapshot.

    Provider errors are intentionally captured as health failures.  The
    snapshot caller can persist them for audit without keeping a database lock
    while a data source is slow or unavailable.
    """
    normalized_symbol = symbol.strip().upper()
    try:
        normalized_side = TradeSide(side)
    except ValueError:
        # Keep a valid immutable object for audit; the explicit reason below
        # makes an invalid caller direction fail closed.
        normalized_side = TradeSide.LONG
        quantity = -abs(quantity) if quantity else 0

    quote, quote_failed = _read_or_none(lambda: market_client.get_quote(normalized_symbol))
    latest_trade, _trade_failed = _read_or_none(
        lambda: market_client.get_latest_trade(normalized_symbol)
    )
    status, status_failed = _read_or_none(
        lambda: market_client.get_market_status(normalized_symbol)
    )
    bar, _bar_failed = _read_or_none(
        lambda: market_client.get_latest_one_minute_bar(normalized_symbol, as_of=observed_at)
    )
    account, account_failed = _read_or_none(broker_client.get_account_state)
    borrow, borrow_failed = _read_or_none(
        lambda: broker_client.get_borrow_status(normalized_symbol)
    )
    market_failed = quote_failed or status_failed
    broker_failed = account_failed or (borrow_failed and normalized_side == TradeSide.SHORT)
    health = evaluate_market_data_health(
        symbol=normalized_symbol,
        side=normalized_side,
        quantity=quantity,
        observed_at=observed_at,
        quote=quote,
        market_status=status,
        borrow=borrow,
        account=account,
        policy=policy,
        previous_feed_sequence=previous_feed_sequence,
        market_read_failed=market_failed,
        broker_read_failed=broker_failed,
    )
    payload = {
        "symbol": normalized_symbol,
        "side": normalized_side.value,
        "quantity": quantity,
        "observed_at": observed_at,
        "policy_version": policy.version,
        "quote": asdict(quote) if quote else None,
        "latest_trade": asdict(latest_trade) if latest_trade else None,
        "market_status": asdict(status) if status else None,
        "one_minute_bar": asdict(bar) if bar else None,
        "borrow": asdict(borrow) if borrow else None,
        "account": asdict(account) if account else None,
        "health": asdict(health),
        "market_provider": getattr(market_client, "provider_name", None),
        "broker_provider": getattr(broker_client, "provider_name", None),
    }
    return MarketSnapshot(
        snapshot_id=hashing.sha256_json(payload),
        symbol=normalized_symbol,
        side=normalized_side,
        quantity=quantity,
        observed_at=observed_at,
        policy_version=policy.version,
        quote=quote,
        latest_trade=latest_trade,
        market_status=status,
        one_minute_bar=bar,
        borrow=borrow,
        account=account,
        health=health,
        market_provider=getattr(market_client, "provider_name", None),
        broker_provider=getattr(broker_client, "provider_name", None),
    )


__all__ = ["capture_market_snapshot", "evaluate_market_data_health"]
