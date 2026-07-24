#!/usr/bin/env python3
"""Send one clearly-labeled TEST trade alert through the real phone path.

Manual Phase A verification (plan Task 10): builds a realistic but entirely
FAKE TradeAlertRecord and pushes it through the real renderer and the real
Telegram credentials, proving the signal -> plan -> message -> phone chain end
to end. Every number is invented and the message says so. No database is
touched; nothing is persisted.

Run: uv run python scripts/send_test_trade_alert.py
Exit code 0 iff Telegram accepted the message.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

from market_intelligence.autopilot import notify
from market_intelligence.config import get_config
from market_intelligence.signals.trade_alerts import TradeAlertRecord
from market_intelligence.signals.trade_plan import TradePlan


def main() -> int:
    now = datetime.now(tz=UTC)
    plan = TradePlan(
        entry_ref=74.20,
        stop=71.10,
        target=76.40,
        shares=13,
        notional=964.60,
        risk_amount=40.30,
        time_exit_date=now.date() + timedelta(days=20),
        atr=1.55,
        unsizeable=False,
    )
    record = TradeAlertRecord(
        alert_id="manual-phase-a-check",
        kind="reaction_lag",
        ticker="TEST",
        trigger_key="manual:self:20",
        fired_at=now,
        direction=1,
        plan=plan,
        confidence=72,
        evidence={
            "event_type": "TEST (not a real signal)",
            "basis": "pipeline verification -- manual Phase A checkpoint",
            "hit_rate": 0.588,
            "n_clusters": 41,
        },
        event_id=None,
        edge_id="self",
    )
    sent = notify.send_trade_alerts(get_config(), [record])
    print(f"sent={sent}")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
