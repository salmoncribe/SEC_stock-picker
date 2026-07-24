"""Offline tests for the Telegram notifier.

Every case drives the notifier through ``httpx.MockTransport`` or asserts a
graceful no-op, so the suite never touches the network. The contract under test
is the graceful-degradation one: ``send`` must return a bool and never raise,
whatever Telegram (or the config) does.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import httpx

from market_intelligence.autopilot.notify import (
    TELEGRAM_MAX_CHARS,
    render_message,
    render_trade_alerts,
    send,
    send_trade_alerts,
)
from market_intelligence.autopilot.types import (
    Briefing,
    ChangeKind,
    EventAlert,
    RunStatus,
    SignalChange,
)
from market_intelligence.config import Config
from market_intelligence.signals.trade_alerts import TradeAlertRecord
from market_intelligence.signals.trade_plan import TradePlan


def _activated_change() -> SignalChange:
    return SignalChange(
        event_type="insider_transaction",
        event_subtype="P",
        edge_type="self",
        horizon_days=20,
        kind=ChangeKind.ACTIVATED,
        mean_car=0.032,
        hit_rate=0.588,
        n_clusters=14,
        reason="earned 2nd consecutive holdout confirmation",
    )


def _demoted_change() -> SignalChange:
    return SignalChange(
        event_type="filing_item",
        event_subtype="2.02",
        edge_type="topic",
        horizon_days=5,
        kind=ChangeKind.DEMOTED,
        mean_car=-0.004,
        hit_rate=0.49,
        n_clusters=31,
        reason="weakened below the economic floor",
    )


def _alert() -> EventAlert:
    return EventAlert(
        ticker="AAPL",
        event_type="insider_transaction",
        event_subtype="P",
        available_on=date(2026, 7, 22),
        horizon_days=20,
        direction=1,
        predicted_car=0.021,
        basis="active, holdout hit 58.8%",
    )


def _briefing(**overrides: object) -> Briefing:
    base: dict[str, object] = {
        "as_of": date(2026, 7, 22),
        "run_status": RunStatus.SUCCESS,
        "ingest": {"prices": 12, "filings": 3, "events": 5},
        "changes": [_activated_change(), _demoted_change()],
        "alerts": [_alert()],
        "active_signals": [_activated_change()],
        "notes": [],
    }
    base.update(overrides)
    return Briefing(**base)  # type: ignore[arg-type]


def _with_creds(config: Config, *, token: str | None, chat_id: str | None) -> Config:
    config.env.telegram_bot_token = token
    config.env.telegram_chat_id = chat_id
    return config


# --------------------------------------------------------------------------- #
# render_message                                                              #
# --------------------------------------------------------------------------- #
def test_render_message_is_short_and_names_activated_signal() -> None:
    message = render_message(_briefing())

    assert len(message) < TELEGRAM_MAX_CHARS
    assert RunStatus.SUCCESS in message
    # The ACTIVATED cell must be named, by kind and by label.
    assert "ACTIVATED" in message
    assert "insider_transaction/P/self/20d" in message


def test_render_message_truncates_a_churny_run() -> None:
    many_changes = [_demoted_change() for _ in range(200)]
    many_alerts = [_alert() for _ in range(200)]
    message = render_message(_briefing(changes=many_changes, alerts=many_alerts))

    assert len(message) < TELEGRAM_MAX_CHARS
    # Activated ordering + caps must not drop the count headline.
    assert "200 change(s)" in message


def test_render_message_surfaces_notes_on_failed_run() -> None:
    message = render_message(
        _briefing(run_status=RunStatus.FAILED, notes=["price step failed: timeout"])
    )

    assert RunStatus.FAILED in message
    assert "price step failed: timeout" in message


# --------------------------------------------------------------------------- #
# send — happy path                                                          #
# --------------------------------------------------------------------------- #
def test_send_returns_true_and_posts_expected_request(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send(config, _briefing(), transport=transport)

    assert result is True
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.telegram.org/botbot-token-123/sendMessage"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["chat_id"] == "99887766"
    assert body["disable_web_page_preview"] is True
    assert "Market Intelligence" in body["text"]
    # Plain text: no parse_mode is set, so Telegram cannot 400 on unescaped chars.
    assert "parse_mode" not in body


# --------------------------------------------------------------------------- #
# send — graceful no-op on missing credentials (NO network call)             #
# --------------------------------------------------------------------------- #
def test_send_returns_false_when_token_missing_and_makes_no_call(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token=None, chat_id="99887766")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send(config, _briefing(), transport=transport)

    assert result is False
    assert calls["count"] == 0


def test_send_returns_false_when_chat_id_blank_and_makes_no_call(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="   ")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send(config, _briefing(), transport=transport)

    assert result is False
    assert calls["count"] == 0


# --------------------------------------------------------------------------- #
# send — graceful no-op when Telegram is down                                #
# --------------------------------------------------------------------------- #
def test_send_returns_false_on_500(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    transport = httpx.MockTransport(handler)
    result = send(config, _briefing(), transport=transport)

    assert result is False


def test_send_returns_false_on_connect_error(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    result = send(config, _briefing(), transport=transport)

    assert result is False


# --------------------------------------------------------------------------- #
# render_trade_alerts / send_trade_alerts                                     #
# --------------------------------------------------------------------------- #
def _trade_plan(**overrides: object) -> TradePlan:
    base: dict[str, object] = {
        "entry_ref": 74.20,
        "stop": 71.10,
        "target": 76.40,
        "shares": 13,
        "notional": 964.60,
        "risk_amount": 100.0,
        "time_exit_date": date(2026, 8, 20),
        "atr": 1.55,
        "unsizeable": False,
    }
    base.update(overrides)
    return TradePlan(**base)  # type: ignore[arg-type]


def _trade_alert_record(
    *, plan: TradePlan | None = None, **overrides: object
) -> TradeAlertRecord:
    base: dict[str, object] = {
        "alert_id": "a1",
        "kind": "reaction_lag",
        "ticker": "NKE",
        "trigger_key": "ev1:self:20",
        "fired_at": datetime(2026, 7, 22, 9, 0),
        "direction": 1,
        "plan": plan or _trade_plan(),
        "confidence": 72,
        "evidence": {
            "event_type": "insider_transaction",
            "event_subtype": "P",
            "basis": "active cell, holdout hit 58.8%",
            "hit_rate": 0.588,
            "n_clusters": 41,
        },
        "event_id": "ev1",
        "edge_id": "self",
    }
    base.update(overrides)
    return TradeAlertRecord(**base)  # type: ignore[arg-type]


def test_render_trade_alerts_single_record_has_all_fields() -> None:
    message = render_trade_alerts([_trade_alert_record()])

    assert "TRADE SIGNALS" in message
    assert "NKE ↑" in message
    assert "confidence 72" in message
    assert "Why:" in message
    assert "insider_transaction" in message
    assert "active cell, holdout hit 58.8%" in message
    assert "n=41" in message
    assert "Plan:" in message
    assert "$74.20" in message
    assert "$71.10" in message
    assert "$76.40" in message
    assert "Size:" in message
    assert "13 sh" in message
    assert "risks $100" in message
    assert "exit by 2026-08-20" in message
    assert "Not auto-traded" in message


def test_render_trade_alerts_sorts_highest_confidence_first() -> None:
    low = _trade_alert_record(alert_id="a-low", ticker="LOW", confidence=40)
    high = _trade_alert_record(alert_id="a-high", ticker="HIGH", confidence=90)

    message = render_trade_alerts([low, high])

    assert message.index("HIGH") < message.index("LOW")


def test_render_trade_alerts_unsizeable_record_skips_share_count() -> None:
    plan = _trade_plan(shares=0, notional=0.0, risk_amount=0.0, unsizeable=True)
    record = _trade_alert_record(plan=plan)

    message = render_trade_alerts([record])

    assert "unsizeable at current risk settings" in message
    assert "sh (" not in message


def test_render_trade_alerts_truncates_thirty_records() -> None:
    records = [
        _trade_alert_record(alert_id=f"a{i}", ticker=f"TKR{i}", confidence=50 + i)
        for i in range(30)
    ]

    message = render_trade_alerts(records)

    assert len(message) <= TELEGRAM_MAX_CHARS
    assert message.endswith("(truncated)")


def test_send_trade_alerts_returns_true_and_posts_rendered_text(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")
    records = [_trade_alert_record()]
    expected_text = render_trade_alerts(records)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send_trade_alerts(config, records, transport=transport)

    assert result is True
    assert captured["url"] == "https://api.telegram.org/botbot-token-123/sendMessage"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["text"] == expected_text


def test_send_trade_alerts_returns_false_on_500(tmp_config: Config) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    transport = httpx.MockTransport(handler)
    result = send_trade_alerts(config, [_trade_alert_record()], transport=transport)

    assert result is False


def test_send_trade_alerts_returns_false_when_creds_missing_and_makes_no_call(
    tmp_config: Config,
) -> None:
    config = _with_creds(tmp_config, token=None, chat_id="99887766")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send_trade_alerts(config, [_trade_alert_record()], transport=transport)

    assert result is False
    assert calls["count"] == 0


def test_send_trade_alerts_returns_false_for_empty_records_and_makes_no_call(
    tmp_config: Config,
) -> None:
    config = _with_creds(tmp_config, token="bot-token-123", chat_id="99887766")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    result = send_trade_alerts(config, [], transport=transport)

    assert result is False
    assert calls["count"] == 0
