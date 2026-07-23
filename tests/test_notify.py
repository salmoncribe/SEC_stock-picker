"""Offline tests for the Telegram notifier.

Every case drives the notifier through ``httpx.MockTransport`` or asserts a
graceful no-op, so the suite never touches the network. The contract under test
is the graceful-degradation one: ``send`` must return a bool and never raise,
whatever Telegram (or the config) does.
"""

from __future__ import annotations

from datetime import date

import httpx

from market_intelligence.autopilot.notify import (
    TELEGRAM_MAX_CHARS,
    render_message,
    send,
)
from market_intelligence.autopilot.types import (
    Briefing,
    ChangeKind,
    EventAlert,
    RunStatus,
    SignalChange,
)
from market_intelligence.config import Config


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
