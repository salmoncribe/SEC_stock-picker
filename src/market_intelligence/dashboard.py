# ruff: noqa: E501, RUF001
"""A local browser dashboard for the market-intelligence research ledger.

Run through ``scripts/PAGE`` (or the installed ``PAGE`` command).  The server
only reads the JSON display snapshot written by the daily autopilot, so it is
safe to keep open while DuckDB is busy running the research pipeline.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote

from market_intelligence.config import get_config
from market_intelligence.dashboard_snapshot import refresh_snapshot, snapshot_path

DEFAULT_PORT = 8765


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the Market Intelligence dashboard.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--open", action="store_true", help="Open the dashboard in the default browser."
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Refresh the read-only display cache, then exit.",
    )
    args = parser.parse_args()
    config = get_config()
    config.paths.ensure()
    path = snapshot_path(config)
    try:
        refreshed_path = refresh_snapshot(config)
        print(f"Dashboard snapshot refreshed: {refreshed_path}")
    except Exception as exc:
        print(f"Dashboard snapshot unchanged (database busy or unavailable): {exc}")
    if args.refresh_only:
        return 0
    server = _server(path, args.port, _waiting_payload(path, config.paths.obsidian_vault_dir))
    url = f"http://127.0.0.1:{args.port}"
    if args.open:
        webbrowser.open(url)
    print(f"Dashboard listening at {url}")
    print("Press Ctrl-C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _server(path: Path, port: int, waiting_payload: dict[str, Any]) -> ThreadingHTTPServer:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_html(_page(_load_payload(path, waiting_payload), path))
                return
            if self.path == "/api/snapshot":
                self._send_json(_load_payload(path, waiting_payload))
                return
            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, document: str) -> None:
            body = document.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)


def _load_payload(path: Path, waiting_payload: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return waiting_payload
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "error", "message": f"Could not read dashboard snapshot: {exc}"}
    if not isinstance(data, dict):
        return {"state": "error", "message": "Dashboard snapshot is not a JSON object."}
    return data


def _waiting_payload(path: Path, vault_path: Path) -> dict[str, Any]:
    vault_name = vault_path.name
    return {
        "state": "waiting",
        "message": "No dashboard snapshot has been written yet.",
        "vault": {
            "vault_url": f"obsidian://open?vault={quote(vault_name)}",
            "vault_path": str(vault_path),
        },
        "snapshot_path": str(path),
    }


def _page(payload: dict[str, Any], path: Path) -> str:
    initial = json.dumps(payload).replace("</", "<\\/")
    generated = datetime.now(tz=UTC).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Intelligence — Live Ledger</title>
  <style>{_STYLES}</style>
</head>
<body>
  <main id="app" aria-live="polite"></main>
  <script>window.__DATA__ = {initial}; window.__SNAPSHOT_PATH__ = {json.dumps(str(path))}; window.__GENERATED__ = {json.dumps(generated)};</script>
  <script>{_SCRIPT}</script>
</body>
</html>"""


_STYLES = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Sans:wdth,wght@75..100,400..700&family=Newsreader:opsz,wght@6..72,400..700&display=swap');
:root { --ink:#10232b; --muted:#60757b; --paper:#edf1ed; --panel:#f9fbf8; --line:#cbd5d0; --teal:#007c78; --teal-pale:#d6f0e9; --green:#147a45; --red:#bf3d3d; --amber:#9a661c; --navy:#19384d; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:var(--paper); font-family:'Instrument Sans',system-ui,sans-serif; }
a { color:inherit; }
button { font:inherit; }
.shell { width:min(1510px, calc(100% - 42px)); margin:0 auto; padding:26px 0 50px; }
.mast { display:grid; grid-template-columns:1fr auto; gap:24px; align-items:end; border-bottom:1px solid var(--ink); padding:0 0 22px; }
.eyebrow,.label,.column-title,.side-note { font-family:'DM Mono',ui-monospace,monospace; font-size:11px; letter-spacing:.09em; text-transform:uppercase; }
.eyebrow { color:var(--teal); font-weight:500; }
h1 { margin:5px 0 0; font-family:Newsreader,Georgia,serif; font-size:clamp(42px,6vw,76px); font-weight:500; line-height:.85; letter-spacing:-.052em; }
.mast-actions { display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:end; }
.button { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--ink); padding:10px 13px; border-radius:999px; background:transparent; font-size:13px; text-decoration:none; cursor:pointer; }
.button:hover { background:var(--ink); color:white; }.button.primary { background:var(--teal); border-color:var(--teal); color:white; }.button.primary:hover { background:#00635f; }
.status { border-radius:999px; padding:6px 10px; font:11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.06em; }.status.success,.status.active { background:#cbeed8; color:#075b34; }.status.partial,.status.warning { background:#ffe6b8; color:#754800; }.status.failed,.status.danger { background:#f9d4d4; color:#962e2e; }.status.neutral { background:#dbe4e3; color:#476064; }
.metrics { display:grid; grid-template-columns:2fr repeat(4,1fr); border-bottom:1px solid var(--line); }.metric { min-height:130px; padding:23px 18px; border-right:1px solid var(--line); }.metric:first-child { padding-left:0; }.metric:last-child { border:0; }.metric .value { margin-top:20px; font-family:Newsreader,Georgia,serif; font-size:48px; letter-spacing:-.04em; line-height:.8; }.metric .value.small { font-size:30px; line-height:1; }.metric .sub { margin-top:8px; color:var(--muted); font-size:13px; }
.grid { display:grid; grid-template-columns:minmax(0,1.75fr) minmax(290px,.75fr); gap:30px; margin-top:30px; }.column-title { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--ink); padding-bottom:10px; color:var(--muted); }.column-title strong { color:var(--ink); font-weight:500; }
.ledger { overflow:hidden; background:var(--panel); border:1px solid var(--line); margin-top:12px; }.ledger-head,.trade { display:grid; grid-template-columns:1.2fr .75fr 1fr 1fr 1.1fr .6fr; gap:12px; align-items:center; }.ledger-head { background:#e1e9e5; padding:10px 16px; font:10px 'DM Mono',monospace; color:#50676b; letter-spacing:.05em; text-transform:uppercase; }.trade { padding:17px 16px; border-top:1px solid var(--line); }.trade:first-of-type { border-top:0; }.ticker-line { display:flex; align-items:center; gap:8px; }.ticker { font:600 19px 'DM Mono',monospace; letter-spacing:-.06em; text-decoration:none; }.tag { padding:3px 6px; border-radius:3px; background:#e3ebe8; color:#48656b; font:10px 'DM Mono',monospace; text-transform:uppercase; }.direction { font:10px 'DM Mono',monospace; letter-spacing:.04em; text-transform:uppercase; }.direction.long { color:var(--green); }.direction.short { color:var(--red); }.small { font-size:12px; color:var(--muted); margin-top:4px; }.mono { font-family:'DM Mono',monospace; font-size:12px; }.positive { color:var(--green); }.negative { color:var(--red); }.gauge { height:6px; overflow:hidden; background:#d8e1dd; border-radius:99px; margin-top:7px; }.gauge > span { display:block; height:100%; background:var(--teal); border-radius:99px; }.deadline { font:12px 'DM Mono',monospace; }.deadline.overdue { color:var(--red); }
.card { padding:19px; margin-top:12px; background:var(--panel); border:1px solid var(--line); }.card + .card { margin-top:14px; }.signal-row,.event-row,.outcome-row,.health-row { padding:13px 0; border-top:1px solid var(--line); }.signal-row:first-child,.event-row:first-child,.outcome-row:first-child,.health-row:first-child { border-top:0; padding-top:0; }.signal-row:last-child,.event-row:last-child,.outcome-row:last-child,.health-row:last-child { padding-bottom:0; }.signal-main,.event-top,.outcome-top,.health-top { display:flex; gap:9px; justify-content:space-between; align-items:baseline; }.signal-name { font:12px 'DM Mono',monospace; overflow-wrap:anywhere; }.reason { margin-top:6px; color:var(--muted); font-size:12px; line-height:1.35; }.measure { font:500 14px 'DM Mono',monospace; }.breakdown { display:flex; gap:10px; color:var(--muted); margin-top:7px; font:11px 'DM Mono',monospace; }.event-ticker,.outcome-ticker { font:600 14px 'DM Mono',monospace; text-decoration:none; }.event-detail { color:var(--muted); font-size:12px; margin-top:4px; }.outcome { text-transform:capitalize; font:10px 'DM Mono',monospace; padding:4px 6px; border:1px solid var(--line); border-radius:3px; }.outcome.hit-target { color:var(--green); border-color:#a8d3b5; }.outcome.hit-stop { color:var(--red); border-color:#e7b8b8; }.callout { background:var(--navy); color:#eff8f5; padding:21px; margin-top:12px; }.callout .label { color:#a2cbc3; }.callout p { margin:10px 0 15px; line-height:1.45; font-size:14px; }.callout .button { border-color:#a2cbc3; }.callout .button:hover { background:#eff8f5; color:var(--navy); }.system { margin-top:30px; }.health-row { display:grid; grid-template-columns:1fr auto; gap:12px; }.notes { color:var(--muted); font-size:12px; line-height:1.45; margin:12px 0 0; padding-left:17px; }.empty { padding:32px 20px; text-align:center; color:var(--muted); font-size:14px; }.waiting { max-width:680px; margin:14vh auto; padding:32px; background:var(--panel); border:1px solid var(--line); }.waiting h1 { font-size:53px; }.waiting code { font:12px 'DM Mono',monospace; overflow-wrap:anywhere; }.foot { display:flex; justify-content:space-between; gap:20px; margin-top:26px; color:var(--muted); font:11px 'DM Mono',monospace; }.flash { animation:flash .5s ease-out; } @keyframes flash { from { outline:3px solid #72d3c1; } to { outline:0 solid transparent; } }
@media (max-width:1000px) { .metrics { grid-template-columns:repeat(3,1fr); }.metric:first-child { grid-column:span 3; padding-left:0; }.grid { grid-template-columns:1fr; }.ledger-head { display:none; }.trade { grid-template-columns:1.2fr 1fr 1fr; }.trade > :nth-child(5) { grid-column:span 2; }.trade > :nth-child(6) { grid-column:3; grid-row:1; } }
@media (max-width:620px) { .shell { width:min(100% - 28px, 1510px); padding-top:18px; }.mast { grid-template-columns:1fr; }.mast-actions { justify-content:start; }.metrics { grid-template-columns:1fr 1fr; }.metric:first-child { grid-column:span 2; }.metric { min-height:105px; padding:18px 10px; }.metric:nth-child(odd) { padding-left:0; }.metric .value { font-size:38px; }.trade { grid-template-columns:1fr 1fr; gap:14px 8px; }.trade > :nth-child(5) { grid-column:span 2; }.trade > :nth-child(6) { grid-column:2; grid-row:1; }.foot { flex-direction:column; gap:4px; } }
@media (prefers-reduced-motion: reduce) { .flash { animation:none; } }
"""


_SCRIPT = r"""
const app = document.getElementById('app');
const asText = value => value === null || value === undefined || value === '' ? '—' : value;
const pct = value => value === null || value === undefined ? '—' : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`;
const money = value => value === null || value === undefined ? '—' : `$${Number(value).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
const stamp = value => value ? new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(value)) : '—';
const day = value => value ? new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric'}).format(new Date(`${value}T12:00:00`)) : '—';
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const stateClass = state => ['success','active','healthy','ok'].includes(String(state).toLowerCase()) ? 'success' : ['partial','warning','degraded'].includes(String(state).toLowerCase()) ? 'partial' : ['failed','error','unhealthy'].includes(String(state).toLowerCase()) ? 'failed' : 'neutral';
function openRows(rows) { if (!rows.length) return '<div class="empty">No open model calls. The ledger is clear.</div>'; return rows.map(a => { const p=a.progress_to_target===null?0:Math.max(0,Math.min(100,a.progress_to_target*100)); const actualClass=a.actual_return===null?'':a.actual_return>=0?'positive':'negative'; const overdue=a.exit_date && new Date(`${a.exit_date}T23:59:59`) < new Date(); return `<article class="trade"><div><div class="ticker-line"><a class="ticker" href="${esc(a.quote_url)}" target="_blank" rel="noreferrer">${esc(a.ticker)}</a><span class="tag">${esc(a.kind)}</span></div><a class="small" href="${esc(a.vault_url)}">Open company note ↗</a></div><div><div class="direction ${esc(a.direction)}">${esc(a.direction)}</div><div class="small">${stamp(a.fired_at)}</div></div><div><div class="mono">${money(a.entry)} → ${money(a.target)}</div><div class="small">stop ${money(a.stop)}</div></div><div><div class="mono ${actualClass}">${pct(a.actual_return)}</div><div class="small">last ${money(a.last_price)} · ${day(a.price_date)}</div></div><div><div class="mono">${pct(a.expected_return)} expected</div><div class="gauge" title="Progress toward target"><span style="width:${p}%"></span></div><div class="small">${a.actual_return===null?'awaiting price':'mark vs. model target'}</div></div><div><div class="deadline ${overdue?'overdue':''}">${day(a.exit_date)}</div><div class="small">exit date</div></div></article>`; }).join(''); }
function signals(rows) { if (!rows.length) return '<div class="empty">No signal cells are active yet. The model has not earned the right to fire.</div>'; return rows.map(s => `<div class="signal-row"><div class="signal-main"><span class="signal-name">${esc(s.label)}</span><span class="direction ${esc(s.direction)}">${esc(s.direction)}</span></div><div class="breakdown"><span class="measure ${s.mean_car>=0?'positive':'negative'}">${pct(s.mean_car)}</span><span>${pct(s.hit_rate)} hit</span><span>${esc(s.clusters)} clusters</span><span>${esc(s.confirm_streak)}× confirmed</span></div>${s.reason?`<div class="reason">${esc(s.reason)}</div>`:''}</div>`).join(''); }
function outcomes(rows) { if (!rows.length) return '<div class="empty">No resolved calls yet.</div>'; return rows.map(r => `<div class="outcome-row"><div class="outcome-top"><a class="outcome-ticker" href="${esc(r.quote_url)}" target="_blank" rel="noreferrer">${esc(r.ticker)}</a><span class="outcome ${esc(r.outcome).replace('_','-')}">${esc(r.outcome).replace('_',' ')}</span></div><div class="breakdown"><span>${esc(r.direction)}</span><span>model ${money(r.entry)} → ${money(r.target)}</span><span class="${r.actual_return>=0?'positive':'negative'}">actual ${pct(r.actual_return)}</span></div></div>`).join(''); }
function events(rows) { if (!rows.length) return '<div class="empty">No recent events in the signal feed.</div>'; return rows.map(r => `<div class="event-row"><div class="event-top"><a class="event-ticker" href="${esc(r.quote_url)}" target="_blank" rel="noreferrer">${esc(r.ticker)}</a><span class="small">${stamp(r.available_at)}</span></div><div class="event-detail">${esc(r.event_type)}${r.event_subtype?` / ${esc(r.event_subtype)}`:''}</div></div>`).join(''); }
function health(rows) { if (!rows.length) return '<div class="empty">No component health checks recorded.</div>'; return rows.map(r => `<div class="health-row"><div><div class="mono">${esc(r.component)}</div>${r.details?`<div class="small">${esc(r.details)}</div>`:''}</div><div><span class="status ${stateClass(r.status)}">${esc(r.status)}</span><div class="small">${stamp(r.observed_at)}</div></div></div>`).join(''); }
function render(data) { if (data.state) { app.innerHTML=`<section class="waiting"><div class="eyebrow">Market Intelligence / dashboard</div><h1>Standing by.</h1><p>${esc(data.message)}</p><p>The running autopilot writes its first dashboard snapshot after it finishes its next briefing. This page will become live without needing a competing database connection.</p><p><code>${esc(window.__SNAPSHOT_PATH__)}</code></p><div class="mast-actions"><a class="button primary" href="${esc(data.vault?.vault_url)}">Open Obsidian vault ↗</a><button class="button" onclick="location.reload()">Check again</button></div></section>`; return; } const b=data.briefing||{}; const m=data.metrics||{}; const notes=(b.notes||[]).map(n=>`<li>${esc(n)}</li>`).join(''); app.innerHTML=`<div class="shell"><header class="mast"><div><div class="eyebrow">Research ledger · no execution</div><h1>What the model said.<br>What the tape did.</h1></div><div class="mast-actions"><span class="status ${stateClass(b.status)}">${esc(b.status||'unknown')} run</span><a class="button primary" href="${esc(data.vault?.vault_url)}">Open Obsidian vault ↗</a><a class="button" href="${esc(data.vault?.briefing_url)}">Today’s briefing ↗</a><button class="button" onclick="location.reload()">Refresh</button></div></header><section class="metrics"><div class="metric"><div class="label">Latest model pass</div><div class="value small">${day(b.as_of)} <span class="status ${stateClass(b.status)}">${esc(b.status||'unknown')}</span></div><div class="sub">Snapshot ${stamp(data.captured_at)} · prices through ${day(m.latest_price_date)}</div></div><div class="metric"><div class="label">Open calls</div><div class="value">${esc(m.open_alerts||0)}</div><div class="sub">prediction ledger</div></div><div class="metric"><div class="label">Validated cells</div><div class="value">${esc(m.active_signals||0)}</div><div class="sub">currently firing</div></div><div class="metric"><div class="label">Targets / stops</div><div class="value small"><span class="positive">${esc(m.targets_hit||0)}</span> / <span class="negative">${esc(m.stops_hit||0)}</span></div><div class="sub">resolved trade calls</div></div><div class="metric"><div class="label">Time exits</div><div class="value">${esc(m.expired||0)}</div><div class="sub">closed at horizon</div></div></section><div class="grid"><section><div class="column-title"><strong>Active calls</strong><span>Actual mark versus expected target</span></div><div class="ledger"><div class="ledger-head"><span>Signal</span><span>Direction</span><span>Plan</span><span>Actual</span><span>Progress</span><span>Deadline</span></div>${openRows(data.open_alerts||[])}</div><div class="system"><div class="column-title"><strong>Recent intelligence</strong><span>Latest public inputs</span></div><div class="card">${events(data.recent_events||[])}</div></div></section><aside><div class="column-title"><strong>Trusted model</strong><span>Measured, not guessed</span></div><div class="card">${signals(data.active_signals||[])}</div><div class="callout"><div class="label">Research notebook</div><p>Every active call, source note, briefing, and company relationship lives in your local Obsidian vault.</p><a class="button" href="${esc(data.vault?.vault_url)}">Open the vault ↗</a></div><div class="system"><div class="column-title"><strong>Resolved calls</strong><span>Prediction versus result</span></div><div class="card">${outcomes(data.resolved_alerts||[])}</div></div><div class="system"><div class="column-title"><strong>System health</strong><span>Latest checks</span></div><div class="card">${health(data.system_health||[])}</div></div></aside></div>${notes?`<ul class="notes">${notes}</ul>`:''}<footer class="foot"><span>Dashboard is a read-only projection; it never submits orders.</span><span>Auto-refresh is manual · press Refresh after an autopilot pass</span></footer></div>`; }
render(window.__DATA__);
"""


if __name__ == "__main__":
    raise SystemExit(main())
