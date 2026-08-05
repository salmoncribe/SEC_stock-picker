# Robinhood MCP — what it can do, and how our strategy maps onto it

Researched 2026-08-04. Companion to `2026-08-04-trading-strategy-RESULTS.md`.

## What exists

Robinhood shipped **official agentic trading** in May 2026 — one of the first
brokerage MCP servers from a major US retail broker.

| | |
|---|---|
| endpoint | `https://agent.robinhood.com/mcp/trading` |
| auth | OAuth through Robinhood (the agent never sees the password) |
| clients supported | Claude Code, Claude Desktop, ChatGPT, Codex, Cursor, Grok |
| account | a **separate, dedicated Agentic account**, funded on its own |
| asset classes | **equities only** in beta (options/crypto/futures "coming soon") |

The dedicated-account design is the important safety property: the agent can
only place orders in the Agentic account. Every other Robinhood account is
read-only to it. Requires a primary individual account in good standing first.

## The 12 tools

**Read (7):** `get_accounts` · `get_portfolio` (value, balances, buying power) ·
`get_equity_positions` · `get_equity_quotes` · `get_equity_orders` · `search` ·
`get_watchlists`

**Watchlist (2):** `add_to_watchlist` · `update_watchlist`

**Order (3):** `review_equity_order` (simulate, return pre-trade warnings) ·
`place_equity_order` · `cancel_equity_order`

`review_equity_order` → `place_equity_order` is the safety pair. Note it is a
*convention*, not enforced — an agent configured to run autonomously can place
without review. Robinhood sends push notifications on every trade and the agent
can be disconnected instantly.

## Answering the leverage question

**Short answer: no, and we shouldn't yet anyway.**

1. **Margin in the Agentic account is undocumented.** Not mentioned in the
   overview, the newsroom post, or any third-party writeup. Must be checked
   in-app once the account exists. Treat as unavailable until proven otherwise.
2. **Robinhood margin generally** needs **$2,000 minimum portfolio value** —
   your account is *exactly* at that line, so any loss drops you below
   eligibility. Rate is **5%** on balances up to $50k (floats with Fed Funds).
3. **Robinhood Gold** ($5/mo) gives the **first $1,000 of margin interest-free.**
   On a $2,000 account that is 1.5× leverage at zero interest — which sounds
   perfect until you price it: our backtest puts 1.5× at 21.64% vs 16.79%, so
   +4.85% on $2,000 = **+$97/yr**, against Gold's **$60/yr** fee. Net **+$37/yr**
   for a drawdown that goes from −17% to −25%. Not worth it.
4. **Our own result says wait regardless.** The backtest window contains no bear
   market. Levering before replaying 2020 and 2022 is levering an untested
   drawdown.

**Shorting** is separately interesting: Robinhood *does* support short selling
now (margin enabled, $2,000 minimum, Good-for-Day only to open, borrow fees vary
by symbol). That is the beta-hedge leg our strategy wants — but it needs margin,
which needs more than $2,000, and it is undocumented whether the Agentic account
allows it. Park this until the account is larger.

## The finding that changes the design

**Do not use a native trailing-stop order, even if `place_equity_order` supports
one.**

Our backtested rule evaluates the trailing stop **on the daily close** and exits
at that close. A broker-native trailing stop triggers **intraday**, on any wick
that touches the threshold, and fills wherever the market is at that moment.
Those are different strategies. The handoff already measured what intraday
triggering costs: 279 stop exits, **100% realized worse than the stop level**,
mean −12.3%, worst −44.7%.

So the trailing stop must be **our daily monitor**, not a resting broker order:

```
each trading day, after the close:
    for each open position:
        peak = max(peak_so_far, today's close)
        if today's close <= peak * 0.85:
            queue a market sell for the next open
```

This also sidesteps the fact that **which order types `place_equity_order`
accepts is undocumented** — the only type publicly demonstrated is market. A
market order at the next open is all our design needs.

## Architecture

I can't place trades — that's a firm line on my side. But the split falls out
cleanly anyway, and it's the right architecture regardless:

```
  our repo (quant)                    │  your agent + Robinhood MCP
  ────────────────────────────────────┼──────────────────────────────
  signals → admission filter          │  get_portfolio    (reconcile)
  → 40 equal-weight slots             │  get_equity_positions
  → trailing-stop state per position  │  review_equity_order  ← every order
  → emits daily_plan.json             │  place_equity_order
                                      │  get_equity_orders (confirm fills)
```

The interface is one file. Our side owns *what to trade and why*; your agent
owns *placing it*. Every order is auditable because the plan that produced it is
on disk before anything is sent.

```json
{
  "as_of": "2026-08-05",
  "account_equity": 2000.00,
  "plan_hash": "sha256:...",
  "orders": [
    {"action": "BUY",  "symbol": "ABC", "notional": 50.00,
     "reason": "insider_purchase P 2026-08-04, slot 12/40"},
    {"action": "SELL", "symbol": "XYZ", "shares": 1.8342,
     "reason": "trailing stop: close 41.20 <= peak 48.50 * 0.85"}
  ],
  "holds": 38,
  "invariants": {"max_orders": 8, "max_notional_per_order": 60.00,
                 "no_symbol_twice": true, "long_only": true}
}
```

The `invariants` block is the point: your agent refuses any plan that violates
them. A bug that tries to put 90% of the account into one name gets stopped by
the file format, not by anyone noticing.

## Three constraints worth knowing before you build

1. **There is no paper trading.** Real money from the first order. Mitigation:
   run **propose-only for 4–6 weeks** — emit `daily_plan.json`, place nothing,
   and each day compare what the plan *would* have done against what actually
   happened. That is also how we get the one number the backtest can't give us:
   **realized fill quality vs the 27bps we modelled.**
2. **OAuth reportedly fails outside localhost.** Reviewers found the integration
   works when a terminal is running on the laptop and authorization fails when
   deployed elsewhere. Fine for us — the Mac mini is always on and already runs
   `ai.quant.autopilot` under launchd. But it means no cloud deployment.
3. **One agentic account per user.** No running two strategies side by side, and
   no separate test account. Reinforces the propose-only phase.

## Suggested sequence

| phase | what | gate to advance |
|---|---|---|
| 0 | Connect the MCP; open + fund the Agentic account; check in-app whether margin/shorting are even offered there | tools respond to `get_accounts` |
| 1 | Build `daily_plan.json` emitter (candidates, slots, trailing-stop state) | plan reproduces the backtest's decisions on historical days |
| 2 | **Propose-only, 4–6 weeks.** No orders placed. | modelled vs realized cost gap measured and < 2× |
| 3 | Live, review-then-place, one order at a time | 20+ fills, no invariant violations |
| 4 | Reconsider leverage — only after 2020/2022 replays | replays pass at 1.5× |

Phase 2 is the one to not skip. It costs six weeks and buys the only honest
answer to "does this work outside a backtest."

## Open questions the docs don't answer

- Which order types `place_equity_order` accepts (only market is demonstrated).
- Whether the Agentic account supports margin at all.
- Whether it supports shorting.
- Rate limits, order size caps, response shapes (one reviewer had to
  reverse-engineer them).

All four are answerable in ten minutes once the account is connected — probe
with `review_equity_order`, which simulates without placing.

## Sources

[Agentic Trading overview](https://robinhood.com/us/en/support/articles/agentic-trading-overview/) ·
[Robinhood is Now Open to Agents](https://robinhood.com/us/en/newsroom/robinhood-is-now-open-to-agents/) ·
[Agentic Trading product page](https://robinhood.com/us/en/agentic-trading/) ·
[Robinhood MCP tool list (Gamut)](https://www.gamut.so/mcp/payments-finance/robinhood) ·
[MCP URL explained (SecProve)](https://secprove.com/trading-agent-safety/robinhood-trading-mcp-url) ·
[Short selling](https://robinhood.com/us/en/support/articles/short-selling/) ·
[Margin rates](https://www.firstcard.app/learn/robinhood-margin-interest-rate) ·
[Critical review (NexusTrade)](https://nexustrade.io/blog/robinhood-agentic-trading-mcp-review-20260708) ·
[TechCrunch coverage](https://techcrunch.com/2026/05/27/robinhood-now-lets-your-ai-agents-trade-stocks/)
