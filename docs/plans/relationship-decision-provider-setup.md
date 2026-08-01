# Relationship Decision Provider Setup

Last updated: 2026-07-26

The relationship-decision pipeline is built but intentionally research-only until
three external rails are configured and validated.  For the current testing
phase, only the SEC rail is required; news/first-public research can use bounded
web scraping plus the local Ollama model.

1. First-public verification outside EDGAR.
2. Realtime market snapshots with quotes, spreads, halts, and replay data.
3. Read-only broker/borrow availability.

Telegram credentials are already present locally, so message delivery is not the
missing piece. The missing piece is provider-grade evidence.

## Recommended setup path

### 0. Current testing path

Use this now:

- SEC EDGAR APIs: already available and free.
- Public web sources supplied per test run, such as issuer IR pages, PR
  Newswire, GlobeNewswire, Business Wire, AccessWire, or SEC pages.
- Local Ollama via `market_intelligence.llm.ollama.OllamaProvider`.
- Research-only scraper/classifier:
  `market_intelligence.collectors.public_news.collect_local_news_evidence`.

This path fetches exact page bytes, stores source hashes, extracts visible HTML
text, and asks the local LLM for a structured relevance/timing read.  It is good
for testing and Telegram research summaries.  It does not count as production
first-public proof because a scraped public page plus model interpretation is
not the same as a timestamped vendor receipt.

### 1. Start with free research messages

Use this now for daily status or research summaries:

- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Nasdaq halt RSS: https://www.nasdaqtrader.com/Trader.aspx?id=TradeHaltRSS
- PR Newswire browse/RSS: https://www.prnewswire.com/news-releases/
- GlobeNewswire newsroom: https://www.globenewswire.com/newsroom
- RTPR free wire: https://rtpr.io/

This is enough to send "research status" messages, but not enough for live
candidate alerts that claim cost-aware 2% net expectancy.

### 2. Lowest-friction API keys

Open these first:

- Alpaca signup: https://app.alpaca.markets/signup
- Alpaca market data docs: https://docs.alpaca.markets/us/docs/market-data-faq
- Databento pricing/signup: https://databento.com/pricing
- RTPR free wire / Pro trial: https://rtpr.io/
- IBKR securities lending dashboard: https://www.interactivebrokers.com/en/trading/securities-lending-dashboard.php
- IBKR Web API docs: https://www.interactivebrokers.com/campus/ibkr-api-page/web-api-trading/

### 3. Best current default call

For research-only messages:

- first_public_source: sec_plus_free_wire_pages
- realtime_market_source: alpaca_iex_or_tiingo_iex_research_only
- broker_read_only_source: none

For paper candidate messages:

- first_public_source: rtpr_pro_or_equivalent_wire_api
- realtime_market_source: databento_us_equities_mini_or_alpaca_sip
- broker_read_only_source: ibkr_client_portal_read_only

For production candidate messages:

- Use a paid or account-gated realtime market rail with full quote/spread/halts
  and historical replay.
- Use broker/borrrow data from the actual broker account.
- Keep short candidates disabled until borrow rate and availability are captured
  with immutable receipts.

## Free alternatives and limits

- SEC APIs are free and near real time, but EDGAR alone cannot prove "first
  public outside EDGAR."
- Nasdaq halt RSS is free and useful, but it is only the halt/pause rail.
- Alpaca free data gives realtime IEX only, not full SIP/NBBO.
- Tiingo free/low-cost IEX is useful for research snapshots, but IEX is still a
  partial-market view.
- PR Newswire and GlobeNewswire public pages are useful corroboration sources,
  but scraping public pages is weaker than a timestamped API receipt.
- RTPR has a free wire page and paid realtime API/dashboard; this is the most
  interesting low-cost first-public rail found so far.
- FINRA short-interest data is useful context, but it is not live borrow
  availability.
- IBKR securities lending is the most useful practical borrow source if an IBKR
  account is available.

## Config fields to fill after validation

In `config/settings.yaml`:

```yaml
relationship_decision:
  enabled: true
  first_public_source: "provider-id"
  realtime_market_source: "provider-id"
  broker_read_only_source: "provider-id"
  short_candidates_enabled: false
  kill_switch_enabled: true
```

Do not remove the hard readiness block until the provider adapters, immutable
receipts, paper replay, and shadow-review checks are wired and tested.
