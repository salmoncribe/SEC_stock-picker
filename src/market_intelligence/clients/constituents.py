"""Wikipedia "List of S&P 500 companies" client + pure point-in-time reconstruction.

Why Wikipedia: it is the only free source that publishes *both* the current S&P 500
roster and a dated history of additions/removals, which is the raw material this
platform needs to answer "who was in the index on date D?" (the survivorship-bias
defence -- see ``docs/specs/2026-07-22-signal-graph-design.md`` decision B).

Coverage -- read before trusting a backtest
--------------------------------------------
The page carries two tables:

* **Current constituents** (``id="constituents"``): one row per ticker in the index
  today, with an exact ``Date added``. This half is complete and precisely dated.
* **Selected changes** (``id="changes"``): one row per addition/removal event, dated
  to the day. This half is what makes history reconstructable at all, but it is
  community-maintained and its coverage thins with age -- entries go back
  intermittently to the 1970s-1990s, but there is **no guarantee every historical
  change is recorded**, especially before the mid-1990s.

Consequence for :func:`reconstruct_membership`: when a ticker's earliest known event
is a *removal* with no matching prior *addition* -- i.e. it was already a constituent
before this page's change history for it begins -- its true join date is unknown. We
do not guess one. We use :data:`SP500_INCEPTION_DATE` (1957-03-04, the well-documented
date the index was expanded to its current 500-company form) as a defensible floor,
and record the approximation in that window's ``note`` rather than presenting it as
exact. A caller that needs the true date for a pre-1990s membership should not treat
this platform's ``added_date`` as authoritative for such rows -- check ``note`` /
``is_estimated`` first.

A second, rarer gap: a ticker can vanish from the current table with **no** matching
removal event at all (the changes table simply never logged it) -- historically this
has happened for a same-company ticker-symbol-only change (e.g. Facebook's ticker
changing from FB to META was never logged as a constituent change, since the company
never left the index). :func:`reconstruct_membership` still emits an open window for
such a ticker (it has no evidence of a removal, so it does not invent one), but flags
it via ``note`` so a caller can treat its "still open" status with appropriate
suspicion rather than trusting it as current.

What this DOES support: a reasonably reliable point-in-time universe for backtests
concentrated in the last ~30 years, where the changes table is dense. What this does
NOT support: a guarantee of zero survivorship bias for windows dated before the
changes table's effective coverage for a given ticker, or for undocumented
same-company ticker changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup

from market_intelligence.clients.base import BaseAPIClient
from market_intelligence.config import Config, HttpConfig, RetryConfig

# Wikipedia asks for a descriptive User-Agent identifying the client and its purpose
# (https://meta.wikimedia.org/wiki/User-Agent_policy). Unlike SEC EDGAR this is not an
# access gate, so a fixed, honest string is used rather than a required credential.
DEFAULT_USER_AGENT = "market-intelligence/0.1 (local, non-commercial research tool)"

PAGE_PATH = "/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"

# The date the S&P 500 was expanded to its current 500-company form -- a documented
# historical constant, not a guess. Used only as the earliest defensible floor for a
# membership window whose true start predates the "changes" table's coverage for that
# ticker (see the module docstring).
SP500_INCEPTION_DATE = date(1957, 3, 4)

_CHANGE_DATE_FORMAT = "%B %d, %Y"


@dataclass(frozen=True)
class FetchResult:
    """A single Wikipedia fetch: the resolved URL and raw response bytes."""

    url: str
    raw: bytes


@dataclass(frozen=True)
class CurrentConstituentRow:
    """One row of the "current constituents" table -- an open membership window.

    Wikipedia's own ``Date added`` column is curated ground truth for this row: it is
    trusted directly and is never estimated.
    """

    ticker: str
    company_name: str
    cik: str | None
    added_date: date


@dataclass(frozen=True)
class ChangeRow:
    """One row of the "selected changes" table: an addition and/or removal event.

    A single row may carry both (a company replaced by another under a fresh ticker)
    or just one (a straight addition, or a removal with no same-day replacement).
    """

    effective_date: date
    added_ticker: str | None
    added_security: str | None
    removed_ticker: str | None
    removed_security: str | None


@dataclass(frozen=True)
class MembershipWindow:
    """One reconstructed point-in-time membership window.

    ``is_estimated`` marks a window whose ``added_date`` is the inception-date floor
    rather than an attested date. ``note`` explains either that approximation or a
    detected coverage gap (see the module docstring); the collector folds it into the
    persisted record's ``validation_errors`` rather than inventing a new column.
    """

    ticker: str
    company_name: str | None
    cik: str | None
    added_date: date
    removed_date: date | None
    is_estimated: bool
    note: str | None = None


def _cell_text(cell: object) -> str:
    """Text of a table cell with citation markers (``<sup>...</sup>``) stripped.

    Wikipedia inlines footnote markers as ``<sup class="reference">[5]</sup>``; left
    in place, ``get_text()`` would fold them into the visible text (e.g. a "Reason"
    cell reading "Market capitalization change. [ 5 ]"). Decomposing them first keeps
    the extracted text exactly what a reader sees before the footnote mark.
    """
    for sup in cell.find_all("sup"):  # type: ignore[attr-defined]
        sup.decompose()
    return cell.get_text(" ", strip=True)  # type: ignore[attr-defined]


def _parse_iso_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_change_date(text: str) -> date | None:
    try:
        return datetime.strptime(text, _CHANGE_DATE_FORMAT).date()
    except ValueError:
        return None


def parse_current_constituents(html: bytes) -> list[CurrentConstituentRow]:
    """Parse the "current constituents" table (``id="constituents"``).

    Column order (fixed by inspection of the live page): Symbol, Security, GICS
    Sector, GICS Sub-Industry, Headquarters Location, Date added, CIK, Founded. Only
    the columns this platform stores are read; sector/sub-industry/headquarters/
    founded are not part of ``ConstituentRecord`` and are discarded here rather than
    threaded through unused.

    A row is skipped -- not defaulted -- when its ticker or ``Date added`` cell is
    missing or unparseable: on the live page this has not happened in practice (every
    row carries a strict ``YYYY-MM-DD`` date), but a caller must not silently receive
    a fabricated date for a row the source itself left ambiguous.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="constituents")
    if table is None:
        raise ValueError('expected a table with id="constituents" on the S&P 500 page')

    rows: list[CurrentConstituentRow] = []
    for tr in table.find_all("tr")[1:]:  # skip the single header row
        cells = tr.find_all(["td", "th"])
        if len(cells) < 7:
            continue  # malformed row; skip rather than guess column meaning
        ticker = _cell_text(cells[0])
        company_name = _cell_text(cells[1])
        added = _parse_iso_date(_cell_text(cells[5]))
        cik = _cell_text(cells[6]) or None
        if not ticker or added is None:
            continue
        rows.append(
            CurrentConstituentRow(
                ticker=ticker, company_name=company_name, cik=cik, added_date=added
            )
        )
    return rows


def parse_changes(html: bytes) -> list[ChangeRow]:
    """Parse the "selected changes" table (``id="changes"``).

    Column order: Effective Date, Added Ticker, Added Security, Removed Ticker,
    Removed Security, Reason (two physical header rows: the grouped headers and the
    Ticker/Security sub-headers). The Reason column is prose, not reconstructable
    state, and is dropped.

    A row whose date cannot be parsed is skipped rather than guessed -- see the
    module docstring: :func:`reconstruct_membership` never invents a date, so an
    unparseable event is safest treated as absent rather than mis-sequenced.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="changes")
    if table is None:
        raise ValueError('expected a table with id="changes" on the S&P 500 page')

    result: list[ChangeRow] = []
    for tr in table.find_all("tr")[2:]:  # skip the two header rows
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        effective = _parse_change_date(_cell_text(cells[0]))
        if effective is None:
            continue
        added_ticker = _cell_text(cells[1]) or None
        added_security = _cell_text(cells[2]) or None
        removed_ticker = _cell_text(cells[3]) or None
        removed_security = _cell_text(cells[4]) or None
        result.append(
            ChangeRow(
                effective_date=effective,
                added_ticker=added_ticker,
                added_security=added_security,
                removed_ticker=removed_ticker,
                removed_security=removed_security,
            )
        )
    return result


def reconstruct_membership(
    current_rows: list[CurrentConstituentRow],
    changes: list[ChangeRow],
) -> list[MembershipWindow]:
    """Reconstruct point-in-time membership windows for every known ticker.

    Per ticker, its "added"/"removed" events (from ``changes``) are walked in
    chronological order, alternating open -> close -> open into zero or more CLOSED
    windows. On a date tie -- one row naming the same ticker as both added and
    removed, i.e. one company replaced by another under an unchanged ticker (e.g.
    FOXA: 21st Century Fox -> Fox Corporation) -- the "removed" side is applied first
    so the old window closes before the new one opens, rather than producing a
    zero-length window.

    The final, OPEN window for a ticker present in the current table always uses that
    table's own ``added_date`` (Wikipedia's curated ground truth), not a trailing
    "added" event from ``changes``: the two occasionally disagree by a few days (S&P's
    announcement date vs. its effective date), and the current table is authoritative
    for "is this ticker a member today".

    A ticker with a dangling, unmatched "added" event and no current-table row (see
    the module docstring's FB/META example) still gets an open window -- no removal is
    invented -- but it carries a ``note`` flagging the gap.

    Two "added" events for the same ticker with no "removed" between them is an
    internally inconsistent source row; the earlier is kept and the duplicate is
    dropped rather than guessed at. This has not occurred on the live page.
    """
    current_by_ticker = {row.ticker: row for row in current_rows}

    # events[ticker] -> [(date, tie_priority, kind, security_name), ...]. priority 0
    # (removed) sorts before priority 1 (added) on a date tie -- see docstring.
    events: dict[str, list[tuple[date, int, str, str | None]]] = {}
    for change in changes:
        if change.added_ticker:
            events.setdefault(change.added_ticker, []).append(
                (change.effective_date, 1, "added", change.added_security)
            )
        if change.removed_ticker:
            events.setdefault(change.removed_ticker, []).append(
                (change.effective_date, 0, "removed", change.removed_security)
            )
    for ticker_events in events.values():
        ticker_events.sort(key=lambda e: (e[0], e[1]))

    all_tickers = sorted(set(current_by_ticker) | set(events))
    windows: list[MembershipWindow] = []

    for ticker in all_tickers:
        open_added: date | None = None
        open_name: str | None = None

        for effective_date, _, kind, name in events.get(ticker, []):
            if kind == "added":
                if open_added is not None:
                    continue  # duplicate add with no intervening removal: keep earliest
                open_added, open_name = effective_date, name
                continue

            if open_added is None:
                windows.append(
                    MembershipWindow(
                        ticker=ticker,
                        company_name=name,
                        cik=None,
                        added_date=SP500_INCEPTION_DATE,
                        removed_date=effective_date,
                        is_estimated=True,
                        note=(
                            f"added_date is a floor, not an attested date: {ticker} was "
                            "already an S&P 500 constituent before the earliest change "
                            "Wikipedia's changes table records for it; "
                            f"{SP500_INCEPTION_DATE.isoformat()} (the index's inception "
                            "date in its current 500-company form) is used as the "
                            "earliest defensible bound, not the true join date."
                        ),
                    )
                )
            else:
                windows.append(
                    MembershipWindow(
                        ticker=ticker,
                        company_name=open_name,
                        cik=None,
                        added_date=open_added,
                        removed_date=effective_date,
                        is_estimated=False,
                    )
                )
                open_added, open_name = None, None

        current = current_by_ticker.get(ticker)
        if current is not None:
            windows.append(
                MembershipWindow(
                    ticker=ticker,
                    company_name=current.company_name,
                    cik=current.cik,
                    added_date=current.added_date,
                    removed_date=None,
                    is_estimated=False,
                )
            )
        elif open_added is not None:
            windows.append(
                MembershipWindow(
                    ticker=ticker,
                    company_name=open_name,
                    cik=None,
                    added_date=open_added,
                    removed_date=None,
                    is_estimated=False,
                    note=(
                        f"{ticker} has no recorded removal after {open_added.isoformat()} "
                        "but is absent from the current constituents table; the changes "
                        "table likely never logged its departure (e.g. a same-company, "
                        "ticker-only rename). Treat this window's open status with "
                        "caution rather than as a confirmed current membership."
                    ),
                )
            )

    return windows


class ConstituentsClient(BaseAPIClient):
    """Fetches the Wikipedia "List of S&P 500 companies" page."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        http_config: HttpConfig | None = None,
        retry_config: RetryConfig | None = None,
        requests_per_second: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=WIKIPEDIA_BASE_URL,
            headers={"User-Agent": user_agent},
            http_config=http_config,
            retry_config=retry_config,
            requests_per_second=requests_per_second,
            transport=transport,
        )

    def __enter__(self) -> ConstituentsClient:
        return self

    @classmethod
    def from_config(
        cls, config: Config, *, transport: httpx.BaseTransport | None = None
    ) -> ConstituentsClient:
        """Construct from the platform :class:`Config` (shared HTTP/retry settings)."""
        return cls(
            http_config=config.settings.http,
            retry_config=config.settings.retry,
            transport=transport,
        )

    def fetch_page(self) -> FetchResult:
        """Fetch the raw page bytes, preserved verbatim by the caller."""
        response = self.request("GET", PAGE_PATH)
        return FetchResult(url=str(response.request.url), raw=response.content)


__all__ = [
    "DEFAULT_USER_AGENT",
    "SP500_INCEPTION_DATE",
    "ChangeRow",
    "ConstituentsClient",
    "CurrentConstituentRow",
    "FetchResult",
    "MembershipWindow",
    "parse_changes",
    "parse_current_constituents",
    "reconstruct_membership",
]
