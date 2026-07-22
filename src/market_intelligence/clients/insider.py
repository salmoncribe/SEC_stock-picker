"""SEC pre-parsed quarterly Form 3/4/5 insider-transaction dataset: fetch + parse.

SEC publishes one ZIP per calendar quarter containing every Form 3/4/5 filed in
that quarter, already split into TSVs -- no XML parsing, no LLM, and no
per-filing HTTP request (the "why Form 4 comes before LLM extraction" section
of ``docs/specs/2026-07-22-signal-graph-design.md`` §9 covers why this is the
first event source built).

Fetching and parsing are kept strictly separate, mirroring
``extraction/sections.py``: :class:`InsiderClient` does network I/O only and
hands back raw bytes; :func:`parse_quarter_zip` and its helpers are pure
functions of those bytes (no clock, no I/O, no randomness) so they can be unit
tested against small in-memory ZIP fixtures without a collector or a database.

**Dates.** SEC ships every date in this dataset as ``DD-MON-YYYY`` (e.g.
``31-MAR-2025``), not ISO. :func:`parse_sec_date` parses this format
explicitly and returns ``None`` on anything that does not match -- never a
best-effort guess -- so a malformed date surfaces as "unknown", to be rejected
by the caller, rather than silently becoming a wrong date or defaulting to
"today".
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date

import httpx

from market_intelligence.clients.base import BaseAPIClient
from market_intelligence.config import Config, HttpConfig, RetryConfig
from market_intelligence.schemas.sec import normalize_cik

# SEC's quarterly archive path. Coverage is confirmed 2006q1 through 2026q1 as
# of this module's writing; a 404 for a recent quarter means "not published
# yet" (see QuarterNotPublished), not "the endpoint moved".
QUARTER_PATH_TEMPLATE = (
    "/files/structureddata/data/insider-transactions-data-sets/{quarter}_form345.zip"
)

# The three TSVs this collector reads out of each quarter's ZIP. SEC also
# ships DERIV_TRANS.tsv, OWNER_SIGNATURE.tsv, and others; out of scope here --
# only non-derivative transactions (NONDERIV_TRANS) feed the event graph.
SUBMISSION_FILE = "SUBMISSION.tsv"
NONDERIV_TRANS_FILE = "NONDERIV_TRANS.tsv"
REPORTINGOWNER_FILE = "REPORTINGOWNER.tsv"

_QUARTER_RE = re.compile(r"^(?P<year>\d{4})q(?P<q>[1-4])$", re.IGNORECASE)

# Explicit English month abbreviations. See parse_sec_date for why this is not
# delegated to strptime's locale-dependent %b.
_MONTH_ABBREVIATIONS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}  # fmt: skip


class QuarterNotPublished(Exception):
    """SEC has not yet published the quarterly dataset for this quarter.

    Raised on an HTTP 404 from the archive endpoint. SEC publishes each
    quarter's ZIP with a lag of days to a few weeks after the quarter closes,
    so a 404 for the most recent one or two quarters is the expected steady
    state, not a failure -- distinct from ``httpx.HTTPStatusError`` on any
    other status, which is left to propagate as a real error.
    """


@dataclass(frozen=True)
class QuarterFetchResult:
    """One quarter's fetched ZIP: the resolved URL and the raw bytes."""

    quarter: str
    url: str
    raw: bytes


@dataclass(frozen=True)
class InsiderTransactionRow:
    """One non-derivative transaction, joined against its filing and owner.

    One row per ``NONDERIV_TRANS.tsv`` record. ``trans_sk`` is SEC's own
    per-transaction surrogate key (``NONDERIV_TRANS_SK``) -- the producer's
    ``event_key``. All numeric/date fields are already coerced to their typed
    form (or ``None`` when absent/unparseable); nothing downstream needs to
    re-parse SEC's raw string formats.
    """

    accession_number: str
    filing_date: date | None
    period_of_report: date | None
    document_type: str | None
    issuer_cik: str | None
    issuer_name: str | None
    issuer_trading_symbol: str | None
    trans_sk: str | None
    security_title: str | None
    trans_date: date | None
    trans_code: str | None
    trans_shares: float | None
    trans_price_per_share: float | None
    trans_acquired_disp_cd: str | None
    shares_owned_following_transaction: float | None
    direct_indirect_ownership: str | None
    owner_cik: str | None
    owner_name: str | None
    owner_relationship: str | None
    owner_title: str | None


# --------------------------------------------------------------------------- #
# pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def _parse_quarter_id(value: str) -> tuple[int, int]:
    match = _QUARTER_RE.match(value.strip().lower())
    if not match:
        raise ValueError(f"malformed quarter id: {value!r} (expected e.g. '2016q3')")
    return int(match.group("year")), int(match.group("q"))


def quarters_between(start: str, end: str) -> list[str]:
    """Every quarter id from ``start`` to ``end`` inclusive, chronological order.

    Pure and total for well-formed input. ``start`` after ``end`` yields an
    empty list rather than raising, so a caller can pass a possibly-inverted
    range without special-casing it.
    """
    start_year, start_q = _parse_quarter_id(start)
    end_year, end_q = _parse_quarter_id(end)

    start_index = start_year * 4 + (start_q - 1)
    end_index = end_year * 4 + (end_q - 1)

    result: list[str] = []
    for index in range(start_index, end_index + 1):
        year, zero_based_q = divmod(index, 4)
        result.append(f"{year}q{zero_based_q + 1}")
    return result


def quarter_for_date(day: date) -> str:
    """The quarter id (e.g. ``"2026q3"``) containing ``day``."""
    return f"{day.year}q{(day.month - 1) // 3 + 1}"


def parse_sec_date(value: str | None) -> date | None:
    """Parse SEC's ``DD-MON-YYYY`` date format (e.g. ``31-MAR-2025``).

    The month map is explicit rather than using ``strptime("%d-%b-%Y")``,
    because ``%b`` resolves month names through the process locale. SEC always
    publishes English abbreviations, so under ``LC_TIME=de_DE`` or any other
    non-English locale every date in the dataset would fail to parse, every
    record would be rejected for a missing clock, and the run would report
    success having stored nothing. The wire format is a fixed property of the
    source; it must not be interpreted through a runtime setting.

    Returns ``None`` on any empty or unparseable input -- explicitly, not by
    letting a wrong format silently produce a wrong or defaulted date. A
    caller must treat ``None`` as "this row's timing is unknown", never as
    "assume today" or "assume the first of the month" (see
    ``validators.events`` for how an unusable clock is rejected downstream).
    """
    if not value:
        return None
    parts = value.strip().upper().split("-")
    if len(parts) != 3:
        return None

    day_text, month_text, year_text = parts
    month = _MONTH_ABBREVIATIONS.get(month_text)
    if month is None:
        return None
    try:
        return date(int(year_text), month, int(day_text))
    except ValueError:
        # Out-of-range day/year (e.g. 31-FEB-2025) -- unparseable, not defaulted.
        return None


def parse_sec_float(value: str | None) -> float | None:
    """Parse a numeric TSV cell, returning ``None`` for empty/unparseable input."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_normalize_cik(value: str | None) -> str | None:
    """``normalize_cik`` that returns ``None`` instead of raising.

    Coerces to the same canonical 10-digit zero-padded form ``companies.cik``
    and ``filings.cik`` use, which is what makes the universe join in
    ``collectors/insider.py`` line up at all -- SEC's bulk TSVs store CIKs
    unpadded (e.g. ``"1045810"``), and comparing that directly against a
    zero-padded key would silently match nothing.
    """
    if not value or not value.strip():
        return None
    try:
        return normalize_cik(value)
    except ValueError:
        return None


def _clean(value: str | None) -> str | None:
    """Strip a TSV cell and fold an empty result to ``None``."""
    if value is None:
        return None
    text = value.strip()
    return text or None


def _clean_upper(value: str | None) -> str | None:
    """``_clean`` plus upper-casing, for codes/symbols compared case-insensitively
    elsewhere (``TRANS_CODE``, ``TRANS_ACQUIRED_DISP_CD``, ticker symbols).
    """
    cleaned = _clean(value)
    return cleaned.upper() if cleaned is not None else None


def _read_tsv_rows(zf: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    """Read one member of the ZIP as a TSV, matched case-insensitively.

    SEC has shipped these archives with inconsistent member-name casing across
    the ~80 quarters this collector spans; matching case-insensitively avoids
    silently returning zero rows for an older quarter. Returns ``[]`` (not an
    error) when the member is absent, since an empty owner or transaction
    table is a legitimate, if unlikely, quarter shape.
    """
    lowered = filename.lower()
    member = next((name for name in zf.namelist() if name.lower() == lowered), None)
    if member is None:
        return []
    with zf.open(member) as handle:
        text = handle.read().decode("latin-1")
    # Normalize line endings before csv sees them: SEC ships CRLF, and a
    # stray \r left in the final column of a row would corrupt that value.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return list(reader)


def parse_quarter_zip(zip_bytes: bytes) -> list[InsiderTransactionRow]:
    """Parse one quarter's ZIP into joined, typed transaction rows.

    Joins ``SUBMISSION``, ``NONDERIV_TRANS``, and ``REPORTINGOWNER`` on
    ``ACCESSION_NUMBER``. One output row per ``NONDERIV_TRANS`` row -- a
    filing reporting N transactions produces N rows sharing the same filing
    and owner context, and two transactions in the same filing with different
    ``TRANS_CODE`` values (e.g. a purchase and a grant) become two separate
    rows here, never merged.

    A transaction whose ``ACCESSION_NUMBER`` has no matching ``SUBMISSION``
    row is dropped: ``SUBMISSION`` is the anchor for ``FILING_DATE`` and
    ``ISSUERCIK``, and a transaction that cannot be dated or attributed to an
    issuer cannot become a usable event.

    A filing with more than one ``REPORTINGOWNER`` row (a joint filing, e.g.
    an officer and a trust they control filing together) is attributed to the
    *first* owner row in file order. ``NONDERIV_TRANS`` carries no
    owner-specific key to split on, so representing a joint filing as more
    than one event per transaction is not possible from this data alone; this
    is a documented simplification, not a data-quality gap.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        submissions = _read_tsv_rows(zf, SUBMISSION_FILE)
        transactions = _read_tsv_rows(zf, NONDERIV_TRANS_FILE)
        owners = _read_tsv_rows(zf, REPORTINGOWNER_FILE)

    submission_by_accession = {
        row["ACCESSION_NUMBER"]: row for row in submissions if row.get("ACCESSION_NUMBER")
    }

    owner_by_accession: dict[str, dict[str, str]] = {}
    for row in owners:
        accession = row.get("ACCESSION_NUMBER")
        if accession and accession not in owner_by_accession:
            owner_by_accession[accession] = row

    result: list[InsiderTransactionRow] = []
    for trans in transactions:
        accession = trans.get("ACCESSION_NUMBER")
        if not accession:
            continue
        submission = submission_by_accession.get(accession)
        if submission is None:
            continue

        owner = owner_by_accession.get(accession, {})

        result.append(
            InsiderTransactionRow(
                accession_number=accession,
                filing_date=parse_sec_date(submission.get("FILING_DATE")),
                period_of_report=parse_sec_date(submission.get("PERIOD_OF_REPORT")),
                document_type=_clean(submission.get("DOCUMENT_TYPE")),
                issuer_cik=_safe_normalize_cik(submission.get("ISSUERCIK")),
                issuer_name=_clean(submission.get("ISSUERNAME")),
                issuer_trading_symbol=_clean_upper(submission.get("ISSUERTRADINGSYMBOL")),
                trans_sk=_clean(trans.get("NONDERIV_TRANS_SK")),
                security_title=_clean(trans.get("SECURITY_TITLE")),
                trans_date=parse_sec_date(trans.get("TRANS_DATE")),
                trans_code=_clean_upper(trans.get("TRANS_CODE")),
                trans_shares=parse_sec_float(trans.get("TRANS_SHARES")),
                trans_price_per_share=parse_sec_float(trans.get("TRANS_PRICEPERSHARE")),
                trans_acquired_disp_cd=_clean_upper(trans.get("TRANS_ACQUIRED_DISP_CD")),
                shares_owned_following_transaction=parse_sec_float(
                    trans.get("SHRS_OWND_FOLWNG_TRANS")
                ),
                direct_indirect_ownership=_clean(trans.get("DIRECT_INDIRECT_OWNERSHIP")),
                owner_cik=_safe_normalize_cik(owner.get("RPTOWNERCIK")),
                owner_name=_clean(owner.get("RPTOWNERNAME")),
                owner_relationship=_clean(owner.get("RPTOWNER_RELATIONSHIP")),
                owner_title=_clean(owner.get("RPTOWNER_TITLE")),
            )
        )
    return result


# --------------------------------------------------------------------------- #
# client                                                                       #
# --------------------------------------------------------------------------- #
class InsiderClient(BaseAPIClient):
    """Fetches SEC's quarterly pre-parsed Form 3/4/5 insider-transaction ZIPs."""

    def __init__(
        self,
        *,
        user_agent: str,
        base_url: str,
        http_config: HttpConfig | None = None,
        retry_config: RetryConfig | None = None,
        requests_per_second: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            http_config=http_config,
            retry_config=retry_config,
            requests_per_second=requests_per_second,
            transport=transport,
        )

    def __enter__(self) -> InsiderClient:
        return self

    @classmethod
    def from_config(
        cls, config: Config, *, transport: httpx.BaseTransport | None = None
    ) -> InsiderClient:
        """Construct from the platform :class:`Config` (SEC User-Agent + pacing)."""
        return cls(
            user_agent=config.require_sec_user_agent(),
            base_url=config.settings.sec.base_url,
            http_config=config.settings.http,
            retry_config=config.settings.retry,
            requests_per_second=config.settings.pacing.sec.requests_per_second,
            transport=transport,
        )

    def fetch_quarter(self, quarter: str) -> QuarterFetchResult:
        """Fetch one quarter's Form 3/4/5 ZIP, verbatim.

        Raises :class:`QuarterNotPublished` on an HTTP 404 -- see that
        exception's docstring. Any other HTTP error propagates as-is.
        """
        url = QUARTER_PATH_TEMPLATE.format(quarter=quarter)
        try:
            response = self.request("GET", url)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise QuarterNotPublished(f"{quarter}: not yet published at {url}") from exc
            raise
        return QuarterFetchResult(
            quarter=quarter, url=str(response.request.url), raw=response.content
        )


__all__ = [
    "NONDERIV_TRANS_FILE",
    "QUARTER_PATH_TEMPLATE",
    "REPORTINGOWNER_FILE",
    "SUBMISSION_FILE",
    "InsiderClient",
    "InsiderTransactionRow",
    "QuarterFetchResult",
    "QuarterNotPublished",
    "parse_quarter_zip",
    "parse_sec_date",
    "parse_sec_float",
    "quarter_for_date",
    "quarters_between",
]
