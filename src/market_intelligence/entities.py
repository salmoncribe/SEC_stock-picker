"""Resolve a company *name* from a filing to a CIK/ticker we track -- or refuse.

The extractor hands us a target as a raw string ("Under Armour", "Columbia",
"adidas"). Turning that string into a ticker is what lets an edge be validated,
but a *wrong* resolution is uniquely expensive: it does not merely drop an edge,
it invents a relationship between two real companies and can fire a false alert
on a price series that has nothing to do with the filer. A missed edge costs one
observation; a wrong edge poisons the graph. So every rule here tilts the same
way -- when the name does not land on exactly one company, we say so rather than
guess.

Three outcomes, mirroring :class:`ResolutionStatus`. A unique exact match on the
normalized name is ``RESOLVED``. A normalized name shared by more than one
company ("Columbia" -> both a bank and a sportswear maker) is ``AMBIGUOUS`` and
resolves to nothing. Everything else -- foreign, private, or simply absent from
our ~8000-row universe -- is ``UNRESOLVED``. A conservative token-subset
fallback may promote a near-miss to ``RESOLVED`` only when the query carries two
or more tokens *and* lands on exactly one company; a lone generic token ("Puma",
"Meta") is never resolved on a subset, because such words collide with unrelated
real tickers, and inventing that edge is the failure we most want to avoid.

Normalization is the whole game: legal suffixes (Inc., Corp., Ltd., N.V., ...)
and punctuation carry no identity, so both the query and every company name are
reduced to the same bare token string ("V.F. Corporation" and "VF CORP" both
become ``"VF"``) before anything is compared. A company with no ticker cannot be
a usable target and is dropped from the index entirely, so it can never be
returned -- not even as the reason an otherwise-unique name is called ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from market_intelligence.schemas.edges import ResolutionStatus

#: Confidence attached to a unique exact normalized match -- the only case we
#: trust completely.
_EXACT_CONFIDENCE = 1.0

#: Confidence for a multi-token subset match: real, but a normalization gap we
#: could not close by suffix stripping alone, so it is deliberately not 1.0.
_SUBSET_CONFIDENCE = 0.8

#: Minimum query tokens before a subset match may *resolve*. One generic token
#: ("Puma", "Delta", "Meta") too easily lands on an unrelated real ticker; two
#: or more together are distinctive enough to trust when unique.
_MIN_SUBSET_TOKENS = 2

#: Intra-word marks deleted outright so "V.F." collapses to "VF" rather than
#: splitting into two tokens.
_DROP_RE = re.compile(r"[.']")

#: Every other run of non-alphanumeric characters becomes a single separator,
#: so commas, hyphens, slashes, and ampersands all break tokens uniformly.
_SEP_RE = re.compile(r"[^A-Z0-9]+")

#: Trailing legal-form tokens that carry no identity. Stripped from the *end*
#: only (and repeatedly), so a meaningful interior "Group" or "Holdings"
#: survives while "... Holdings Corp" reduces to its distinctive stem.
_LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COS",
        "COMPANY",
        "COMPANIES",
        "LTD",
        "LIMITED",
        "LLC",
        "LLP",
        "LP",
        "PLC",
        "PC",
        "HOLDING",
        "HOLDINGS",
        "GROUP",
        "GRP",
        "TRUST",
        "NV",
        "SA",
        "SE",
        "AG",
        "AB",
        "AS",
        "ASA",
        "BV",
        "GMBH",
        "SPA",
        "OYJ",
        "KGAA",
    }
)

#: Leading article, stripped so "The Coca-Cola Company" and "Coca-Cola Co" agree.
_LEADING_NOISE: frozenset[str] = frozenset({"THE"})


def normalize_company_name(name: str) -> str:
    """Reduce a company name to a bare, comparable token string.

    Uppercase; delete intra-word marks so ``V.F.`` becomes ``VF``; turn every
    other punctuation run into a separator; then strip a leading article and any
    run of trailing legal suffixes. The query and every stored name pass through
    this exact function, so equality of the result is the only thing "same
    company" ever means here. Returns ``""`` for blank or all-suffix input,
    which the caller treats as "not a usable name."
    """
    dropped = _DROP_RE.sub("", name.upper())
    tokens = _SEP_RE.sub(" ", dropped).split()

    start = 0
    while start < len(tokens) and tokens[start] in _LEADING_NOISE:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] in _LEGAL_SUFFIXES:
        end -= 1

    return " ".join(tokens[start:end])


@dataclass(frozen=True)
class Resolution:
    """The verdict for one name lookup.

    ``status`` is a :class:`ResolutionStatus` value. ``cik``/``ticker`` are set
    only on ``RESOLVED``; ``AMBIGUOUS`` and ``UNRESOLVED`` leave them ``None`` by
    construction, since the whole point is that no single company was chosen.
    ``matched_name`` echoes the stored company name we matched, for auditing.
    """

    status: str
    cik: str | None
    ticker: str | None
    confidence: float
    matched_name: str | None


@dataclass(frozen=True)
class _Candidate:
    """A ticker-bearing company that a query may resolve to."""

    cik: str
    ticker: str
    name: str


def _unresolved() -> Resolution:
    return Resolution(ResolutionStatus.UNRESOLVED.value, None, None, 0.0, None)


def _ambiguous() -> Resolution:
    return Resolution(ResolutionStatus.AMBIGUOUS.value, None, None, 0.0, None)


class CompanyIndex:
    """Normalized lookup over the ``companies`` table, built once and reused.

    Two structures cover the two match kinds: ``_exact`` maps a normalized name
    to its candidates for the common unique/ambiguous exact case, and
    ``_postings`` inverts tokens to candidate indices so a subset query is an
    intersection of posting lists rather than a scan. Rows with a null or blank
    ticker, or whose name normalizes to nothing, are dropped at build time and
    therefore never surface in any answer.
    """

    def __init__(self, companies: list[tuple[str, str | None, str]]) -> None:
        self._candidates: list[_Candidate] = []
        self._exact: dict[str, list[_Candidate]] = {}
        self._postings: dict[str, set[int]] = {}

        for cik, ticker, name in companies:
            if ticker is None or not ticker.strip():
                # No ticker means no price series: unusable as a target, and
                # dropped so it cannot even make a unique name look ambiguous.
                continue
            normalized = normalize_company_name(name)
            if not normalized:
                continue

            candidate = _Candidate(cik=cik, ticker=ticker, name=name)
            index = len(self._candidates)
            self._candidates.append(candidate)
            self._exact.setdefault(normalized, []).append(candidate)
            for token in set(normalized.split()):
                self._postings.setdefault(token, set()).add(index)

    @classmethod
    def from_connection(cls, con: duckdb.DuckDBPyConnection) -> CompanyIndex:
        """Build the index from the ``companies`` table -- a thin read only."""
        rows = con.execute("SELECT cik, ticker, company_name FROM companies").fetchall()
        companies: list[tuple[str, str | None, str]] = [
            (str(cik), None if ticker is None else str(ticker), "" if name is None else str(name))
            for cik, ticker, name in rows
        ]
        return cls(companies)

    def resolve(self, name: str) -> Resolution:
        """Resolve a filing-supplied name, refusing rather than guessing.

        Exact unique -> ``RESOLVED``; exact but shared -> ``AMBIGUOUS``. With no
        exact match, a subset match resolves only when unique *and* backed by two
        or more query tokens; multiple subset candidates are ``AMBIGUOUS``, and
        anything weaker is ``UNRESOLVED``.
        """
        normalized = normalize_company_name(name)
        if not normalized:
            return _unresolved()

        exact = self._exact.get(normalized)
        if exact:
            return self._decide(exact, _EXACT_CONFIDENCE, require_tokens=1, tokens=1)

        tokens = normalized.split()
        candidates = [self._candidates[i] for i in self._subset_candidates(tokens)]
        return self._decide(
            candidates, _SUBSET_CONFIDENCE, require_tokens=_MIN_SUBSET_TOKENS, tokens=len(tokens)
        )

    def _subset_candidates(self, tokens: list[str]) -> set[int]:
        """Indices of companies whose token set contains every query token.

        Computed as the intersection of the tokens' posting lists; a query token
        absent from the index empties the result immediately, so only genuine
        supersets survive.
        """
        postings: list[set[int]] = []
        for token in tokens:
            bucket = self._postings.get(token)
            if not bucket:
                return set()
            postings.append(bucket)
        return set.intersection(*postings)

    @staticmethod
    def _decide(
        candidates: list[_Candidate],
        confidence: float,
        *,
        require_tokens: int,
        tokens: int,
    ) -> Resolution:
        """Turn a candidate list into a verdict under the conservatism rules.

        More than one distinct company is always ``AMBIGUOUS``. A single company
        resolves only when the query is distinctive enough (``tokens >=
        require_tokens``); otherwise it is left ``UNRESOLVED`` rather than betting
        on a lone generic token.
        """
        distinct = {c.cik: c for c in candidates}
        if len(distinct) > 1:
            return _ambiguous()
        if len(distinct) == 1 and tokens >= require_tokens:
            match = next(iter(distinct.values()))
            return Resolution(
                ResolutionStatus.RESOLVED.value,
                match.cik,
                match.ticker,
                confidence,
                match.name,
            )
        return _unresolved()


__all__ = [
    "CompanyIndex",
    "Resolution",
    "normalize_company_name",
]
