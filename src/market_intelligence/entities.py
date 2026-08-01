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

import json
import re
from dataclasses import dataclass

import duckdb

from market_intelligence import hashing
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


# --------------------------------------------------------------------------- #
# People (docs/specs/2026-07-25-people-insider-graph-design.md, Decision A)     #
# --------------------------------------------------------------------------- #
#: Intra-word marks / separators reused from the company normalizer -- person
#: names carry the same casing/punctuation noise, just none of the legal-form
#: suffixes, so the suffix-stripping half of that pipeline does not apply here.
def normalize_person_name(name: str) -> str:
    """Reduce a person's name to a bare, comparable token string.

    Unlike :func:`normalize_company_name`, this strips no trailing "suffix"
    tokens -- a person's name has no legal form to strip, and guessing that a
    trailing word is noise (rather than, say, a real surname) is exactly the
    kind of invented rule this module avoids. Only case, punctuation, and
    whitespace are normalized, so "Doe, Jane" and "DOE, JANE" agree but
    "Doe, Jane" and "Jane Doe" deliberately do not -- SEC's own filer-name
    field orders names inconsistently across filings, and folding both orders
    together risks merging two different real people who happen to share a
    surname and a given name in swapped fields.
    """
    dropped = _DROP_RE.sub("", name.upper())
    tokens = _SEP_RE.sub(" ", dropped).split()
    return " ".join(tokens)


def person_id_for_cik(reporting_owner_cik: str) -> str:
    """Derive the stable ``person_id`` for a normalized SEC reporting-owner CIK.

    Pure and deterministic, so any producer that already knows a CIK --
    ``collectors/roles.py``, ``analytics/cluster_buy.py`` -- computes the same
    id without a database round trip, and both always agree on identity. This
    is the one function that owns that derivation; nothing else should hash a
    person id independently.
    """
    return hashing.content_hash("person", reporting_owner_cik)


@dataclass(frozen=True)
class PersonResolution:
    """The verdict for one person lookup -- mirrors :class:`Resolution`.

    ``status`` is a :class:`ResolutionStatus` value. ``person_id`` /
    ``reporting_owner_cik`` / ``canonical_name`` are set only on ``RESOLVED``.
    """

    status: str
    person_id: str | None
    reporting_owner_cik: str | None
    canonical_name: str | None


def _person_unresolved() -> PersonResolution:
    return PersonResolution(ResolutionStatus.UNRESOLVED.value, None, None, None)


def _person_ambiguous() -> PersonResolution:
    return PersonResolution(ResolutionStatus.AMBIGUOUS.value, None, None, None)


class PersonIndex:
    """Normalized lookup over the ``people`` table, built once and reused.

    Identity is keyed on ``reporting_owner_cik`` (Decision A) -- a persistent
    SEC identifier already present on every Form 3/4/5 row this platform
    stores, so :meth:`resolve_cik` is an exact lookup with no ambiguity
    possible: two different people never share a CIK. :meth:`resolve_name`
    exists for the one case a CIK cannot cover -- a name observed with no CIK
    attached (the deferred Phase 4 8-K executive-change case) -- and mirrors
    :class:`CompanyIndex`'s conservatism: a normalized name shared by more
    than one known person is ``AMBIGUOUS`` and resolves to nothing, never a
    guess, because inventing a match invents a person's involvement in a
    company they may have nothing to do with.
    """

    def __init__(
        self, people: list[tuple[str, str, str | None, list[str]]]
    ) -> None:
        """Build from ``(person_id, reporting_owner_cik, canonical_name, name_variants)`` rows."""
        self._by_cik: dict[str, tuple[str, str | None]] = {}
        self._by_name: dict[str, list[tuple[str, str]]] = {}

        for person_id, cik, canonical_name, variants in people:
            self._by_cik[cik] = (person_id, canonical_name)
            names = {n for n in (canonical_name, *variants) if n}
            for name in names:
                normalized = normalize_person_name(name)
                if not normalized:
                    continue
                self._by_name.setdefault(normalized, []).append((person_id, cik))

    @classmethod
    def from_connection(cls, con: duckdb.DuckDBPyConnection) -> PersonIndex:
        """Build the index from the ``people`` table -- a thin read only."""
        rows = con.execute(
            "SELECT person_id, reporting_owner_cik, canonical_name, name_variants FROM people"
        ).fetchall()
        people: list[tuple[str, str, str | None, list[str]]] = []
        for person_id, cik, canonical_name, variants_json in rows:
            variants = json.loads(variants_json) if variants_json else []
            people.append((str(person_id), str(cik), canonical_name, list(variants)))
        return cls(people)

    def resolve_cik(self, reporting_owner_cik: str) -> PersonResolution:
        """Exact lookup by the persistent SEC identity key. Never ambiguous."""
        hit = self._by_cik.get(reporting_owner_cik)
        if hit is None:
            return _person_unresolved()
        person_id, canonical_name = hit
        return PersonResolution(
            ResolutionStatus.RESOLVED.value, person_id, reporting_owner_cik, canonical_name
        )

    def resolve_name(self, name: str) -> PersonResolution:
        """Resolve a bare name, refusing rather than guessing.

        Exact normalized match on a single known person -> ``RESOLVED``; the
        same normalized name shared by more than one distinct CIK ->
        ``AMBIGUOUS``. No subset/token fallback -- unlike a company name, a
        person's name has no distinguishing legal-suffix noise to explain away
        a partial match, so anything short of an exact hit is left
        ``UNRESOLVED`` rather than risked.
        """
        normalized = normalize_person_name(name)
        if not normalized:
            return _person_unresolved()
        candidates = self._by_name.get(normalized)
        if not candidates:
            return _person_unresolved()
        distinct_ciks = {cik for _, cik in candidates}
        if len(distinct_ciks) > 1:
            return _person_ambiguous()
        person_id, cik = candidates[0]
        canonical_name = self._by_cik.get(cik, (None, None))[1]
        return PersonResolution(ResolutionStatus.RESOLVED.value, person_id, cik, canonical_name)


__all__ = [
    "CompanyIndex",
    "PersonIndex",
    "PersonResolution",
    "Resolution",
    "normalize_company_name",
    "normalize_person_name",
    "person_id_for_cik",
]
