"""Name resolution as a conservatism table.

The load-bearing guarantee is asymmetric: a missed edge costs one observation, a
wrong edge invents a relationship between two real companies. So these tests pin
not just the clean resolutions but every way the resolver is expected to *refuse*
-- shared names, generic single tokens, and companies with no ticker.
"""

from __future__ import annotations

import duckdb

from market_intelligence.entities import (
    CompanyIndex,
    PersonIndex,
    Resolution,
    normalize_company_name,
    normalize_person_name,
    person_id_for_cik,
)
from market_intelligence.schemas.edges import ResolutionStatus

# (cik, ticker, company_name) -- names carry the casing/punctuation the real
# table does. "VF Corp" is stored terse on purpose: the query "V.F. Corporation"
# must reach it through normalization alone.
COMPANIES: list[tuple[str, str | None, str]] = [
    ("0000320187", "NKE", "NIKE, Inc."),
    ("0001336917", "UAA", "Under Armour, Inc."),
    ("0001397187", "LULU", "lululemon athletica inc."),
    ("0000103379", "VFC", "VF Corp"),
    ("0001725579", "COLB", "Columbia Banking System, Inc."),
    ("0001050797", "COLM", "Columbia Sportswear Company"),
    ("0001562476", "TMHC", "Taylor Morrison Home Corporation"),
    ("0001401667", "PBYI", "Puma Biotechnology, Inc."),
    ("0000000001", None, "Ghost Holdings Corp"),  # no ticker: unusable target
]


def index() -> CompanyIndex:
    return CompanyIndex(COMPANIES)


# --------------------------------------------------------------------------- #
# clean resolutions through normalization                                     #
# --------------------------------------------------------------------------- #
def test_suffix_and_case_are_normalized_away():
    """A legal suffix and stray casing must not stop an exact match."""
    ua = index().resolve("Under Armour, Inc.")
    assert ua.status == ResolutionStatus.RESOLVED.value
    assert ua.ticker == "UAA"
    assert ua.cik == "0001336917"
    assert ua.confidence == 1.0

    lulu = index().resolve("lululemon athletica inc.")
    assert lulu.status == ResolutionStatus.RESOLVED.value
    assert lulu.ticker == "LULU"


def test_punctuation_and_spacing_collapse_to_one_token():
    """'V.F. Corporation' must reach the terse 'VF Corp' row via normalization."""
    vf = index().resolve("V.F. Corporation")

    assert vf.status == ResolutionStatus.RESOLVED.value
    assert vf.ticker == "VFC"
    assert vf.matched_name == "VF Corp"


# --------------------------------------------------------------------------- #
# refusals -- the expensive-mistake guards                                    #
# --------------------------------------------------------------------------- #
def test_shared_name_is_ambiguous_never_a_guess():
    """'Columbia' names two real companies; picking one would be a fabrication."""
    result = index().resolve("Columbia")

    assert result.status == ResolutionStatus.AMBIGUOUS.value
    assert result.cik is None
    assert result.ticker is None


def test_absent_name_is_unresolved():
    """Foreign/private targets simply are not in our universe."""
    result = index().resolve("adidas")

    assert result.status == ResolutionStatus.UNRESOLVED.value
    assert result.cik is None


def test_blank_input_is_unresolved():
    for blank in ("", "   ", "\t"):
        result = index().resolve(blank)
        assert result.status == ResolutionStatus.UNRESOLVED.value
        assert result.cik is None


def test_null_ticker_company_is_never_a_match():
    """A company with no ticker has no price series, so it cannot be a target."""
    result = index().resolve("Ghost Holdings Corporation")

    assert result.status == ResolutionStatus.UNRESOLVED.value
    assert result.cik is None
    assert result.ticker is None


def test_lone_generic_token_does_not_resolve_on_subset():
    """'Puma' (foreign) must not resolve to Puma Biotechnology on one token."""
    result = index().resolve("Puma")

    assert result.status == ResolutionStatus.UNRESOLVED.value
    assert result.ticker != "PBYI"


# --------------------------------------------------------------------------- #
# the conservative subset fallback                                            #
# --------------------------------------------------------------------------- #
def test_multi_token_subset_resolves_when_unique():
    """Two distinctive tokens uniquely identify a longer legal name."""
    result = index().resolve("Taylor Morrison")

    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.ticker == "TMHC"
    assert 0.0 < result.confidence < 1.0  # trusted, but below an exact match


def test_resolution_is_frozen():
    """Verdicts are immutable records."""
    result = index().resolve("NIKE, Inc.")
    assert isinstance(result, Resolution)
    try:
        result.ticker = "XXX"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - only reached if frozen protection regressed
        raise AssertionError("Resolution should be frozen")


# --------------------------------------------------------------------------- #
# normalization in isolation                                                  #
# --------------------------------------------------------------------------- #
def test_normalize_strips_suffixes_and_punctuation():
    assert normalize_company_name("NIKE, Inc.") == "NIKE"
    assert normalize_company_name("Under Armour, Inc.") == "UNDER ARMOUR"
    # Punctuation + spacing converge on the same token.
    assert normalize_company_name("V.F. Corporation") == normalize_company_name("VF Corp") == "VF"
    # A representative sweep of legal forms, all stripped.
    assert normalize_company_name("Acme Holdings, LLC") == "ACME"
    assert normalize_company_name("Global Industries N.V.") == "GLOBAL INDUSTRIES"
    assert normalize_company_name("Kering S.A.") == "KERING"
    # Leading article and hyphen handling.
    assert normalize_company_name("The Coca-Cola Company") == "COCA COLA"
    # Interior "Group" is identity, not a suffix, so it survives.
    assert normalize_company_name("Group 1 Automotive, Inc.") == "GROUP 1 AUTOMOTIVE"
    # Blank / all-suffix input reduces to nothing.
    assert normalize_company_name("   ") == ""


# --------------------------------------------------------------------------- #
# PersonIndex -- CIK-keyed identity, name resolution mirrors CompanyIndex     #
# --------------------------------------------------------------------------- #
CIK_JANE = "0002000001"
CIK_JOHN = "0002000002"
CIK_OTHER_JANE = "0002000003"

# (person_id, reporting_owner_cik, canonical_name, name_variants)
PEOPLE: list[tuple[str, str, str | None, list[str]]] = [
    ("person-jane", CIK_JANE, "Jane Doe", ["Jane Doe", "J. Doe"]),
    ("person-john", CIK_JOHN, "John Roe", ["John Roe"]),
    # A second, distinct real person who happens to share Jane's name.
    ("person-jane2", CIK_OTHER_JANE, "Jane Doe", ["Jane Doe"]),
]


def person_index() -> PersonIndex:
    return PersonIndex(PEOPLE)


def test_person_id_for_cik_is_pure_and_deterministic() -> None:
    assert person_id_for_cik(CIK_JANE) == person_id_for_cik(CIK_JANE)
    assert person_id_for_cik(CIK_JANE) != person_id_for_cik(CIK_JOHN)


def test_normalize_person_name_folds_case_and_whitespace() -> None:
    assert normalize_person_name("jane   doe") == normalize_person_name("JANE DOE") == "JANE DOE"


def test_normalize_person_name_does_not_reorder_tokens() -> None:
    """Unlike company names, no attempt is made to reconcile "Doe, Jane" with
    "Jane Doe" -- SEC's own filer-name field orders inconsistently, and
    guessing the order risks merging two different real people.
    """
    assert normalize_person_name("Doe, Jane") != normalize_person_name("Jane Doe")


def test_resolve_cik_finds_an_exact_match() -> None:
    result = person_index().resolve_cik(CIK_JANE)
    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.person_id == "person-jane"
    assert result.canonical_name == "Jane Doe"


def test_resolve_cik_unknown_cik_is_unresolved() -> None:
    result = person_index().resolve_cik("0009999999")
    assert result.status == ResolutionStatus.UNRESOLVED.value
    assert result.person_id is None


def test_resolve_name_unique_name_resolves() -> None:
    result = person_index().resolve_name("John Roe")
    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.person_id == "person-john"
    assert result.reporting_owner_cik == CIK_JOHN


def test_resolve_name_shared_by_two_real_people_is_ambiguous_never_a_guess() -> None:
    """Two distinct CIKs share the exact name 'Jane Doe' -- refuse, don't pick one."""
    result = person_index().resolve_name("Jane Doe")
    assert result.status == ResolutionStatus.AMBIGUOUS.value
    assert result.person_id is None


def test_resolve_name_matches_a_stored_variant() -> None:
    result = person_index().resolve_name("J. Doe")
    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.person_id == "person-jane"


def test_resolve_name_is_case_and_whitespace_insensitive() -> None:
    result = person_index().resolve_name("john   roe")
    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.person_id == "person-john"


def test_resolve_name_unknown_name_is_unresolved() -> None:
    result = person_index().resolve_name("Nobody Here")
    assert result.status == ResolutionStatus.UNRESOLVED.value
    assert result.person_id is None


def test_resolve_name_blank_input_is_unresolved() -> None:
    for blank in ("", "   ", "\t"):
        result = person_index().resolve_name(blank)
        assert result.status == ResolutionStatus.UNRESOLVED.value


def test_person_index_from_connection_reads_the_people_table(
    memory_db: duckdb.DuckDBPyConnection,
) -> None:
    memory_db.execute(
        """
        INSERT INTO people (person_id, reporting_owner_cik, canonical_name, name_variants)
        VALUES (?, ?, ?, ?)
        """,
        ["person-jane", CIK_JANE, "Jane Doe", '["Jane Doe", "J. Doe"]'],
    )
    index = PersonIndex.from_connection(memory_db)

    result = index.resolve_cik(CIK_JANE)
    assert result.status == ResolutionStatus.RESOLVED.value
    assert result.person_id == "person-jane"

    by_variant = index.resolve_name("J. Doe")
    assert by_variant.person_id == "person-jane"
