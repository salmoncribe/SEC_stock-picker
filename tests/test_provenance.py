"""Pure provenance-contract tests; no database or live SEC dependency."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from market_intelligence import hashing
from market_intelligence.schemas.provenance import (
    DisclosureEligibility,
    DisclosurePackage,
    FilingPackage,
    FilingRelation,
    FirstPublicState,
    ImmutableDocument,
    ObservedSource,
    SourceObservation,
    classify_disclosure,
)

NOW = datetime(2024, 5, 7, 15, tzinfo=UTC)


def _digest(label: str) -> str:
    return hashing.sha256_text(label)


def _filing(
    *, form: str = "8-K", relation: FilingRelation = FilingRelation.ORIGINAL
) -> FilingPackage:
    return FilingPackage(
        accession_number="0001045810-24-000100",
        cik="0001045810",
        form=form,
        submission_sha256=_digest("submission"),
        acceptance_datetime_raw="20240506163504",
        edgar_accepted_at=datetime(2024, 5, 6, 20, 35, 4, tzinfo=UTC),
        documents=(
            ImmutableDocument(
                document_name="main.htm",
                sha256=_digest("main"),
                byte_size=100,
                is_primary=True,
            ),
        ),
        relation=relation,
    )


def _release(first_public_at: datetime = datetime(2024, 5, 7, 14, tzinfo=UTC)) -> SourceObservation:
    return SourceObservation(
        source=ObservedSource.ISSUER_IR,
        url="https://investor.example/release",
        observed_at=NOW,
        asserted_public_at=first_public_at,
        raw_sha256=_digest("release"),
    )


def test_hashes_are_immutable_and_document_names_cannot_duplicate() -> None:
    document = ImmutableDocument(
        document_name="main.htm", sha256=_digest("main"), byte_size=1, is_primary=True
    )
    with pytest.raises(ValidationError):
        document.sha256 = _digest("other")  # type: ignore[misc]

    with pytest.raises(ValidationError, match="document names must be unique"):
        FilingPackage(
            accession_number="0001045810-24-000100",
            cik="0001045810",
            form="8-K",
            submission_sha256=_digest("submission"),
            acceptance_datetime_raw="20240506163504",
            edgar_accepted_at=datetime(2024, 5, 6, 20, 35, 4, tzinfo=UTC),
            documents=(
                document,
                ImmutableDocument(document_name="main.htm", sha256=_digest("other"), byte_size=2),
            ),
        )


def test_duplicate_exhibit_bytes_are_visible_in_package_contract() -> None:
    package = _filing().model_copy(
        update={
            "documents": (
                ImmutableDocument(
                    document_name="main.htm", sha256=_digest("main"), byte_size=100, is_primary=True
                ),
                ImmutableDocument(
                    document_name="ex-1.htm", sha256=_digest("same-exhibit"), byte_size=20
                ),
                ImmutableDocument(
                    document_name="ex-2.htm", sha256=_digest("same-exhibit"), byte_size=20
                ),
            )
        }
    )

    assert package.duplicate_exhibit_hashes == {_digest("same-exhibit")}
    assert package.package_hash == package.package_hash


def test_edgar_only_coverage_is_research_only_with_unknown_first_public() -> None:
    disclosure = DisclosurePackage(
        filing=_filing(),
        observations=(
            SourceObservation(
                source=ObservedSource.SEC_EDGAR,
                url="https://sec.gov/Archives/example",
                observed_at=NOW,
                raw_sha256=_digest("sec-observation"),
            ),
        ),
    )

    result = classify_disclosure(disclosure, now=NOW, max_age=timedelta(hours=4))

    assert result.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert result.assessment.state is FirstPublicState.UNKNOWN


def test_late_observation_cannot_retroactively_verify_first_public() -> None:
    late = _release().model_copy(update={"observed_at": NOW + timedelta(minutes=1)})
    result = classify_disclosure(
        DisclosurePackage(filing=_filing(), observations=(late,)),
        now=NOW,
        max_age=timedelta(hours=4),
    )
    assert result.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert result.assessment.state is FirstPublicState.UNKNOWN


def test_verified_issuer_release_is_first_public_verified_only_when_fresh() -> None:
    result = classify_disclosure(
        DisclosurePackage(filing=_filing(), observations=(_release(),)),
        now=NOW,
        max_age=timedelta(hours=4),
    )

    assert result.eligibility is DisclosureEligibility.FIRST_PUBLIC_VERIFIED
    assert result.assessment.first_public_at == datetime(2024, 5, 7, 14, tzinfo=UTC)


def test_prior_issuer_release_makes_the_later_filing_research_only() -> None:
    result = classify_disclosure(
        DisclosurePackage(
            filing=_filing(),
            observations=(_release(datetime(2024, 5, 6, 20, tzinfo=UTC)),),
        ),
        now=NOW,
        max_age=timedelta(days=1),
    )

    assert result.assessment.state is FirstPublicState.VERIFIED
    assert result.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert "first_public:prior_issuer_release" in result.reasons


def test_stale_or_conflicting_first_public_evidence_is_research_only() -> None:
    stale = classify_disclosure(
        DisclosurePackage(
            filing=_filing(),
            observations=(_release(datetime(2024, 5, 6, 14, tzinfo=UTC)),),
        ),
        now=NOW,
        max_age=timedelta(hours=4),
    )
    conflicted = classify_disclosure(
        DisclosurePackage(
            filing=_filing(),
            observations=(_release(), _release(datetime(2024, 5, 7, 13, tzinfo=UTC))),
        ),
        now=NOW,
        max_age=timedelta(hours=4),
    )

    assert stale.assessment.state is FirstPublicState.STALE
    assert stale.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert conflicted.assessment.state is FirstPublicState.CONFLICTED
    assert conflicted.eligibility is DisclosureEligibility.RESEARCH_ONLY


@pytest.mark.parametrize(
    ("form", "correction", "expected_reason"),
    [
        ("8-K/A", False, "filing_relation:amendment"),
        ("8-K", True, "filing_relation:correction"),
    ],
)
def test_amendments_and_corrections_cancel_first_public_eligibility(
    form: str, correction: bool, expected_reason: str
) -> None:
    result = classify_disclosure(
        DisclosurePackage(filing=_filing(form=form), observations=(_release(),)),
        now=NOW,
        max_age=timedelta(hours=4),
        correction=correction,
    )

    assert result.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert expected_reason in result.reasons


def test_prior_submission_hash_marks_a_duplicate_and_preserves_original_package() -> None:
    filing = _filing()
    result = classify_disclosure(
        DisclosurePackage(filing=filing, observations=(_release(),)),
        now=NOW,
        max_age=timedelta(hours=4),
        prior_submission_hashes=frozenset({filing.submission_sha256}),
    )

    assert result.eligibility is DisclosureEligibility.RESEARCH_ONLY
    assert "filing_relation:duplicate" in result.reasons
    assert filing.relation is FilingRelation.ORIGINAL
