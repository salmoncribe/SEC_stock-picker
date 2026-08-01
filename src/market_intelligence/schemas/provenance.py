"""Immutable, storage-neutral provenance contracts for disclosure packages.

These objects deliberately do not decide whether a trade should be made.
They answer the narrower, prerequisite question: what exact bytes were seen,
when and where were they observed, and is the package demonstrably first
public rather than a duplicate, amendment, correction, or stale observation?
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_intelligence import hashing

SHA256_HEX_LENGTH = 64


class ObservedSource(StrEnum):
    """A source where a package was observed, not an assertion of publicness."""

    SEC_EDGAR = "sec_edgar"
    ISSUER_IR = "issuer_ir"
    ISSUER_WEBCAST = "issuer_webcast"
    NEWSWIRE = "newswire"
    OTHER = "other"


class FilingRelation(StrEnum):
    ORIGINAL = "original"
    DUPLICATE = "duplicate"
    AMENDMENT = "amendment"
    CORRECTION = "correction"


class FirstPublicState(StrEnum):
    """Confidence in first broad public availability, not a trading decision."""

    VERIFIED = "verified"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONFLICTED = "conflicted"


class DisclosureEligibility(StrEnum):
    """Timing/provenance eligibility only; all later gates still apply."""

    FIRST_PUBLIC_VERIFIED = "first_public_verified"
    RESEARCH_ONLY = "research_only"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be normalized to UTC")
    return value.astimezone(UTC)


def _sha256(value: str) -> str:
    if len(value) != SHA256_HEX_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("must be a lowercase SHA-256 hex digest")
    return value


class ImmutableDocument(BaseModel):
    """One exact primary document or exhibit inside a filing package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_name: str = Field(min_length=1)
    sha256: str
    byte_size: int = Field(ge=0)
    document_type: str | None = None
    is_primary: bool = False

    _validate_sha256 = field_validator("sha256")(_sha256)


class SourceObservation(BaseModel):
    """An immutable retrieval/monitoring observation of a disclosure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: ObservedSource
    url: str = Field(min_length=1)
    observed_at: datetime
    raw_sha256: str
    asserted_public_at: datetime | None = None
    source_record_id: str | None = None

    _validate_raw_sha256 = field_validator("raw_sha256")(_sha256)
    _validate_observed_at = field_validator("observed_at")(_utc)
    _validate_asserted_public_at = field_validator("asserted_public_at")(_utc)

    @property
    def observation_hash(self) -> str:
        return hashing.sha256_json(self.model_dump(mode="json"))


class FilingPackage(BaseModel):
    """The complete SEC submission plus its primary document and exhibits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    cik: str = Field(pattern=r"^\d{10}$")
    form: str = Field(min_length=1)
    submission_sha256: str
    acceptance_datetime_raw: str = Field(pattern=r"^\d{14}$")
    edgar_accepted_at: datetime
    documents: tuple[ImmutableDocument, ...] = Field(min_length=1)
    relation: FilingRelation = FilingRelation.ORIGINAL
    related_accession_number: str | None = Field(default=None, pattern=r"^\d{10}-\d{2}-\d{6}$")

    _validate_submission_sha256 = field_validator("submission_sha256")(_sha256)
    _validate_accepted_at = field_validator("edgar_accepted_at")(_utc)

    @model_validator(mode="after")
    def _has_unique_document_names_and_one_primary(self) -> FilingPackage:
        names = [document.document_name for document in self.documents]
        if len(set(names)) != len(names):
            raise ValueError("document names must be unique within a filing package")
        if sum(document.is_primary for document in self.documents) != 1:
            raise ValueError("a filing package must identify exactly one primary document")
        return self

    @property
    def package_hash(self) -> str:
        """Stable digest of every immutable filing-package field."""
        return hashing.sha256_json(self.model_dump(mode="json"))

    @property
    def duplicate_exhibit_hashes(self) -> frozenset[str]:
        """Exhibit bytes reused inside this package (primary document excluded)."""
        exhibits = [document.sha256 for document in self.documents if not document.is_primary]
        return frozenset(value for value in exhibits if exhibits.count(value) > 1)


class DisclosurePackage(BaseModel):
    """A filing and all observed publication-source evidence attached to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    filing: FilingPackage
    observations: tuple[SourceObservation, ...] = ()

    @property
    def disclosure_hash(self) -> str:
        return hashing.sha256_json(self.model_dump(mode="json"))


class FirstPublicAssessment(BaseModel):
    """Pure result explaining why a package is verified or research-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: FirstPublicState
    first_public_at: datetime | None = None
    reason: str

    _validate_first_public_at = field_validator("first_public_at")(_utc)


class DisclosureClassification(BaseModel):
    """Auditable first-public classification for downstream deterministic gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    eligibility: DisclosureEligibility
    assessment: FirstPublicAssessment
    reasons: tuple[str, ...]


def is_amendment_form(form: str) -> bool:
    """True for SEC amendment forms such as ``4/A`` and ``10-K/A``."""
    return form.strip().upper().endswith("/A")


def filing_relation(
    package: FilingPackage,
    *,
    prior_submission_hashes: frozenset[str] = frozenset(),
    correction: bool = False,
) -> FilingRelation:
    """Classify duplicate/amendment/correction without mutating package history."""
    if correction:
        return FilingRelation.CORRECTION
    if package.submission_sha256 in prior_submission_hashes:
        return FilingRelation.DUPLICATE
    if is_amendment_form(package.form):
        return FilingRelation.AMENDMENT
    return package.relation


def assess_first_public(
    disclosure: DisclosurePackage,
    *,
    now: datetime,
    max_age: timedelta,
) -> FirstPublicAssessment:
    """Require an independent issuer/public-distribution observation.

    EDGAR acceptance and SEC discovery observations are retained as important
    regulatory provenance, but they cannot establish first broad public
    distribution on their own.  This preserves the research-only default for
    EDGAR-only coverage.
    """
    now = _utc(now)
    times: set[datetime] = set()
    for observation in disclosure.observations:
        if (
            observation.source in {ObservedSource.ISSUER_IR, ObservedSource.ISSUER_WEBCAST}
            and observation.asserted_public_at is not None
            # A page discovered after the historical decision cannot be used
            # retrospectively to establish what the market could have known.
            and observation.observed_at <= now
        ):
            times.add(observation.asserted_public_at)
    if not times:
        return FirstPublicAssessment(
            state=FirstPublicState.UNKNOWN,
            reason="no verified issuer release or webcast public-distribution observation",
        )

    if len(times) != 1:
        return FirstPublicAssessment(
            state=FirstPublicState.CONFLICTED,
            reason="issuer public-distribution observations disagree on first-public time",
        )
    (first_public_at,) = times
    if first_public_at > now or now - first_public_at > max_age:
        return FirstPublicAssessment(
            state=FirstPublicState.STALE,
            first_public_at=first_public_at,
            reason="verified first-public time is outside the allowed freshness window",
        )
    return FirstPublicAssessment(
        state=FirstPublicState.VERIFIED,
        first_public_at=first_public_at,
        reason="verified by issuer public-distribution observation",
    )


def classify_disclosure(
    disclosure: DisclosurePackage,
    *,
    now: datetime,
    max_age: timedelta,
    prior_submission_hashes: frozenset[str] = frozenset(),
    correction: bool = False,
) -> DisclosureClassification:
    """Classify first-public timing; suppress duplicate/amended/corrected packages."""
    assessment = assess_first_public(disclosure, now=now, max_age=max_age)
    relation = filing_relation(
        disclosure.filing,
        prior_submission_hashes=prior_submission_hashes,
        correction=correction,
    )
    reasons: list[str] = []
    if relation is not FilingRelation.ORIGINAL:
        reasons.append(f"filing_relation:{relation.value}")
    if assessment.state is not FirstPublicState.VERIFIED:
        reasons.append(f"first_public:{assessment.state.value}")
    elif (
        assessment.first_public_at is not None
        and assessment.first_public_at < disclosure.filing.edgar_accepted_at
    ):
        # A release can be correctly verified and still prove that this *filing*
        # is late to the public event.  Its contents remain valuable research,
        # but it cannot be promoted as a first-public filing disclosure.
        reasons.append("first_public:prior_issuer_release")
    if reasons:
        return DisclosureClassification(
            eligibility=DisclosureEligibility.RESEARCH_ONLY,
            assessment=assessment,
            reasons=tuple(reasons),
        )
    return DisclosureClassification(
        eligibility=DisclosureEligibility.FIRST_PUBLIC_VERIFIED,
        assessment=assessment,
        reasons=("first_public:verified",),
    )


__all__ = [
    "SHA256_HEX_LENGTH",
    "DisclosureClassification",
    "DisclosureEligibility",
    "DisclosurePackage",
    "FilingPackage",
    "FilingRelation",
    "FirstPublicAssessment",
    "FirstPublicState",
    "ImmutableDocument",
    "ObservedSource",
    "SourceObservation",
    "assess_first_public",
    "classify_disclosure",
    "filing_relation",
    "is_amendment_form",
]
