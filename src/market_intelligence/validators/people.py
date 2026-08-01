"""Validation rules for people/role_memberships (``schemas.people``).

Follows the platform contract (see ``validators/events.py``): a problem is
recorded on the record itself via ``add_error`` and classified ``warning``
(stored, flagged) or ``rejected`` (kept for audit, not persisted).

Per the design doc's validation section: a ``role_memberships`` row is
rejected when ``person_id``/``company_id`` don't resolve, the same as any
other record type. In practice ``collectors/roles.py`` only ever builds a
candidate for a CIK/company pair it already resolved, so these checks are a
defensive backstop against a degenerate upstream row (e.g. an empty-string
id slipping through), not the primary filter.
"""

from __future__ import annotations

from market_intelligence.schemas.people import PersonRecord, RoleMembershipRecord


def validate_person(record: PersonRecord) -> PersonRecord:
    """Validate one person identity record.

    Rejections: ``person_id`` or ``reporting_owner_cik`` missing/blank -- the
    natural key this table is built and deduped on.
    """
    if not record.person_id or not record.person_id.strip():
        record.add_error("person_id is required", reject=True)
    if not record.reporting_owner_cik or not record.reporting_owner_cik.strip():
        record.add_error("reporting_owner_cik is required", reject=True)
    return record


def validate_role_membership(record: RoleMembershipRecord) -> RoleMembershipRecord:
    """Validate one person-to-company role relationship.

    Rejections: ``person_id`` or ``company_id`` missing/blank -- the natural
    key ``(person_id, company_id)`` this table is deduped on.

    Warnings: none of ``is_officer``/``is_director``/``is_ten_pct_owner`` set
    -- SEC's free-text relationship field did not match any known category, so
    the row is kept (a real relationship was asserted) but flagged as worth a
    look, since it usually means the parser's substring rules missed a real
    phrasing.
    """
    if not record.person_id or not record.person_id.strip():
        record.add_error("person_id is required", reject=True)
    if not record.company_id or not record.company_id.strip():
        record.add_error("company_id is required", reject=True)
    if not (record.is_officer or record.is_director or record.is_ten_pct_owner):
        record.add_error(
            "no qualifying relationship flag (officer/director/10% owner) is set"
        )
    return record


__all__ = ["validate_person", "validate_role_membership"]
