"""Tests for deterministic Item-section extraction.

Fully offline: the two fixtures are real NVIDIA filings (10-K FY2026 and the Q1 FY2027
10-Q) trimmed for size but structurally intact -- cover page, full table of contents, and
every Item header are preserved, so the TOC false positive is genuinely exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from market_intelligence.extraction import (
    EXTRACTION_METHOD,
    ITEMS_10K,
    ITEMS_10Q,
    ExtractedSection,
    extract_sections,
    html_to_text,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOC_LISTING = "Item 1A.\n\nRisk Factors\n\n12"


@pytest.fixture(scope="module")
def tenk_bytes() -> bytes:
    return (FIXTURES / "sec_doc_nvda_10k.htm").read_bytes()


@pytest.fixture(scope="module")
def tenq_bytes() -> bytes:
    return (FIXTURES / "sec_doc_nvda_10q.htm").read_bytes()


def _codes(sections: list[ExtractedSection]) -> list[str]:
    return [section.item_code for section in sections]


def test_extraction_method_is_pinned() -> None:
    assert EXTRACTION_METHOD == "item_header_v1"


def test_taxonomies_cover_the_official_items() -> None:
    assert list(ITEMS_10K) == [
        "1",
        "1A",
        "1B",
        "1C",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "7A",
        "8",
        "9",
        "9A",
        "9B",
        "9C",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
    ]
    assert list(ITEMS_10Q) == [
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "II-1",
        "II-1A",
        "II-2",
        "II-3",
        "II-4",
        "II-5",
        "II-6",
    ]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_extraction_is_byte_identical_across_runs(tenk_bytes: bytes, tenq_bytes: bytes) -> None:
    assert extract_sections(tenk_bytes, "10-K") == extract_sections(tenk_bytes, "10-K")
    assert extract_sections(tenq_bytes, "10-Q") == extract_sections(tenq_bytes, "10-Q")


def test_html_to_text_is_byte_identical_across_runs(tenk_bytes: bytes) -> None:
    assert html_to_text(tenk_bytes) == html_to_text(tenk_bytes)


def test_sections_are_frozen(tenk_bytes: bytes) -> None:
    section = extract_sections(tenk_bytes, "10-K")[0]
    with pytest.raises(AttributeError):
        section.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Table-of-contents false positive
# ---------------------------------------------------------------------------


def test_fixture_actually_contains_a_table_of_contents(tenk_bytes: bytes) -> None:
    """Guard the guard: if trimming ever removed the TOC the next test would pass vacuously."""
    text = html_to_text(tenk_bytes)
    assert TOC_LISTING in text
    assert text.index(TOC_LISTING) < text.index("Item 1. Business")


def test_toc_entries_are_not_returned_as_sections(tenk_bytes: bytes) -> None:
    sections = extract_sections(tenk_bytes, "10-K")
    by_code = {section.item_code: section for section in sections}

    # A TOC-derived "Item 1" body would be the next TOC rows: "Business 4 Item 1A. ...".
    assert not by_code["1"].text.startswith("Business\n\n4")
    assert by_code["1"].text.startswith("Business\n\nOur Company")

    for section in sections:
        assert TOC_LISTING not in section.text
        assert len(section.text) >= 200

    assert len(_codes(sections)) == len(set(_codes(sections)))


def test_toc_entries_are_not_returned_as_sections_for_10q(tenq_bytes: bytes) -> None:
    sections = extract_sections(tenq_bytes, "10-Q")
    by_code = {section.item_code: section for section in sections}
    assert by_code["I-1"].text.startswith("Financial Statements (Unaudited)\n\nNVIDIA")
    assert len(_codes(sections)) == len(set(_codes(sections)))


# ---------------------------------------------------------------------------
# 10-K
# ---------------------------------------------------------------------------


def test_10k_sections_are_in_ascending_document_order(tenk_bytes: bytes) -> None:
    sections = extract_sections(tenk_bytes, "10-K")
    assert [section.order for section in sections] == list(range(len(sections)))

    taxonomy_order = list(ITEMS_10K)
    positions = [taxonomy_order.index(section.item_code) for section in sections]
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions))


def test_10k_risk_factors_has_substantial_body(tenk_bytes: bytes) -> None:
    section = next(s for s in extract_sections(tenk_bytes, "10-K") if s.item_code == "1A")
    assert section.item_title == "Risk Factors"
    assert len(section.text) > 5_000
    assert "risk factors" in section.text[:200].lower()


def test_10k_extracts_expected_codes(tenk_bytes: bytes) -> None:
    codes = _codes(extract_sections(tenk_bytes, "10-K"))
    for expected in ("1", "1A", "2", "5", "7", "7A", "8", "9A", "15"):
        assert expected in codes


def test_10k_titles_come_from_the_taxonomy(tenk_bytes: bytes) -> None:
    for section in extract_sections(tenk_bytes, "10-K"):
        assert section.item_title == ITEMS_10K[section.item_code]


def test_amendment_is_treated_as_the_base_form(tenk_bytes: bytes) -> None:
    assert extract_sections(tenk_bytes, "10-K/A") == extract_sections(tenk_bytes, "10-K")
    assert extract_sections(tenk_bytes, " 10-k/a ") == extract_sections(tenk_bytes, "10-K")


# ---------------------------------------------------------------------------
# 10-Q
# ---------------------------------------------------------------------------


def test_10q_codes_are_part_prefixed(tenq_bytes: bytes) -> None:
    sections = extract_sections(tenq_bytes, "10-Q")
    codes = _codes(sections)
    assert codes == sorted(codes, key=list(ITEMS_10Q).index)
    assert all(code.startswith(("I-", "II-")) for code in codes)
    assert "I-1" in codes
    assert "II-1A" in codes


def test_10q_same_number_in_both_parts_gets_different_titles(tenq_bytes: bytes) -> None:
    by_code = {s.item_code: s for s in extract_sections(tenq_bytes, "10-Q")}
    assert by_code["I-1"].item_title == "Financial Statements"
    assert by_code["II-1"].item_title == "Legal Proceedings"
    assert by_code["I-1"].text != by_code["II-1"].text


def test_10q_risk_factors_is_part_two(tenq_bytes: bytes) -> None:
    by_code = {s.item_code: s for s in extract_sections(tenq_bytes, "10-Q")}
    assert by_code["II-1A"].item_title == "Risk Factors"
    assert len(by_code["II-1A"].text) > 5_000


def test_10q_sections_are_in_ascending_document_order(tenq_bytes: bytes) -> None:
    sections = extract_sections(tenq_bytes, "10-Q")
    assert [s.order for s in sections] == list(range(len(sections)))
    positions = [list(ITEMS_10Q).index(s.item_code) for s in sections]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Unsupported forms and malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["8-K", "S-1", "10-K405", "DEF 14A", "", "   "])
def test_unsupported_forms_return_empty(tenk_bytes: bytes, form: str) -> None:
    assert extract_sections(tenk_bytes, form) == []


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"   ",
        b"<html><body><p>No items here at all.</p></body></html>",
        b"<html><body><p>Item 1. Business</p>",  # truncated, body under threshold
        b"<<<>>> not really markup &amp;",
        b"\x00\x01\x02\xff\xfe binary junk",
        b"<html><body><table><tr><td>Item</td></tr></table></body></html>",
    ],
)
def test_malformed_input_returns_empty_without_raising(content: bytes) -> None:
    assert extract_sections(content, "10-K") == []
    assert isinstance(html_to_text(content), str)


def test_cross_references_alone_do_not_create_sections() -> None:
    body = "Please refer to Item 1A. Risk Factors elsewhere. " * 40
    content = f"<html><body><p>{body}</p></body></html>".encode()
    assert extract_sections(content, "10-K") == []


# ---------------------------------------------------------------------------
# Encoding and text normalization
# ---------------------------------------------------------------------------


def test_latin1_bytes_decode_without_raising() -> None:
    content = "<html><body><p>caf\xe9 na\xefve</p></body></html>".encode("latin-1")
    assert "café naïve" in html_to_text(content)


def test_declared_meta_charset_is_honoured() -> None:
    content = (
        b"<html><head><meta charset='windows-1252'></head>"
        b"<body><p>fee \x93quoted\x94 rate</p></body></html>"
    )
    assert '"quoted"' in html_to_text(content)


def test_bogus_declared_charset_falls_back_to_utf8() -> None:
    content = (
        "<html><head><meta charset='definitely-not-a-codec'></head>"
        "<body><p>ünïcode</p></body></html>"
    ).encode()
    assert "ünïcode" in html_to_text(content)


def test_utf8_bom_is_stripped() -> None:
    content = b"\xef\xbb\xbf<html><body><p>hello</p></body></html>"
    assert html_to_text(content) == "hello"


def test_whitespace_and_punctuation_are_normalized() -> None:
    content = (
        "<html><body><p>a\xa0\xa0b\t\tc</p>"
        "<p>NVIDIA\u2019s \u201cnote\u201d \u2014 done</p></body></html>"
    ).encode()
    text = html_to_text(content)
    assert "a b c" in text
    assert 'NVIDIA\'s "note" - done' in text
    assert "\xa0" not in text
    assert "\n\n\n" not in text


def test_script_style_and_xbrl_header_are_dropped() -> None:
    content = (
        b"<html><body>"
        b"<div style='display:none'><ix:header><ix:hidden>"
        b"<ix:nonnumeric name='dei:EntityCentralIndexKey'>0001045810</ix:nonnumeric>"
        b"</ix:hidden></ix:header></div>"
        b"<script>var x = 'scripted';</script><style>.c{color:red}</style>"
        b"<p>visible</p></body></html>"
    )
    text = html_to_text(content)
    assert text == "visible"


def test_br_and_block_tags_become_newlines() -> None:
    content = b"<html><body><p>one<br/>two</p><div>three</div></body></html>"
    assert html_to_text(content).split("\n\n") == ["one", "two", "three"]
