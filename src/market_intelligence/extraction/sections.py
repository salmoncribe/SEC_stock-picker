"""Deterministic extraction of SEC 10-K / 10-Q "Item" sections from filing HTML.

Why deterministic: downstream tables key sections by ``(accession, item_code)`` and
re-run the extractor over the same archived bytes. If parsing wobbled between runs the
platform could not tell a real amendment from parser drift. So everything here is a pure
function of ``(content, form)`` — no clocks, no randomness, no network, no model calls,
no iteration over unordered sets. ``EXTRACTION_METHOD`` is stored alongside results so a
change in parsing semantics is visible in the data rather than silent.

Pipeline
--------
1. ``html_to_text`` decodes the bytes, drops script/style/XBRL-header content, converts
   block-level markup to newlines, and normalizes whitespace and a small fixed set of
   unicode punctuation.
2. ``extract_sections`` regex-scans the *cleaned text* for item headers anchored to the
   start of a line, resolves the table-of-contents false positive with the fixed rule
   documented on that function, and slices bodies between accepted headers.

10-Q item codes
---------------
A 10-Q reuses the same item numbers in Part I and Part II (``Item 1`` is "Financial
Statements" in Part I and "Legal Proceedings" in Part II), so 10-Q codes are prefixed
with the roman-numeral part: ``I-1``, ``I-2``, ``II-1``, ``II-1A``, ... A 10-K has one
flat item sequence, so its codes carry no prefix: ``1``, ``1A``, ``7A``, ``9C``.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype, NavigableString, ProcessingInstruction, Tag

EXTRACTION_METHOD = "item_header_v1"

ITEMS_10K: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": (
        "Market for Registrant's Common Equity, Related Stockholder Matters and "
        "Issuer Purchases of Equity Securities"
    ),
    "6": "[Reserved]",
    "7": ("Management's Discussion and Analysis of Financial Condition and Results of Operations"),
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": ("Changes in and Disagreements with Accountants on Accounting and Financial Disclosure"),
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": (
        "Security Ownership of Certain Beneficial Owners and Management and Related "
        "Stockholder Matters"
    ),
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEMS_10Q: dict[str, str] = {
    "I-1": "Financial Statements",
    "I-2": (
        "Management's Discussion and Analysis of Financial Condition and Results of Operations"
    ),
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings",
    "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "II-3": "Defaults Upon Senior Securities",
    "II-4": "Mine Safety Disclosures",
    "II-5": "Other Information",
    "II-6": "Exhibits",
}

_TAXONOMIES: dict[str, dict[str, str]] = {"10-K": ITEMS_10K, "10-Q": ITEMS_10Q}

# A candidate header whose provisional body is shorter than this is treated as a
# table-of-contents / cross-reference artifact rather than a real section start.
_MIN_BODY_CHARS = 200

_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "head",
        "meta",
        "link",
        "ix:header",
        "ix:hidden",
        "ix:references",
        "ix:resources",
        "xbrl",
    }
)

_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "caption",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

# Conservative punctuation folding: only characters whose ASCII meaning is unambiguous,
# so section text hashes stably regardless of which word processor produced the filing.
_QUOTE_CHARS = "\u2018\u2019\u201a\u201b\u2032"
_DQUOTE_CHARS = "\u201c\u201d\u201e\u201f\u2033"
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_SPACE_CHARS = (
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
)
_NEWLINE_CHARS = "\u2028\u2029"
_ZERO_WIDTH_CHARS = "\u00ad\u200b\u200c\u200d\ufeff"

_CHAR_MAP: dict[str, str] = {}
_CHAR_MAP.update(dict.fromkeys(_QUOTE_CHARS, "'"))
_CHAR_MAP.update(dict.fromkeys(_DQUOTE_CHARS, '"'))
_CHAR_MAP.update(dict.fromkeys(_DASH_CHARS, "-"))
_CHAR_MAP.update(dict.fromkeys(_SPACE_CHARS, " "))
_CHAR_MAP.update(dict.fromkeys(_NEWLINE_CHARS, "\n"))
_CHAR_MAP.update(dict.fromkeys(_ZERO_WIDTH_CHARS, ""))
_TRANSLATION = str.maketrans(_CHAR_MAP)

_XML_DECL_RE = re.compile(r"^<\?xml[^>]*\?>", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_\-.:]+)""", re.IGNORECASE)
_XML_ENCODING_RE = re.compile(
    rb"""<\?xml[^>]*encoding\s*=\s*["']([A-Za-z0-9_\-.:]+)["']""", re.IGNORECASE
)

# Header must start a line: real item headings occupy their own block, while inline
# cross-references ("as described in Item 1A above") sit mid-sentence and are ignored.
_ITEM_RE = re.compile(
    r"^[ \t]*ITEM[ \t]*(?P<num>\d{1,2})[ \t]*(?P<suffix>[A-C])?[ \t]*[.:)\-]*[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_PART_RE = re.compile(r"^[ \t]*PART[ \t]+(?P<part>IV|III|II|I)\b", re.IGNORECASE | re.MULTILINE)

_INLINE_SPACE_RE = re.compile(r"[ \t\f\v]+")
_TRAILING_SPACE_RE = re.compile(r"[ \t]*\n[ \t]*")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ExtractedSection:
    """One item section of a filing, sliced from the cleaned document text."""

    item_code: str
    item_title: str
    text: str
    order: int


@dataclass(frozen=True)
class _Candidate:
    """A regex hit that looks like an item header, before acceptance rules run."""

    start: int
    body_start: int
    code: str


def _decode(content: bytes) -> str:
    """Decode filing bytes without ever raising.

    Order: BOM, declared meta/XML charset, utf-8, latin-1. Each candidate is decoded
    strictly so a wrong declaration falls through instead of producing mojibake; latin-1
    maps every byte, so the chain always terminates.
    """
    if content.startswith(codecs.BOM_UTF8):
        content = content[len(codecs.BOM_UTF8) :]
    if content.startswith(codecs.BOM_UTF16_LE) or content.startswith(codecs.BOM_UTF16_BE):
        try:
            return content.decode("utf-16")
        except (UnicodeDecodeError, LookupError):
            pass

    head = content[:4096]
    candidates: list[str] = []
    for pattern in (_XML_ENCODING_RE, _META_CHARSET_RE):
        found = pattern.search(head)
        if found:
            candidates.append(found.group(1).decode("ascii", errors="ignore"))
    candidates.extend(("utf-8", "latin-1"))

    for encoding in candidates:
        if not encoding:
            continue
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("latin-1", errors="replace")


def _emit_text(root: Tag) -> str:
    """Depth-first, iterative text emission with newlines around block-level tags.

    Iterative rather than recursive because SEC filings nest tables hundreds of levels
    deep in places and a recursive walk hits Python's recursion limit. Inline tags emit
    no separator, so ``<b>Item</b> 1A.`` stays a single matchable line.
    """
    out: list[str] = []
    stack: list[tuple[str, Any]] = [("node", root)]
    while stack:
        kind, payload = stack.pop()
        if kind == "newline":
            out.append("\n")
            continue
        if kind == "text":
            out.append(payload)
            continue
        tag: Tag = payload
        name = (tag.name or "").lower()
        if name in _DROP_TAGS:
            continue
        is_block = name in _BLOCK_TAGS
        if is_block:
            out.append("\n")
            stack.append(("newline", None))
        children = list(tag.children)
        for child in reversed(children):
            if isinstance(child, (Comment, ProcessingInstruction, Doctype)):
                continue
            if isinstance(child, NavigableString):
                stack.append(("text", str(child)))
            elif isinstance(child, Tag):
                stack.append(("node", child))
    return "".join(out)


def html_to_text(content: bytes) -> str:
    """Decode + strip HTML to normalized plain text. Deterministic.

    Drops ``<script>``/``<style>`` and inline-XBRL header/hidden blocks (which carry
    machine-only fact text that would otherwise land in section bodies), converts block
    markup to newlines, folds a fixed table of unicode dashes/quotes/spaces to ASCII,
    collapses horizontal whitespace runs to one space, and caps blank-line runs at one.
    """
    if not content:
        return ""
    # An inline-XBRL filing opens with an XML declaration; dropping it keeps BeautifulSoup
    # from warning that an XML document is being parsed as HTML (HTML is what we want --
    # the lxml HTML parser is the tolerant one, and filings are not well-formed XML).
    markup = _XML_DECL_RE.sub("", _decode(content).lstrip(), count=1)
    soup = BeautifulSoup(markup, "lxml")
    text = _emit_text(soup)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_TRANSLATION)
    text = _INLINE_SPACE_RE.sub(" ", text)
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _normalize_form(form: str) -> str:
    """Map ``10-K/A`` -> ``10-K``: amendments carry the same item taxonomy."""
    normalized = (form or "").strip().upper()
    if normalized.endswith("/A"):
        normalized = normalized[:-2]
    return normalized


def _part_at(part_marks: list[tuple[int, str]], offset: int) -> str:
    """Roman numeral of the most recent ``PART`` heading at or before ``offset``."""
    current = "I"
    for position, part in part_marks:
        if position > offset:
            break
        current = part
    return current


def _find_candidates(text: str, base_form: str, taxonomy: dict[str, str]) -> list[_Candidate]:
    """All line-anchored item headers whose code exists in this form's taxonomy."""
    part_marks: list[tuple[int, str]] = []
    if base_form == "10-Q":
        part_marks = [
            (found.start(), found.group("part").upper()) for found in _PART_RE.finditer(text)
        ]

    candidates: list[_Candidate] = []
    for found in _ITEM_RE.finditer(text):
        suffix = found.group("suffix") or ""
        code = f"{int(found.group('num'))}{suffix.upper()}"
        if base_form == "10-Q":
            code = f"{_part_at(part_marks, found.start())}-{code}"
        if code not in taxonomy:
            continue
        candidates.append(_Candidate(start=found.start(), body_start=found.end(), code=code))
    return candidates


def _select(candidates: list[_Candidate], taxonomy: dict[str, str], end: int) -> list[_Candidate]:
    """Resolve duplicate headers and enforce taxonomy order. Fixed rule, no tuning.

    Table-of-contents rule — a 10-K/10-Q lists every item in a TOC before the body, so
    most codes match more than once:

    1. Give every candidate a *provisional body span*: from the end of its header to the
       start of the next candidate in document order (or end of document for the last).
       TOC rows sit next to each other, so their spans are a few dozen characters; the
       real heading opens the section and spans thousands.
    2. For each item code keep exactly one candidate — the one with the LARGEST
       provisional span; ties broken by earliest offset, so the choice is total and
       reproducible.
    3. Drop any survivor whose provisional span is under ``_MIN_BODY_CHARS`` (200). This
       catches codes that appear *only* in the TOC. The cost is that a genuinely empty
       section (``Item 6. [Reserved]``) is not emitted, which is preferred over emitting
       a TOC line as a section.
    4. Keep the longest subsequence of survivors that is strictly increasing in both
       document offset and taxonomy position, so a stray match (typically a code that
       only ever appears in the TOC, such as ``Item 16``) cannot reorder the sequence or
       truncate the real body. Ties are resolved toward the earliest candidate, making
       the choice total.
    """
    if not candidates:
        return []

    order_index = {code: index for index, code in enumerate(taxonomy)}
    spans: list[int] = []
    for index, candidate in enumerate(candidates):
        stop = candidates[index + 1].start if index + 1 < len(candidates) else end
        spans.append(max(stop - candidate.body_start, 0))

    best: dict[str, tuple[int, int]] = {}
    for index, candidate in enumerate(candidates):
        current = best.get(candidate.code)
        if current is None or spans[index] > current[1]:
            best[candidate.code] = (index, spans[index])

    survivors = sorted(index for index, span in best.values() if span >= _MIN_BODY_CHARS)
    if not survivors:
        return []

    positions = [order_index[candidates[index].code] for index in survivors]
    lengths = [1] * len(survivors)
    previous = [-1] * len(survivors)
    for i in range(len(survivors)):
        for j in range(i):
            if positions[j] < positions[i] and lengths[j] + 1 > lengths[i]:
                lengths[i] = lengths[j] + 1
                previous[i] = j

    tail = max(range(len(survivors)), key=lambda i: (lengths[i], -i))
    chain: list[int] = []
    while tail != -1:
        chain.append(survivors[tail])
        tail = previous[tail]
    chain.reverse()
    return [candidates[index] for index in chain]


def extract_sections(content: bytes, form: str) -> list[ExtractedSection]:
    """Split a 10-K/10-Q document into its Item sections, in document order.

    ``form`` is e.g. ``"10-K"``, ``"10-K/A"``, ``"10-Q"``, ``"10-Q/A"`` (amendments are
    treated as the base form). Returns ``[]`` for unsupported forms or when no item
    headers are found. Section text runs from the end of a header to the start of the
    next accepted header; the last section runs to the end of the document.
    """
    base_form = _normalize_form(form)
    taxonomy = _TAXONOMIES.get(base_form)
    if taxonomy is None:
        return []

    text = html_to_text(content)
    if not text:
        return []

    accepted = _select(_find_candidates(text, base_form, taxonomy), taxonomy, len(text))

    sections: list[ExtractedSection] = []
    for order, candidate in enumerate(accepted):
        stop = accepted[order + 1].start if order + 1 < len(accepted) else len(text)
        sections.append(
            ExtractedSection(
                item_code=candidate.code,
                item_title=taxonomy[candidate.code],
                text=text[candidate.body_start : stop].strip(),
                order=order,
            )
        )
    return sections


__all__ = [
    "EXTRACTION_METHOD",
    "ITEMS_10K",
    "ITEMS_10Q",
    "ExtractedSection",
    "extract_sections",
    "html_to_text",
]
