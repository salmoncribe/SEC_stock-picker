"""Concrete Financial Statement and Event Extractor."""

from __future__ import annotations

import re
from typing import Dict, Any


class FinancialExtractor:
    """Extracts concrete numerical financial line items and executive/corporate event summaries from filing text."""

    def _parse_dollar_amount(self, text_snippet: str) -> float | None:
        """Parse dollar amount from strings like '$ 25.4 billion', '$1,234 million', '$ 5.12'."""
        match = re.search(r"\$\s*([\d\,\.]+)\s*(billion|million|thousand)?", text_snippet, re.IGNORECASE)
        if not match:
            return None
        val_str = match.group(1).replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            return None

        unit = (match.group(2) or "").lower()
        if unit == "billion":
            val *= 1_000_000_000
        elif unit == "million":
            val *= 1_000_000
        elif unit == "thousand":
            val *= 1_000
        return val

    def extract_financial_data(self, text: str, sections: Dict[str, str], form: str = "") -> Dict[str, Any]:
        """Extract concrete numerical line items and text summaries."""
        data: Dict[str, Any] = {}

        # 1. Revenue
        rev_match = re.search(
            r"(?:total\s+net\s+revenues?|total\s+revenues?|net\s+sales|revenues?)\s*(?:was|were|of)?\s*(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?)",
            text,
            re.IGNORECASE,
        )
        if rev_match:
            data["revenue_text"] = rev_match.group(0).strip()
            data["revenue_usd"] = self._parse_dollar_amount(rev_match.group(1))

        # 2. Net Income / Loss
        net_inc_match = re.search(
            r"(?:net\s+income|net\s+loss)\s*(?:was|were|of)?\s*(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?)",
            text,
            re.IGNORECASE,
        )
        if net_inc_match:
            data["net_income_text"] = net_inc_match.group(0).strip()
            data["net_income_usd"] = self._parse_dollar_amount(net_inc_match.group(1))

        # 3. Diluted EPS
        eps_match = re.search(
            r"(?:diluted\s+(?:earnings|loss)\s+per\s+share|diluted\s+eps)\s*(?:was|were|of)?\s*\$\s*([\d\.]+)",
            text,
            re.IGNORECASE,
        )
        if eps_match:
            eps_str = eps_match.group(1).rstrip(".")
            try:
                data["eps_diluted"] = float(eps_str)
            except ValueError:
                pass


        # 4. Cash and Cash Equivalents
        cash_match = re.search(
            r"(?:cash\s+and\s+cash\s+equivalents?)\s*(?:was|were|of)?\s*(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?)",
            text,
            re.IGNORECASE,
        )
        if cash_match:
            data["cash_and_equivalents_usd"] = self._parse_dollar_amount(cash_match.group(1))

        # 5. Executive / Director Event Extraction (Item 5.02 or disclosures)
        exec_match = re.search(
            r"(?:departure\s+of\s+directors|appointment\s+of|resignation\s+of|retired\s+from\s+the\s+board|appointed\s+(?:mr\.|ms\.|dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+)",
            text,
            re.IGNORECASE,
        )
        if exec_match:
            # Extract surrounding context sentence
            start = max(0, exec_match.start() - 50)
            end = min(len(text), exec_match.end() + 250)
            data["executive_event_summary"] = text[start:end].replace("\n", " ").strip()

        # 6. Business Description Excerpt
        if "ITEM_1_BUSINESS" in sections:
            data["business_description_sample"] = sections["ITEM_1_BUSINESS"][:400].replace("\n", " ").strip()

        return data
