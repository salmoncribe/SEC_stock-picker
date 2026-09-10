"""B2B Corporate Purchase & Acquisition Extractor."""

from __future__ import annotations

import re
from typing import Dict, Any, List


class TransactionExtractor:
    """Extracts corporate purchases, acquisitions, and material B2B commercial agreements from filing text."""

    def _parse_dollar_amount(self, text_snippet: str) -> float | None:
        """Parse dollar amount from text like '$ 1.5 billion', '$ 450 million'."""
        match = re.search(r"\$\s*([\d\,\.]+)\s*(billion|million|thousand)?", text_snippet, re.IGNORECASE)
        if not match:
            return None
        val_str = match.group(1).replace(",", "").rstrip(".")
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

    def extract_transactions(self, text: str, sections: Dict[str, str], ticker: str = "") -> List[Dict[str, Any]]:
        """Extract corporate purchases, asset acquisitions, and M&A deals."""
        transactions: List[Dict[str, Any]] = []
        if not text:
            return transactions

        # Regex patterns for corporate purchases and acquisitions
        patterns = [
            # Pattern 1: Acquired [Company] for $[Amount]
            (
                r"(?:acquired|acquire|purchased|purchase)\s+([A-Z][A-Za-z0-9\s\,\.\&]{2,40}?)\s+(?:for|at)\s+(?:a\s+purchase\s+price\s+of\s+)?(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?)",
                "Acquisition",
            ),
            # Pattern 2: Entered into an agreement to acquire [Company] for $[Amount]
            (
                r"(?:entered\s+into\s+(?:an?\s+)?(?:definitive\s+)?agreement\s+(?:to\s+acquire|with))\s+([A-Z][A-Za-z0-9\s\,\.\&]{2,40}?)\s+(?:for|valued\s+at)\s+(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?)",
                "Material Purchase Agreement",
            ),
            # Pattern 3: Completed the acquisition of [Company]
            (
                r"(?:completed\s+the\s+acquisition\s+of)\s+([A-Z][A-Za-z0-9\s\,\.\&]{2,40}?)(?:\s+(?:for|valued\s+at)\s+(\$\s*[\d\,\.]+\s*(?:billion|million|thousand)?))?",
                "Acquisition",
            ),
        ]

        seen_targets = set()

        for pattern, deal_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                target_raw = groups[0].strip()
                price_snippet = groups[1] if len(groups) > 1 and groups[1] else ""

                # Filter out generic words
                target_clean = re.sub(r"\s+(?:for|with|and|in|on|at|by|the|a|an)$", "", target_raw, flags=re.IGNORECASE)
                if len(target_clean) < 3 or target_clean.lower() in {"the company", "a company", "certain assets", "all assets", "substantially all"}:
                    continue

                if target_clean.lower() in seen_targets:
                    continue
                seen_targets.add(target_clean.lower())

                price_val = self._parse_dollar_amount(price_snippet) if price_snippet else None

                # Determine consideration type (Cash / Stock / Mixed)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 150)
                context = text[start:end].replace("\n", " ").strip()

                consideration = "Unknown"
                context_lower = context.lower()
                if "cash" in context_lower and ("stock" in context_lower or "shares" in context_lower):
                    consideration = "Mixed (Cash & Stock)"
                elif "cash" in context_lower:
                    consideration = "Cash"
                elif "stock" in context_lower or "shares" in context_lower:
                    consideration = "Stock"

                transactions.append({
                    "buyer_ticker": ticker,
                    "seller_target_name": target_clean,
                    "transaction_type": deal_type,
                    "purchase_price_usd": price_val,
                    "consideration_type": consideration,
                    "context_summary": context,
                })

        return transactions
