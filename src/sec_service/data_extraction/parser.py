"""SEC Filing HTML/Text Parser and Section Extractor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class FilingParser:
    """Parses raw SEC filing HTML/text files into clean text and extracted sections."""

    # Key section headers for 10-K, 10-Q, 8-K
    SECTION_PATTERNS = {
        "ITEM_1_BUSINESS": r"(?:item\s+1[\.\:]?\s+business|item\s+1\.\s+description\s+of\s+business)",
        "ITEM_1A_RISK_FACTORS": r"(?:item\s+1a[\.\:]?\s+risk\s+factors)",
        "ITEM_7_MDA": r"(?:item\s+7[\.\:]?\s+management(?:'|\u2019)?s\s+discussion\s+and\s+analysis)",
        "ITEM_8_FINANCIALS": r"(?:item\s+8[\.\:]?\s+financial\s+statements)",
        "PART1_ITEM2_MDA": r"(?:part\s+i\s+.*?item\s+2[\.\:]?\s+management(?:'|\u2019)?s\s+discussion)",
        "PART2_ITEM1A_RISK": r"(?:part\s+ii\s+.*?item\s+1a[\.\:]?\s+risk\s+factors)",
        "ITEM_202_EARNINGS": r"(?:item\s+2\.02[\.\:]?\s+results\s+of\s+operations)",
    }

    def clean_html(self, raw_content: str) -> str:
        """Strip HTML markup, scripts, and redundant spacing to produce clean readable text."""
        if not raw_content:
            return ""

        # Quick check if content looks like HTML
        if "<html" in raw_content.lower() or "<body" in raw_content.lower() or "<div" in raw_content.lower() or "<?xml" in raw_content.lower():
            soup = BeautifulSoup(raw_content, "lxml")
            for element in soup(["script", "style", "head", "title", "meta", "noscript", "ix:header"]):
                element.extract()
            text = soup.get_text(separator="\n")
        else:
            text = raw_content


        # Normalize line breaks & excessive spaces
        text = re.sub(r"\r\n|\r", "\n", text)
        lines = [line.strip() for line in text.split("\n")]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned

    def parse_file(self, raw_file_path: str | Path) -> Tuple[str, Dict[str, str]]:
        """Read file from path, clean text, and extract identifiable sections.

        Returns:
            Tuple of (full_clean_text, dictionary_of_sections)
        """
        path = Path(raw_file_path)
        if not path.exists():
            return "", {}

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return "", {}

        clean_text = self.clean_html(content)
        sections = self.extract_sections(clean_text)
        return clean_text, sections

    def extract_sections(self, clean_text: str) -> Dict[str, str]:
        """Extract standard filing sections using header regex matches."""
        sections: Dict[str, str] = {}
        matches = []

        # Find all section header positions
        for sec_name, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, clean_text, re.IGNORECASE):
                matches.append((match.start(), sec_name, match.group(0)))

        if not matches:
            # Fallback if no specific section headers matched
            sections["FULL_DOCUMENT"] = clean_text[:100000]
            return sections

        # Sort matches by position in text
        matches.sort(key=lambda x: x[0])

        # Slice text between consecutive section headers
        for i in range(len(matches)):
            start_pos, sec_name, header = matches[i]
            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(clean_text)
            sec_text = clean_text[start_pos:end_pos].strip()
            if len(sec_text) > 20:  # Only save non-trivial sections
                sections[sec_name] = sec_text[:50000]  # Cap section text size


        if not sections:
            sections["FULL_DOCUMENT"] = clean_text[:100000]

        return sections
