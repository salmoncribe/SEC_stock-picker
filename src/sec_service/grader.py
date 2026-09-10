"""Filing Scoring and Grading Engine."""

from __future__ import annotations

from typing import Dict, Any


class FilingGrader:
    """Grades SEC filings based on management sentiment, transparency/clarity, and structural financial signals."""

    def compute_grade(self, metrics: Dict[str, Any], sections: Dict[str, str], form: str = "10-Q") -> Dict[str, Any]:
        """Compute individual dimension scores, composite score, letter grade, and summary notes."""
        word_count = metrics.get("word_count", 0)
        net_sentiment = metrics.get("net_sentiment_score", 0.0)
        fog_index = metrics.get("fog_index", 15.0)
        litigious_count = metrics.get("litigious_count", 0)
        uncertainty_count = metrics.get("uncertainty_count", 0)

        # 1. Management Sentiment & Tone Score (0 - 100)
        # Net sentiment range: [-1.0, 1.0]. Standard baseline is 50.
        sentiment_base = 50.0 + (net_sentiment * 100.0)
        # Penalize high litigious/uncertainty ratio
        unc_ratio = (uncertainty_count / max(1, word_count)) * 100.0
        lit_ratio = (litigious_count / max(1, word_count)) * 100.0

        sentiment_score = sentiment_base - (unc_ratio * 5.0) - (lit_ratio * 10.0)
        sentiment_score = max(0.0, min(100.0, sentiment_score))

        # 2. Transparency & Readability Score (0 - 100)
        # Ideal Fog index for corporate reports is ~12-16. > 20 indicates high obfuscation.
        if fog_index <= 14.0:
            transparency_score = 90.0
        elif fog_index <= 18.0:
            transparency_score = 80.0
        elif fog_index <= 22.0:
            transparency_score = 65.0
        else:
            transparency_score = 50.0

        if word_count > 80000:  # Excessive document bloat penalty
            transparency_score -= 10.0

        transparency_score = max(0.0, min(100.0, transparency_score))

        # 3. Financial Structure & Completeness Score (0 - 100)
        financial_score = 70.0
        if "ITEM_7_MDA" in sections or "PART1_ITEM2_MDA" in sections:
            financial_score += 15.0
        if "ITEM_1A_RISK_FACTORS" in sections or "PART2_ITEM1A_RISK" in sections:
            financial_score += 15.0

        financial_score = max(0.0, min(100.0, financial_score))

        # 4. Composite Overall Score (Weighted)
        overall_score = (0.40 * sentiment_score) + (0.30 * transparency_score) + (0.30 * financial_score)
        overall_score = round(overall_score, 1)

        # 5. Letter Grade Conversion
        if overall_score >= 93.0:
            letter_grade = "A+"
        elif overall_score >= 90.0:
            letter_grade = "A"
        elif overall_score >= 85.0:
            letter_grade = "B+"
        elif overall_score >= 80.0:
            letter_grade = "B"
        elif overall_score >= 75.0:
            letter_grade = "C+"
        elif overall_score >= 70.0:
            letter_grade = "C"
        elif overall_score >= 60.0:
            letter_grade = "D"
        else:
            letter_grade = "F"

        # 6. Generate Summary Notes
        notes = []
        if net_sentiment > 0.1:
            notes.append("Positive management tone in filing discussion.")
        elif net_sentiment < -0.05:
            notes.append("Cautious or defensive tone detected in management text.")
        else:
            notes.append("Neutral management tone.")

        if fog_index > 20.0:
            notes.append("High readability complexity (Fog index > 20).")
        else:
            notes.append("Clear disclosure readability.")

        if lit_ratio > 0.5:
            notes.append("Elevated litigious terms mentioned in risk sections.")

        return {
            "overall_grade": letter_grade,
            "overall_score": overall_score,
            "sentiment_score": round(sentiment_score, 1),
            "transparency_score": round(transparency_score, 1),
            "financial_score": round(financial_score, 1),
            "summary_notes": "; ".join(notes),
        }
