"""Filing Information & Sentiment Extractor."""

from __future__ import annotations

import re
from typing import Dict, Any

from sec_service.financial_extractor import FinancialExtractor



class FilingExtractor:
    """Extracts text statistics, sentiment scores, readability metrics, and quantitative signals from filing text."""

    def __init__(self) -> None:
        self.fin_extractor = FinancialExtractor()

    POSITIVE_WORDS = {
        "increase", "increased", "increasing", "growth", "strong", "profitable", "profitability",
        "expand", "expanded", "expansion", "gain", "gains", "gained", "higher", "outperform",
        "favorable", "momentum", "robust", "solid", "record", "advantage", "benefit", "benefits",
        "benefited", "excel", "improved", "improvement", "surpassed", "exceeded", "exceeding"
    }

    NEGATIVE_WORDS = {
        "decline", "declined", "declining", "loss", "losses", "decrease", "decreased", "decreasing",
        "drop", "dropped", "fall", "fallen", "risk", "risks", "deficit", "adverse", "adversely",
        "impair", "impaired", "impairment", "weak", "weakness", "downturn", "lawsuit", "default",
        "write-off", "failure", "pressure", "pressures", "challenge", "challenges", "penalty",
        "penalties", "deterioration", "slowdown", "curtail", "uncertainty"
    }

    UNCERTAINTY_WORDS = {
        "approximate", "approximately", "uncertain", "uncertainty", "depend", "dependent",
        "project", "projection", "projections", "assume", "assumption", "assumptions",
        "fluctuate", "fluctuation", "estimate", "estimates", "volatile", "volatility",
        "condition", "conditions", "potential", "potentially", "maybe", "possible", "possibly",
        "unknown", "unpredictable", "subject"
    }

    LITIGIOUS_WORDS = {
        "claim", "claims", "dispute", "disputes", "litigation", "lawsuit", "lawsuits",
        "investigation", "investigations", "proceeding", "proceedings", "arbitrate",
        "arbitration", "breach", "fine", "fines", "penalty", "penalties", "injunction",
        "alleged", "allegation", "subpoena", "verdict"
    }

    def count_syllables(self, word: str) -> int:
        """Estimate syllable count of an English word."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
        word = re.sub(r'^y', '', word)
        matches = re.findall(r'[aeiouy]{1,2}', word)
        return max(1, len(matches))

    def extract_metrics(self, text: str, sections: Dict[str, str] | None = None, form: str = "") -> Dict[str, Any]:
        """Compute textual metrics, sentiment metrics, readability indicators, and concrete financial numbers from filing text."""
        sections = sections or {}
        fin_data = self.fin_extractor.extract_financial_data(text, sections, form=form)

        if not text:
            res = {
                "word_count": 0,
                "sentence_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "uncertainty_count": 0,
                "litigious_count": 0,
                "net_sentiment_score": 0.0,
                "fog_index": 0.0,
                "flesch_reading_ease": 0.0,
                "complex_word_ratio": 0.0,
            }
            res.update(fin_data)
            return res

        # Tokenize words and sentences
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        sentences = [s for s in re.split(r"[\.\!\?]+", text) if len(s.strip()) > 5]

        word_count = len(words)
        sentence_count = max(1, len(sentences))

        if word_count == 0:
            res = {
                "word_count": 0,
                "sentence_count": sentence_count,
                "positive_count": 0,
                "negative_count": 0,
                "uncertainty_count": 0,
                "litigious_count": 0,
                "net_sentiment_score": 0.0,
                "fog_index": 0.0,
                "flesch_reading_ease": 0.0,
                "complex_word_ratio": 0.0,
            }
            res.update(fin_data)
            return res

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        unc_count = sum(1 for w in words if w in self.UNCERTAINTY_WORDS)
        lit_count = sum(1 for w in words if w in self.LITIGIOUS_WORDS)

        # Net sentiment score normalized between -1.0 and +1.0
        total_sentiment_words = pos_count + neg_count
        net_sentiment = (pos_count - neg_count) / (total_sentiment_words + 1)

        # Syllables and complex words (>= 3 syllables)
        syllable_counts = [self.count_syllables(w) for w in words]
        total_syllables = sum(syllable_counts)
        complex_words = sum(1 for s in syllable_counts if s >= 3)
        complex_word_ratio = complex_words / word_count

        avg_words_per_sentence = word_count / sentence_count
        fog_index = 0.4 * (avg_words_per_sentence + 100 * complex_word_ratio)
        flesch_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (total_syllables / word_count))
        flesch_ease = max(0.0, min(100.0, flesch_ease))

        res = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "uncertainty_count": unc_count,
            "litigious_count": lit_count,
            "net_sentiment_score": round(net_sentiment, 4),
            "fog_index": round(fog_index, 2),
            "flesch_reading_ease": round(flesch_ease, 2),
            "complex_word_ratio": round(complex_word_ratio, 4),
        }
        res.update(fin_data)
        return res

