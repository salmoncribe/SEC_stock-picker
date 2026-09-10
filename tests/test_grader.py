"""Unit tests for SEC filing parser, extractor, grader, and database tables."""

from __future__ import annotations

import duckdb
from sec_service import database
from sec_service.parser import FilingParser
from sec_service.extractor import FilingExtractor
from sec_service.grader import FilingGrader


def test_database_init_and_tables():
    con = duckdb.connect(":memory:")
    database.init_db(con)

    counts = database.get_table_counts(con)
    assert counts["companies"] == 0
    assert counts["filings"] == 0
    assert counts["filing_documents"] == 0
    assert counts["filing_sections"] == 0
    assert counts["filing_metrics"] == 0
    assert counts["filing_grades"] == 0


def test_parser_html_and_sections():
    parser = FilingParser()
    raw_html = """
    <html>
        <body>
            <h1>Item 1. Business</h1>
            <p>We are a leading provider of innovative hardware products.</p>
            <h1>Item 1A. Risk Factors</h1>
            <p>Competitive pressures and supply chain disruptions may adversely affect growth.</p>
            <h1>Item 7. Management's Discussion and Analysis</h1>
            <p>Revenue increased by 25% year over year with strong profitability.</p>
        </body>
    </html>
    """
    clean_text = parser.clean_html(raw_html)
    assert "Item 1. Business" in clean_text
    assert "Item 1A. Risk Factors" in clean_text

    sections = parser.extract_sections(clean_text)
    assert "ITEM_1_BUSINESS" in sections
    assert "ITEM_1A_RISK_FACTORS" in sections
    assert "ITEM_7_MDA" in sections


def test_extractor_sentiment_and_readability():
    extractor = FilingExtractor()
    sample_text = (
        "Revenue increased significantly and profitability was strong due to robust demand and growth. "
        "However, potential risks, uncertainty, and competitive pressures remain adverse factors."
    )
    metrics = extractor.extract_metrics(sample_text)

    assert metrics["word_count"] > 10
    assert metrics["positive_count"] >= 3  # increased, profitability, strong, robust, growth
    assert metrics["negative_count"] >= 2  # risks, pressures, adverse
    assert metrics["uncertainty_count"] >= 2  # potential, uncertainty
    assert "net_sentiment_score" in metrics
    assert "fog_index" in metrics


def test_grader_scoring():
    grader = FilingGrader()
    metrics = {
        "word_count": 5000,
        "sentence_count": 200,
        "positive_count": 40,
        "negative_count": 10,
        "uncertainty_count": 5,
        "litigious_count": 1,
        "net_sentiment_score": 0.58,
        "fog_index": 13.5,
        "flesch_reading_ease": 45.0,
    }
    sections = {
        "ITEM_1_BUSINESS": "Business text...",
        "ITEM_7_MDA": "MDA text...",
        "ITEM_1A_RISK_FACTORS": "Risk text...",
    }

    result = grader.compute_grade(metrics, sections, form="10-K")
    assert result["overall_grade"] in ["A+", "A", "B+", "B"]
    assert result["overall_score"] > 80.0
    assert "Positive management tone" in result["summary_notes"]
