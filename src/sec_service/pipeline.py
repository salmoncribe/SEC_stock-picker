"""Filing Grading Pipeline Service."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Any


from sec_service.config import Config
from sec_service import database
from sec_service.parser import FilingParser
from sec_service.extractor import FilingExtractor
from sec_service.grader import FilingGrader
from sec_service.transaction_extractor import TransactionExtractor

logger = logging.getLogger(__name__)


class FilingGradingPipeline:
    """Orchestrates parsing, metric extraction, and grading for filings stored in DuckDB."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.parser = FilingParser()
        self.extractor = FilingExtractor()
        self.grader = FilingGrader()
        self.txn_extractor = TransactionExtractor()


    def run_grading_pass(self, max_filings: int = 50) -> Dict[str, Any]:
        """Fetch ungraded filings from database, parse, extract metrics, grade, and write back results."""
        with database.connection(self.config.db_path) as con:
            database.init_db(con)
            ungraded = database.get_ungraded_filings(con, limit=max_filings)

            if not ungraded:
                logger.info("No ungraded filings found in database.")
                return {"graded_count": 0, "grades": []}

            logger.info(f"Processing and grading {len(ungraded)} SEC filings...")
            graded_results = []

            for item in ungraded:
                acc = item["accession_number"]
                cik = item["cik"]
                ticker = item["ticker"]
                form = item["form"]
                raw_path = item["raw_file_path"]

                # 1. Parse text & sections
                clean_text, sections = self.parser.parse_file(raw_path)
                if not clean_text:
                    logger.warning(f"Could not parse content for {acc} at {raw_path}")
                    continue

                # 2. Store parsed sections
                sec_rows = [
                    (acc, sec_name, sec_name.replace("_", " "), sec_content, len(sec_content.split()))
                    for sec_name, sec_content in sections.items()
                ]
                database.upsert_sections(con, sec_rows)

                # 3. Extract metrics
                metrics = self.extractor.extract_metrics(clean_text, sections=sections, form=form)
                metric_rows = [
                    (acc, name, float(val) if isinstance(val, (int, float)) else None, str(val))
                    for name, val in metrics.items()
                ]
                database.upsert_metrics(con, metric_rows)

                # 3b. Extract corporate B2B purchases & M&A transactions
                txns = self.txn_extractor.extract_transactions(clean_text, sections, ticker=ticker)
                txn_rows = [
                    (acc, t["buyer_ticker"], t["seller_target_name"], t["transaction_type"], t["purchase_price_usd"], t["consideration_type"], t["context_summary"])
                    for t in txns
                ]
                database.upsert_transactions(con, txn_rows)



                # 4. Compute grade
                grade_res = self.grader.compute_grade(metrics, sections, form)
                database.upsert_grade(
                    con,
                    accession_number=acc,
                    cik=cik,
                    ticker=ticker,
                    form=form,
                    overall_grade=grade_res["overall_grade"],
                    overall_score=grade_res["overall_score"],
                    sentiment_score=grade_res["sentiment_score"],
                    transparency_score=grade_res["transparency_score"],
                    financial_score=grade_res["financial_score"],
                    summary_notes=grade_res["summary_notes"],
                )

                graded_results.append({
                    "ticker": ticker,
                    "form": form,
                    "accession_number": acc,
                    "grade": grade_res["overall_grade"],
                    "score": grade_res["overall_score"],
                    "sentiment": grade_res["sentiment_score"],
                    "transparency": grade_res["transparency_score"],
                    "summary": grade_res["summary_notes"],
                })

            sync_id = uuid.uuid4().hex[:8]
            database.log_sync(con, sync_id, "run_grading_pass", len(graded_results), "success")
            return {"graded_count": len(graded_results), "grades": graded_results}

