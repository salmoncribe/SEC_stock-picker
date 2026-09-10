"""Downloader service orchestrator."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sec_service.config import Config
from sec_service import database
from sec_service.sec_client import SECClient

logger = logging.getLogger(__name__)


def build_filing_url(cik: str, accession_number: str, primary_doc: str) -> str:
    cik10 = str(cik).zfill(10).lstrip("0") or "0"
    acc_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik10}/{acc_nodash}/{primary_doc}"


class SECDownloaderService:
    def __init__(self, config: Config) -> None:
        self.config = config

    def sync_ticker_map(self) -> int:
        """Download latest company ticker map and store in DuckDB."""
        logger.info("Syncing SEC company ticker map...")
        sync_id = uuid.uuid4().hex[:8]
        with SECClient(self.config.sec_user_agent, self.config.requests_per_second) as client:
            ticker_data = client.fetch_company_tickers()

        rows: list[tuple[str, str, str]] = []
        for entry in ticker_data.values():
            if isinstance(entry, dict):
                cik = str(entry.get("cik_str", "")).zfill(10)
                ticker = str(entry.get("ticker", "")).upper()
                title = str(entry.get("title", ""))
                if cik:
                    rows.append((cik, ticker, title))

        with database.connection(self.config.db_path) as con:
            database.init_db(con)
            inserted = database.upsert_companies(con, rows)
            database.log_sync(con, sync_id, "sync_ticker_map", inserted, "success")

        logger.info(f"Synced {inserted} company tickers.")
        return inserted

    def download_filings_for_cik(self, client: SECClient, con: Any, cik: str, limit: int = 50) -> int:
        """Download recent filing metadata for a CIK."""
        cik10 = str(cik).zfill(10)
        try:
            sub = client.fetch_submissions(cik10)
        except Exception as exc:
            logger.warning(f"Could not fetch submissions for CIK {cik10}: {exc}")
            return 0

        recent = sub.get("filings", {}).get("recent", {})
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        rows: list[tuple[str, str, str, str | None, str | None, str | None, str]] = []
        n = min(len(accessions), limit)
        for i in range(n):
            acc = accessions[i]
            form = forms[i]
            f_date = filing_dates[i] if (i < len(filing_dates) and filing_dates[i]) else None
            r_date = report_dates[i] if (i < len(report_dates) and report_dates[i]) else None
            p_doc = primary_docs[i] if (i < len(primary_docs) and primary_docs[i]) else None
            url = build_filing_url(cik10, acc, p_doc) if p_doc else ""

            rows.append((acc, cik10, form, f_date, r_date, p_doc, url))

        return database.upsert_filings(con, rows)

    def download_document(self, client: SECClient, con: Any, cik: str, accession_number: str, primary_doc: str, form: str = "") -> bool:
        """Download raw filing document bytes, save to data/raw/sec/, and log in DB."""
        if not primary_doc:
            return False
        url = build_filing_url(cik, accession_number, primary_doc)
        try:
            doc_bytes = client.fetch_document_bytes(url)
        except Exception as exc:
            logger.warning(f"Failed downloading document {url}: {exc}")
            return False

        # Save to raw disk archive
        acc_clean = accession_number.replace("-", "")
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest_dir = self.config.raw_dir / "filing_document" / acc_clean / today_str
        dest_dir.mkdir(parents=True, exist_ok=True)

        file_path = dest_dir / primary_doc
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(doc_bytes)


        # Log in DB
        database.upsert_document(
            con,
            accession_number=accession_number,
            cik=cik,
            document_name=primary_doc,
            url=url,
            byte_size=len(doc_bytes),
            raw_path=str(file_path),
            form=form,
        )
        return True

    def run_sync_pass(
        self,
        max_companies: int = 500,
        limit_per_company: int = 100,
        docs_per_company: int = 50,
    ) -> dict[str, int]:
        """One complete pass: sync tickers, pick target companies, download filings & primary documents."""
        sync_id = uuid.uuid4().hex[:8]
        tickers_synced = self.sync_ticker_map()

        total_filings = 0
        total_docs = 0

        with database.connection(self.config.db_path) as con:
            database.init_db(con)
            # Pick companies to download
            query = "SELECT cik FROM companies" if max_companies <= 0 else "SELECT cik FROM companies LIMIT ?"
            params = [] if max_companies <= 0 else [max_companies]
            ciks = [row[0] for row in con.execute(query, params).fetchall()]

            with SECClient(self.config.sec_user_agent, self.config.requests_per_second) as client:
                for cik in ciks:
                    f_count = self.download_filings_for_cik(client, con, cik, limit=limit_per_company)
                    total_filings += f_count

                    doc_query = """
                        SELECT f.cik, f.accession_number, f.primary_document, f.form
                        FROM filings f
                        LEFT JOIN filing_documents d ON d.accession_number = f.accession_number
                        WHERE f.cik = ? AND f.primary_document IS NOT NULL AND f.primary_document != '' AND d.accession_number IS NULL
                    """
                    if docs_per_company > 0:
                        doc_query += f" LIMIT {docs_per_company}"

                    unIngested = con.execute(doc_query, [cik]).fetchall()

                    for row in unIngested:
                        if self.download_document(client, con, row[0], row[1], row[2], form=row[3] or ""):
                            total_docs += 1

            database.log_sync(con, sync_id, "run_sync_pass", total_filings + total_docs, "success")

        # Automatically grade all pulled filings immediately
        from sec_service.data_extraction.pipeline import FilingGradingPipeline
        pipeline = FilingGradingPipeline(self.config)

        grade_result = pipeline.run_grading_pass(max_filings=5000)
        graded_count = grade_result.get("graded_count", 0)

        return {
            "tickers_synced": tickers_synced,
            "filings_collected": total_filings,
            "documents_downloaded": total_docs,
            "filings_graded": graded_count,
        }


