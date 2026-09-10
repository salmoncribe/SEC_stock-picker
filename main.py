"""Continuous SEC Downloader Service Entry Point.

Usage:
  python main.py status      - Display database table row counts
  python main.py sync        - Run a single SEC download pass
  python main.py run         - Run the continuous background SEC downloading loop
  python main.py grade       - Parse, extract metrics, and grade downloaded filings
"""

from __future__ import annotations

import logging
import sys
import time

from sec_service.config import Config
from sec_service import database
from sec_service.downloader import SECDownloaderService
from sec_service.pipeline import FilingGradingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sec_service")


def print_status(config: Config) -> None:
    with database.connection(config.db_path) as con:
        database.init_db(con)
        counts = database.get_table_counts(con)

    print("\n=== SEC DuckDB Storage Status ===")
    print(f"Database File: {config.db_path}")
    print(f"Companies:        {counts['companies']:,}")
    print(f"Filings:          {counts['filings']:,}")
    print(f"Filing Documents: {counts['filing_documents']:,}")
    print(f"Filing Sections:  {counts.get('filing_sections', 0):,}")
    print(f"Filing Metrics:   {counts.get('filing_metrics', 0):,}")
    print(f"Filing Grades:    {counts.get('filing_grades', 0):,}")
    print(f"Sync Log Runs:    {counts['sync_log']:,}")
    print("==================================\n")


def run_single_sync(config: Config) -> None:
    service = SECDownloaderService(config)
    result = service.run_sync_pass(max_companies=50, limit_per_company=20)
    print("\n=== SEC Sync Pass Complete ===")
    print(f"Tickers Synced:       {result['tickers_synced']:,}")
    print(f"Filings Collected:    {result['filings_collected']:,}")
    print(f"Documents Downloaded: {result['documents_downloaded']:,}")
    print(f"Filings Graded:       {result.get('filings_graded', 0):,}")
    print("===============================\n")



def run_grading(config: Config, max_filings: int = 50) -> None:
    pipeline = FilingGradingPipeline(config)
    result = pipeline.run_grading_pass(max_filings=max_filings)
    print("\n=== SEC Filing Grading Complete ===")
    print(f"Filings Graded: {result['graded_count']}")
    if result['grades']:
        print("\nGraded Filings Summary:")
        print(f"{'Ticker':<8} {'Form':<6} {'Grade':<6} {'Score':<6} {'Sentiment':<10} {'Summary'}")
        print("-" * 75)
        for g in result['grades']:
            print(f"{g['ticker']:<8} {g['form']:<6} {g['grade']:<6} {g['score']:<6} {g['sentiment']:<10} {g['summary']}")
    print("====================================\n")


def inspect_filing(config: Config, target: str) -> None:
    with database.connection(config.db_path, read_only=True) as con:
        grades = con.execute("""

            SELECT g.ticker, g.form, g.accession_number, g.overall_grade, g.overall_score, g.summary_notes
            FROM filing_grades g
            WHERE UPPER(g.ticker) = UPPER(?) OR g.accession_number = ?
            LIMIT 5
        """, [target, target]).fetchall()

        if not grades:
            print(f"\nNo graded filings found for '{target}'. Run 'python main.py grade' first.")
            return

        for row in grades:
            ticker, form, acc, grade, score, summary = row
            print(f"\n==================================================")
            print(f"  Filing Inspection: {ticker} ({form}) - Accession {acc}")
            print(f"  Grade: {grade} | Composite Score: {score}/100")
            print(f"  Summary Notes: {summary}")
            print(f"==================================================")

            # Print metrics
            metrics = con.execute("""
                SELECT metric_name, metric_value, text_value
                FROM filing_metrics
                WHERE accession_number = ? AND (metric_value IS NOT NULL OR (text_value IS NOT NULL AND text_value != ''))
            """, [acc]).fetchall()
            print("\n  [Extracted Financial & Text Metrics]:")
            for m in metrics:
                val = f"{m[1]:,.2f}" if isinstance(m[1], float) else (m[2] or str(m[1]))
                print(f"    - {m[0]:<28}: {val}")

            # Print section list
            sections = con.execute("""
                SELECT section_name, word_count, left(clean_text, 120)
                FROM filing_sections
                WHERE accession_number = ?
            """, [acc]).fetchall()
            print("\n  [Extracted Document Sections]:")
            for s in sections:
                sample = (s[2] or "").replace("\n", " ")
                print(f"    - {s[0]} ({s[1]} words): {sample}...")
            print("\n")


def run_continuous_service(config: Config, poll_interval_seconds: int = 60) -> None:
    logger.info("Starting SEC Continuous Downloader Service...")
    logger.info(f"Using SEC User-Agent: {config.sec_user_agent}")
    logger.info(f"Target DB: {config.db_path}")
    service = SECDownloaderService(config)

    cycle = 1
    while True:
        logger.info(f"--- Starting Sync Cycle #{cycle} ---")
        try:
            result = service.run_sync_pass(max_companies=50, limit_per_company=20)
            logger.info(
                f"Cycle #{cycle} complete: "
                f"{result['tickers_synced']} tickers synced, "
                f"{result['filings_collected']} filings collected, "
                f"{result['documents_downloaded']} docs downloaded."
            )
        except KeyboardInterrupt:
            logger.info("Service stopped by user.")
            break
        except Exception as exc:
            logger.error(f"Error in cycle #{cycle}: {exc}", exc_info=True)

        logger.info(f"Sleeping for {poll_interval_seconds} seconds before next cycle...")
        time.sleep(poll_interval_seconds)
        cycle += 1


def main() -> None:
    config = Config.load()
    args = sys.argv[1:]

    command = args[0].lower() if args else "status"

    if command == "status":
        print_status(config)
    elif command in ("sync", "once"):
        run_single_sync(config)
    elif command in ("grade", "score"):
        run_grading(config)
    elif command in ("inspect", "show", "view"):
        target = args[1] if len(args) > 1 else "NVDA"
        inspect_filing(config, target)
    elif command in ("run", "service", "daemon"):
        run_continuous_service(config)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [status|sync|grade|inspect <ticker>|run]")
        sys.exit(1)


if __name__ == "__main__":
    main()
