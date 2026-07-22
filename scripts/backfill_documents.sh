#!/bin/zsh
# Run the 10-K/10-Q document backfill to completion, unattended.
#
# The collector is resumable (it skips filings already in filing_documents)
# and durable (it flushes every 100 filings), so the supervisor's job is only
# to restart it when the network or SEC drops the run partway. Each pass
# resumes where the last one stopped; the loop ends when nothing is left.
#
# Usage: scripts/backfill_documents.sh [max_passes]
set -u

cd "$(dirname "$0")/.." || exit 1

MAX_PASSES=${1:-100}
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/backfill_documents.log"

remaining() {
  uv run python - <<'PY'
import os
import duckdb
from dotenv import load_dotenv

load_dotenv(".env")
home = os.environ.get("MARKET_INTELLIGENCE_HOME") or "."
db = os.path.join(home, "data", "database", "market_intelligence.duckdb")
try:
    con = duckdb.connect(db, read_only=True)
except Exception:
    print(-1)
else:
    print(
        con.execute(
            """
            SELECT count(*) FROM filings
            WHERE form IN ('10-K', '10-K/A', '10-Q', '10-Q/A')
              AND primary_document IS NOT NULL AND primary_document <> ''
              AND validation_status <> 'rejected'
              AND NOT EXISTS (
                  SELECT 1 FROM filing_documents d
                  WHERE d.accession_number = filings.accession_number
              )
            """
        ).fetchone()[0]
    )
PY
}

echo "=== backfill started $(date) ===" >>"$LOG"

for pass in $(seq 1 "$MAX_PASSES"); do
  left=$(remaining)
  echo "--- pass $pass: $left filings remaining ($(date)) ---" >>"$LOG"

  if [ "$left" = "0" ]; then
    echo "=== caught up after $((pass - 1)) passes $(date) ===" >>"$LOG"
    exit 0
  fi

  uv run market-intelligence sec ingest-documents >>"$LOG" 2>&1
  status=$?
  echo "--- pass $pass exited $status ($(date)) ---" >>"$LOG"

  # A pass that fails immediately and repeatedly would otherwise spin; give
  # the far side room to recover before trying again.
  [ "$status" -ne 0 ] && sleep 60
done

echo "=== stopped after $MAX_PASSES passes, $(remaining) remaining $(date) ===" >>"$LOG"
