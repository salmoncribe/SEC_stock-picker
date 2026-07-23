#!/bin/zsh
# Extract company relationship edges across the priced universe, unattended.
#
# One pass processes every candidate section; the collector is resumable
# (skip-extracted) and flushes per batch, so the supervisor's only job is to
# ensure ollama is up and restart the run if the model or network drops it.
# Ends when no candidate sections remain.
#
# Usage: scripts/extract_relationships.sh [max_passes]
set -u

cd "$(dirname "$0")/.." || exit 1

MAX_PASSES=${1:-50}
LOG="logs/extract_relationships.log"
mkdir -p logs

OLLAMA_MODELS=/Volumes/Extreme/home-migrated/ollama-models
OLLAMA_BIN=/opt/homebrew/opt/ollama/bin/ollama

ensure_ollama() {
  curl -s http://localhost:11434/api/version >/dev/null 2>&1 && return 0
  echo "--- starting ollama ($(date)) ---" >>"$LOG"
  OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_FLASH_ATTENTION=1 nohup "$OLLAMA_BIN" serve \
    >>logs/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -s http://localhost:11434/api/version >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

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
            SELECT count(*) FROM filing_sections s
            JOIN companies c ON c.cik = s.cik
            WHERE s.item_code = '1'
              AND c.ticker IN (SELECT DISTINCT symbol FROM daily_returns)
              AND s.text_path IS NOT NULL
              AND s.accession_number = (
                  SELECT s2.accession_number FROM filing_sections s2
                  WHERE s2.cik = s.cik AND s2.item_code = '1'
                  ORDER BY s2.report_date DESC NULLS LAST, s2.accession_number LIMIT 1
              )
              AND NOT EXISTS (
                  SELECT 1 FROM company_edges e WHERE e.accession_number = s.accession_number
              )
            """
        ).fetchone()[0]
    )
PY
}

echo "=== extraction started $(date) ===" >>"$LOG"

for pass in $(seq 1 "$MAX_PASSES"); do
  ensure_ollama || { echo "ollama unavailable, retrying ($(date))" >>"$LOG"; sleep 30; continue; }

  left=$(remaining)
  echo "--- pass $pass: $left sections remaining ($(date)) ---" >>"$LOG"
  if [ "$left" = "0" ]; then
    echo "=== caught up after $((pass - 1)) passes $(date) ===" >>"$LOG"
    exit 0
  fi

  uv run market-intelligence signals extract-relationships --items 1 --latest-only >>"$LOG" 2>&1
  status=$?
  echo "--- pass $pass exited $status ($(date)) ---" >>"$LOG"
  [ "$status" -ne 0 ] && sleep 30
done

echo "=== stopped after $MAX_PASSES passes, $(remaining) remaining $(date) ===" >>"$LOG"
