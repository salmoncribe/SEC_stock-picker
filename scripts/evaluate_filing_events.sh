#!/bin/zsh
# Wait for the document backfill to release the database, then score the 8-K
# item-code events through the validation gate.
#
# The dependency here is the DuckDB write lock, not the data: 8-K events need
# only `events` and `daily_returns`, both of which are already complete. So
# this waits for the backfill to finish for exclusivity, and runs regardless of
# whether that backfill succeeded.
set -u

cd "$(dirname "$0")/.." || exit 1

LOG="logs/evaluate_filing_events.log"
mkdir -p logs

echo "=== waiting for the document backfill to release the database $(date) ===" >>"$LOG"

# Poll rather than wait on a pid, so this survives being started before,
# during, or after the backfill.
while pgrep -f "backfill_documents" >/dev/null || pgrep -f "ingest-documents" >/dev/null; do
  sleep 60
done

echo "=== database free, building samples $(date) ===" >>"$LOG"
uv run market-intelligence signals build-dataset --event-types filing_item >>"$LOG" 2>&1
build_status=$?
echo "--- build-dataset exited $build_status ($(date)) ---" >>"$LOG"

if [ "$build_status" -ne 0 ]; then
  echo "=== stopping: no samples to judge $(date) ===" >>"$LOG"
  exit "$build_status"
fi

echo "=== evaluating every cell $(date) ===" >>"$LOG"
uv run market-intelligence signals evaluate >>"$LOG" 2>&1
echo "--- evaluate exited $? ($(date)) ---" >>"$LOG"
echo "=== done $(date) ===" >>"$LOG"
