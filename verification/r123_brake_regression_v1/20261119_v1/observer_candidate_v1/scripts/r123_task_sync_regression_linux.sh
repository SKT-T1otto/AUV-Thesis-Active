#!/usr/bin/env bash
set -euo pipefail
OLD="${1:?original tracking result required}"
VALIDATED="${2:?completed task-sync run with Linux validation required}"
BASE="${3:?fresh regression output required}"
[[ -n "$OLD" && -n "$VALIDATED" && -n "$BASE" ]]
[[ -f "$OLD/both/completed.json" && -f "$VALIDATED/validation/validation.json" ]]
[[ ! -e "$BASE" ]]
mkdir -p "$BASE"
printf 'BASE=%q\n' "$BASE" > "$BASE/session.env"
python -u -B -m tools.r123_task_sync_regression check --output "$BASE/checks" --execute 2>&1 | tee "$BASE/checks.log"
python -u -B -m tools.r123_task_sync_regression prepare --baseline-root "$OLD" --validated-run "$VALIDATED" --output "$BASE/plan" --execute 2>&1 | tee "$BASE/prepare.log"
python -u -B -m tools.r123_task_sync_regression evaluate --root "$BASE/plan" --checks "$BASE/checks" --output "$BASE/sync" --execute 2>&1 | tee "$BASE/evaluate.log"
python -u -B -m tools.r123_task_sync_regression report --root "$BASE/plan" --results "$BASE/sync" --output "$BASE/report" --execute 2>&1 | tee "$BASE/report.log"
[[ -f "$BASE/sync/completed.json" && -f "$BASE/report/report.json" ]]
echo "ALL THREE REGRESSION SCENES COMPLETED: $BASE/report/report.json"
