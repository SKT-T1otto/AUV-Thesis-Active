#!/usr/bin/env bash
set -euo pipefail
OLD="${1:?tracking result directory required}"
BASE="${2:?fresh output directory required}"
[[ -n "$OLD" && -n "$BASE" && -f "$OLD/plan/plan.json" && -f "$OLD/both/completed.json" ]]
[[ ! -e "$BASE" ]]
mkdir -p "$BASE"
printf 'BASE=%q\n' "$BASE" > "$BASE/session.env"
python -u -B -m tools.r123_task_sync_experiment prepare --baseline-root "$OLD" --output "$BASE/plan" --execute 2>&1 | tee "$BASE/prepare.log"
python -u -B -m tools.r123_task_sync_experiment validate --root "$BASE/plan" --output "$BASE/validation" --native-smoke --execute 2>&1 | tee "$BASE/validation.log"
echo "PREPARED: $BASE ; full evaluation has not started"
