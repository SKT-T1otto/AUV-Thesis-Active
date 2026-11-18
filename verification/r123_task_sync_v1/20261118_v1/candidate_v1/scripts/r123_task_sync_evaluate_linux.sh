#!/usr/bin/env bash
set -euo pipefail
BASE="${1:?prepared task-sync output directory required}"
[[ -n "$BASE" && -f "$BASE/plan/plan.json" && -f "$BASE/validation/validation.json" ]]
[[ ! -e "$BASE/sync" && ! -e "$BASE/evaluate.log" ]]
python -u -B -m tools.r123_task_sync_experiment evaluate --root "$BASE/plan" --validation "$BASE/validation/validation.json" --output "$BASE/sync" --execute 2>&1 | tee "$BASE/evaluate.log"
python -u -B -m tools.r123_task_sync_experiment report --root "$BASE/plan" --results "$BASE/sync" --output "$BASE/report" --execute 2>&1 | tee "$BASE/report.log"
[[ -f "$BASE/sync/completed.json" && -f "$BASE/report/report.json" ]]
echo "ALL THREE SCENES COMPLETED: $BASE/report/report.json"
