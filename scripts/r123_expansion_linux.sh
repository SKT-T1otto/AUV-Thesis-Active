#!/usr/bin/env bash
set -euo pipefail
VALIDATED="${1:?completed original task-sync Linux run required}"
BASE="${2:?fresh output directory required}"
[[ -f "$VALIDATED/validation/validation.json" ]]
[[ ! -e "$BASE" ]]
mkdir -p "$BASE"
printf 'BASE=%q\n' "$BASE" > "$BASE/session.env"
python -u -B -m tools.r123_expansion prepare --output "$BASE/plan" --execute 2>&1 | tee "$BASE/prepare.log"
python -u -B -m tools.r123_expansion check --root "$BASE/plan" --output "$BASE/checks" --execute 2>&1 | tee "$BASE/checks.log"
python -u -B -m tools.r123_expansion evaluate --root "$BASE/plan" --checks "$BASE/checks" --validated-run "$VALIDATED" --output "$BASE/evaluation" --execute 2>&1 | tee "$BASE/evaluate.log"
python -u -B -m tools.r123_expansion report --root "$BASE/plan" --results "$BASE/evaluation" --output "$BASE/report" --execute 2>&1 | tee "$BASE/report.log"
[[ -f "$BASE/evaluation/completed.json" && -f "$BASE/report/report.json" ]]
echo "ALL 24 PAIRED EXPANSION SCENES COMPLETED: $BASE/report/report.json"
