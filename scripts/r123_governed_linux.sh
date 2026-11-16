#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
OLD="${1:?Pass the previous rule comparison run directory}"
BASE="${2:?Pass a fresh output directory under work/collision_terminal}"
mkdir -p "$BASE"
printf 'BASE=%q\n' "$BASE" > "$BASE/session.env"
echo '[1/4] Current Linux validation: two cases, at most64 steps, no training'
python -B -m tools.validate_r123_governed --require-linux --native-smoke --output "$BASE/validation" --execute
echo '[2/4] Import previous six scenes and no_recovery results'
python -B -m tools.r123_governed_experiment prepare --previous "$OLD" --output "$BASE/plan" --execute
echo '[3/4] Governed rule: six full episodes, at most2400 steps, no training'
python -B -m tools.r123_governed_experiment evaluate --root "$BASE/plan" \
  --validation "$BASE/validation/validation.json" --output "$BASE/governed" --execute
echo '[4/4] Paired report'
python -B -m tools.r123_governed_experiment report --root "$BASE/plan" \
  --result "$BASE/governed" --output "$BASE/report" --execute
echo "GOVERNED COMPARISON COMPLETED: $BASE/report/report.json"
