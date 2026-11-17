#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
OLD="${1:?Pass the completed governed run directory}"
BASE="${2:?Pass a fresh directory under work/collision_terminal}"
if [ -e "$BASE" ]; then
  echo "Output already exists: $BASE" >&2
  exit 1
fi
mkdir -p "$BASE"
printf 'BASE=%q\n' "$BASE" > "$BASE/session.env"
trap 'rc=$?; echo "[tracking prepare] exit_code=$rc"' EXIT
echo '[1/2] Four short validation cases, at most128 physical steps, no training'
python -B -m tools.validate_r123_tracking --require-linux --native-smoke \
  --output "$BASE/validation" --execute
echo '[2/2] Import the same six scenes and completed governed baseline'
python -B -m tools.r123_tracking_experiment prepare --previous "$OLD" \
  --output "$BASE/plan" --execute
echo "TRACKING PREPARATION COMPLETED: $BASE/plan/plan.json"
echo 'No full episodes or training started. Evaluate each requested arm explicitly.'
