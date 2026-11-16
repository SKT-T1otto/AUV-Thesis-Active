#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
OLD="${1:?Pass the existing r123 run directory}"
OUT="${2:-work/collision_terminal/r123_rule_repair_$(date +%Y%m%d_%H%M%S)_$$}"
python -B -m tools.validate_r123_repair --require-linux --native-smoke \
  --output "$OUT/validation" --execute
python -B -m tools.r123_rule_experiment prepare \
  --development "$OLD/screen/development.json" --legacy-reference "$OLD/rule" \
  --config "$OLD/screen/config.json" --options "$OLD/screen/options.json" \
  --output "$OUT/plan" --execute
printf '\nPrepared only; no training or full evaluation started.\nBASE=%q\n' "$OUT"
