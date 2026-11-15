#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
VALIDATION="work/collision_terminal/r123_validate_${RUN_TAG}/validation.json"
ROOT="work/collision_terminal/r123_screen_${RUN_TAG}"
REFERENCE="${ROOT}_rule/reference_gate.json"
python -B -m tools.validate_r123 --require-linux --native-smoke \
  --output "$(dirname "$VALIDATION")" --execute
python -B -m tools.r123_experiment prepare --stage screen --output "$ROOT" --execute
python -B -m tools.r123_experiment evaluate --root "$ROOT" --validation "$VALIDATION" \
  --mode rule --suite development --output "${ROOT}_rule" --execute
printf '\nROOT=%q\nVALIDATION=%q\nREFERENCE=%q\n' "$ROOT" "$VALIDATION" "$REFERENCE"
python -B -c 'import json,sys; r=json.load(open(sys.argv[1])); print("RULE GATE:",r["passed"],"Training has not started."); sys.exit(0 if r["passed"] else 2)' "$REFERENCE"
