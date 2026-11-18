#!/usr/bin/env bash
# Offline recheck only: 24 existing episodes, zero environment/training steps.
set -euo pipefail
RUN="${1:?Usage: bash scripts/r123_search_audit_linux.sh TRACKING_RUN GOVERNED_FOLDER NEW_OUTPUT}"
OLD="${2:?governed folder required}"
OUT="${3:?new output required}"
[[ -n "$RUN" && -n "$OLD" && -n "$OUT" ]]
[[ -f "$RUN/plan/plan.json" && -f "$RUN/both/completed.json" ]]
[[ -f "$OLD/completed.json" && -f "$OLD/episodes.jsonl" ]]
[[ ! -e "$OUT" && ! -e "${OUT}.log" ]]
mkdir -p "$(dirname "$OUT")"
echo "Offline recheck: 24 existing episodes; physical steps=0; training=0; checkpoint loads=0"
python -u -B -m tools.r123_search_audit --run "$RUN" --baseline "$OLD" --output "$OUT" --execute 2>&1 | tee "${OUT}.log"
[[ -f "$OUT/completed.json" ]]
echo "COMPLETED: $OUT/completed.json"
