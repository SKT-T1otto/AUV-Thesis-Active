#!/usr/bin/env bash
# Explicit manual research only: A/B/C sequentially, fresh 100ep jobs.
set -euo pipefail
if [[ $# -ne 3 || "$3" != "--execute" ]]; then
  echo 'Usage: bash scripts/run_path_option_pilot.sh VALIDATION_JSON FRESH_ROOT --execute'
  exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
validation="$1"
experiment_root="$2"
python -c 'import sys; from chapter3_bser.path_option.source import admission; admission(sys.argv[1])' "$validation"
python -u -m tools.path_option_experiment prepare \
  --stage pilot --config configs/b2.json --options configs/path_option_v1.json \
  --output "$experiment_root" --execute
mkdir "$experiment_root/logs"
for arm in A B C; do
  python -u -m tools.path_option_experiment run \
    --root "$experiment_root" --validation "$validation" \
    --arm "$arm" --seed 20261111 --execute \
    2>&1 | tee "$experiment_root/logs/${arm}.log"
done
python -m tools.path_option_experiment report \
  --root "$experiment_root" --output "$experiment_root/pilot_report" --execute
