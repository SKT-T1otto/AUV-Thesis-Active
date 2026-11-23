#!/usr/bin/env bash
# Stage 1 acceptance uses only Python's standard library.
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
runs_dir="$(pwd -P)/verification/handoff_value_bounds/linux_runs"
mkdir -p -- "$runs_dir"
run_dir=$(mktemp -d "$runs_dir/run_XXXXXXXX")
log="$run_dir/run.log"
commit=$(git rev-parse --verify HEAD)

printf 'Log: %s\nCommit: %s\n' "$log" "$commit" | tee "$log"
python -VV 2>&1 | tee -a "$log"

printf '[1/2] Production module and property tests\n' | tee -a "$log"
python -E -B -u -X utf8 -m unittest -v checks.test_handoff_bounds 2>&1 | tee -a "$log"

printf '[2/2] Reference vertex checks and four-bin examples\n' | tee -a "$log"
python -E -B -u -X utf8 verification/handoff_value_design/20261123_v1/handoff_bounds.py 2>&1 | tee -a "$log"

printf 'PASS physical_steps=0 training_updates=0\n' | tee -a "$log"
