#!/usr/bin/env bash
# Explicit manual stages; missing --execute only prints the Python plan.
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  verify) exec "$PYTHON_BIN" -m tools.validate_checkout --require-linux "$@" ;;
  prepare) exec "$PYTHON_BIN" -m cli map-suite prepare "$@" ;;
  train) exec "$PYTHON_BIN" -m cli map-suite run --stage train "$@" ;;
  diagnose) exec "$PYTHON_BIN" -m cli map-suite run --stage diagnose "$@" ;;
  evaluate) exec "$PYTHON_BIN" -m cli map-suite run --stage evaluate "$@" ;;
  run) exec "$PYTHON_BIN" -m cli map-suite run "$@" ;;
  summarize) exec "$PYTHON_BIN" -m cli map-suite summarize "$@" ;;
  *) printf '%s\n' 'Usage: bash scripts/linux_map_experiment.sh {verify|prepare|train|diagnose|evaluate|run|summarize} [arguments] [--execute]'
     [[ "$STAGE" == help || "$STAGE" == --help ]] ;;
esac
