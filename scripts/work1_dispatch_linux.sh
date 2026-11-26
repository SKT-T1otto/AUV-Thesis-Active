#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python}"
if (( $# > 1 )); then
    echo "Usage: bash scripts/work1_dispatch_linux.sh [new-output-directory]" >&2
    exit 2
fi
if (( $# == 1 )); then
    out=$1
else
    mkdir -p work
    out="$(mktemp -d work/dispatch_linux_XXXXXXXX)/run"
fi
if [[ -e "$out" || -e "$out.tar.gz" ]]; then
    echo "Output already exists: $out" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python_bin" -B -m unittest checks.test_work1_linux_dispatch
"$python_bin" -B -m tools.work1_linux_dispatch "$out" --execute
tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
printf 'REPORT=%s/REPORT.md\nRETURN=%s.tar.gz\n' "$out" "$out"
