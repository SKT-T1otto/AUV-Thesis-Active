#!/usr/bin/env bash
set -euo pipefail
stage=${1:?supply focus or regression}
run=${2:?supply the same fresh run root for both stages}
[[ "$stage" == focus || "$stage" == regression ]]
cd -- "$(dirname -- "$0")/.."
mkdir -p -- "$run"
run=$(cd -- "$run" && pwd -P)
if [[ -e "$run/$stage" ]]; then
    echo "Stage output already exists: $run/$stage" >&2
    exit 1
fi
if [[ "$stage" == regression ]]; then
    test -s "$run/focus/completed.json"
fi
printf 'Stage %s, output %s\n' "$stage" "$run"
log=$(mktemp "$run/${stage}_XXXXXX.log")
if [[ "$stage" == focus ]]; then
    python -B -m unittest checks.test_r123_final_leg 2>&1 | tee -a "$log"
fi
python -u -B -m tools.r123_final_leg "$stage" --output "$run" --execute "${@:3}" 2>&1 | tee -a "$log"
test -s "$run/$stage/completed.json"
if [[ "$stage" == regression ]]; then
    archive="$run.tar.gz"
    test ! -e "$archive"
    tar -czf "$archive" -C "$(dirname -- "$run")" "$(basename -- "$run")"
    printf 'Upload %s\n' "$archive"
fi
