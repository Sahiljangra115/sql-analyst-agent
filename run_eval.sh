#!/usr/bin/env bash
# Usage: ./run_eval.sh   (needs Ollama with gemma4:e2b)
# One GPU, so presets run one after another. Each preset takes the GPU lock on its own (other
# jobs can interleave between presets) and unloads the model before releasing it.
# Env: LIMIT (default 200, 0 = all 500), SEED (default 0), OUT (default results), LOGS (default logs), CONFIGS, GPU_LOCK.
set -euo pipefail
cd "$(dirname "$0")"
source "$ML_VENV/bin/activate"
export SQLAGENT_BACKEND=ollama SQLAGENT_MODEL=${SQLAGENT_MODEL:-gemma4:e2b} PYTHONPATH=src PYTHONUNBUFFERED=1
OUT=${OUT:-results}
LOGS=${LOGS:-logs}
LOCK=${GPU_LOCK:-${LOCKS:-/tmp}/gpu.lock}
LIM=(); [ "${LIMIT:-200}" != 0 ] && LIM=(--limit "${LIMIT:-200}")
mkdir -p "$OUT" "$LOGS"
for c in ${CONFIGS:-baseline retrieval correction full}; do
  flock "$LOCK" bash -c 'python -m sqlagent eval --db-root data/minidev/dbs "${@:5}" --seed "$2" --configs "$1" --out "$3" > "$4/$1.log" 2>&1; rc=$?; ollama stop "$SQLAGENT_MODEL"; exit $rc' \
    _ "$c" "${SEED:-0}" "$OUT" "$LOGS" "${LIM[@]}"
done
