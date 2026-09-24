#!/usr/bin/env bash
# Runs the demo questions through the real agent (Ollama gemma4:e2b) on both demo databases.
# Builds the databases first if missing. Output: demo/*.out
set -euo pipefail
cd "$(dirname "$0")/.."
source "$ML_VENV/bin/activate"
export PYTHONPATH=src SQLAGENT_BACKEND=${SQLAGENT_BACKEND:-ollama}
[ -f data/smogcast.sqlite ] || python -m sqlagent build-smogcast
[ -f data/homecredit.duckdb ] || python -m sqlagent build-homecredit
python -m sqlagent chat data/smogcast.sqlite < demo/smogcast_questions.txt | tee demo/smogcast.out
python -m sqlagent chat data/homecredit.duckdb < demo/homecredit_questions.txt | tee demo/homecredit.out
