#!/usr/bin/env bash
# Runs each config in its own process (bounded memory), then scores. Env selects the LLM (see .env.example).
set -euo pipefail
export PYTHONPATH=src
PY=${PY:-python}
$PY -u eval/run_generation_eval.py --n 25 --phase generate --configs norerank
$PY -u eval/run_generation_eval.py --n 25 --phase generate --configs rerank
$PY -u eval/run_generation_eval.py --n 25 --phase score
$PY eval/summarize.py
