#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

mkdir -p data/longmemeval/memory logs

PYTHONPATH="best_versions/longmemeval:${PYTHONPATH:-}" \
python -u eval/build_longmemeval_memory.py \
  --data "${LONGMEMEVAL_DATA:-data/longmemeval/longmemeval_s_cleaned.json}" \
  --memory-root "${LONGMEMEVAL_MEMORY_ROOT:-data/longmemeval/memory/_memory_cache}" \
  --num-workers "${MEMPRO_MEMORY_WORKERS:-1}" \
  "$@" 2>&1 | tee logs/longmemeval_build_memory.log
