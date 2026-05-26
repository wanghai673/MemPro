#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

mkdir -p results/hotpotqa logs

PYTHONPATH="best_versions/hotpotqa:${PYTHONPATH:-}" \
HOTPOTQA_WORKING_PROMPTS_PATH="${HOTPOTQA_WORKING_PROMPTS_PATH:-best_versions/hotpotqa/mempro_memory/prompts/working_prompts.py}" \
python -u eval/hotpotqa_test.py \
  --data "${HOTPOTQA_DATA:-data/hotpotqa/eval_400.json}" \
  --outdir "${HOTPOTQA_OUTDIR:-results/hotpotqa}" \
  --max-tokens "${HOTPOTQA_MAX_TOKENS:-2048}" \
  --memory-workers "${MEMPRO_MEMORY_WORKERS:-32}" \
  --embedding-model-path "${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}" \
  "$@" 2>&1 | tee logs/hotpotqa_inference.log
