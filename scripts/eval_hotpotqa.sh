#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

# Editable parameters
DATA="${HOTPOTQA_DATA:-data/hotpotqa/eval_400.json}"
OUTDIR="${HOTPOTQA_OUTDIR:-results/hotpotqa}"
MAX_TOKENS="${HOTPOTQA_MAX_TOKENS:-2048}"
NUM_WORKERS="${HOTPOTQA_NUM_WORKERS:-16}"
MEMORY_WORKERS="${HOTPOTQA_MEMORY_WORKERS:-1}"
EMBEDDING_MODEL_PATH="${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}"
PYTHONPATH_PREFIX="${HOTPOTQA_PYTHONPATH_PREFIX:-best_versions/hotpotqa}"
WORKING_PROMPTS_PATH="${HOTPOTQA_WORKING_PROMPTS_PATH:-best_versions/hotpotqa/mempro_memory/prompts/working_prompts.py}"
LOG_FILE="${HOTPOTQA_LOG_FILE:-logs/hotpotqa_inference.log}"

# Example:
# EXTRA_ARGS+=(--start-idx 0 --end-idx 99)
EXTRA_ARGS=()

mkdir -p "$OUTDIR" logs

PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
HOTPOTQA_WORKING_PROMPTS_PATH="$WORKING_PROMPTS_PATH" \
python -u eval/hotpotqa_test.py \
  --data "$DATA" \
  --outdir "$OUTDIR" \
  --max-tokens "$MAX_TOKENS" \
  --num-workers "$NUM_WORKERS" \
  --memory-workers "$MEMORY_WORKERS" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
