#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

# Editable parameters
DATA_DIR="${NARRATIVEQA_DATA_DIR:-data/narrativeqa}"
SPLIT="${NARRATIVEQA_SPLIT:-test}"
OUTDIR="${NARRATIVEQA_OUTDIR:-results/narrativeqa}"
START_IDX="${NARRATIVEQA_START_IDX:-0}"
END_IDX="${NARRATIVEQA_END_IDX:-300}"
MAX_TOKENS="${NARRATIVEQA_MAX_TOKENS:-2048}"
SEED="${NARRATIVEQA_SEED:-42}"
NUM_WORKERS="${MEMPRO_NUM_WORKERS:-16}"
EMBEDDING_MODEL_PATH="${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}"
DENSE_MODEL="${MEMPRO_DENSE_MODEL:-$EMBEDDING_MODEL_PATH}"
DENSE_DEVICES="${MEMPRO_DENSE_DEVICES:-cuda:0}"
MEMORY_API_KEY="${MEMORY_API_KEY:-${OPENAI_API_KEY:-empty}}"
MEMORY_BASE_URL="${MEMORY_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
MEMORY_MODEL="${MEMORY_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
MEMORY_API_TYPE="${MEMORY_API_TYPE:-${OPENAI_API_TYPE:-openai}}"
RESEARCH_API_KEY="${RESEARCH_API_KEY:-${OPENAI_API_KEY:-empty}}"
RESEARCH_BASE_URL="${RESEARCH_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
RESEARCH_MODEL="${RESEARCH_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
RESEARCH_API_TYPE="${RESEARCH_API_TYPE:-${OPENAI_API_TYPE:-openai}}"
WORKING_API_KEY="${WORKING_API_KEY:-${OPENAI_API_KEY:-empty}}"
WORKING_BASE_URL="${WORKING_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
WORKING_MODEL="${WORKING_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
WORKING_API_TYPE="${WORKING_API_TYPE:-${OPENAI_API_TYPE:-openai}}"
PYTHONPATH_PREFIX="${NARRATIVEQA_PYTHONPATH_PREFIX:-best_versions/narrativeqa}"
LOG_FILE="${NARRATIVEQA_LOG_FILE:-logs/narrativeqa_inference.log}"

# Example:
# EXTRA_ARGS+=(--question-workers 8)
EXTRA_ARGS=()

mkdir -p "$OUTDIR" logs

PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
python -u eval/narrativeqa_test.py \
  --data-dir "$DATA_DIR" \
  --split "$SPLIT" \
  --outdir "$OUTDIR" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --embedding-model-path "$EMBEDDING_MODEL_PATH" \
  --dense-model "$DENSE_MODEL" \
  --dense-devices "$DENSE_DEVICES" \
  --memory-api-key "$MEMORY_API_KEY" \
  --memory-base-url "$MEMORY_BASE_URL" \
  --memory-model "$MEMORY_MODEL" \
  --memory-api-type "$MEMORY_API_TYPE" \
  --research-api-key "$RESEARCH_API_KEY" \
  --research-base-url "$RESEARCH_BASE_URL" \
  --research-model "$RESEARCH_MODEL" \
  --research-api-type "$RESEARCH_API_TYPE" \
  --working-api-key "$WORKING_API_KEY" \
  --working-base-url "$WORKING_BASE_URL" \
  --working-model "$WORKING_MODEL" \
  --working-api-type "$WORKING_API_TYPE" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
