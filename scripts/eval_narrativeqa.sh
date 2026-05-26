#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

mkdir -p results/narrativeqa logs

PYTHONPATH="best_versions/narrativeqa:${PYTHONPATH:-}" \
python -u eval/narrativeqa_test.py \
  --data-dir "${NARRATIVEQA_DATA_DIR:-data/narrativeqa}" \
  --split "${NARRATIVEQA_SPLIT:-test}" \
  --outdir "${NARRATIVEQA_OUTDIR:-results/narrativeqa}" \
  --start-idx "${NARRATIVEQA_START_IDX:-0}" \
  --end-idx "${NARRATIVEQA_END_IDX:-300}" \
  --max-tokens "${NARRATIVEQA_MAX_TOKENS:-2048}" \
  --seed "${NARRATIVEQA_SEED:-42}" \
  --num-workers "${MEMPRO_NUM_WORKERS:-1}" \
  --embedding-model-path "${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}" \
  --memory-api-key "${MEMORY_API_KEY:-${OPENAI_API_KEY:-empty}}" \
  --memory-base-url "${MEMORY_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}" \
  --memory-model "${MEMORY_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}" \
  --memory-api-type "${MEMORY_API_TYPE:-${OPENAI_API_TYPE:-openai}}" \
  --research-api-key "${RESEARCH_API_KEY:-${OPENAI_API_KEY:-empty}}" \
  --research-base-url "${RESEARCH_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}" \
  --research-model "${RESEARCH_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}" \
  --research-api-type "${RESEARCH_API_TYPE:-${OPENAI_API_TYPE:-openai}}" \
  --working-api-key "${WORKING_API_KEY:-${OPENAI_API_KEY:-empty}}" \
  --working-base-url "${WORKING_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}" \
  --working-model "${WORKING_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}" \
  --working-api-type "${WORKING_API_TYPE:-${OPENAI_API_TYPE:-openai}}" \
  "$@" 2>&1 | tee logs/narrativeqa_inference.log
