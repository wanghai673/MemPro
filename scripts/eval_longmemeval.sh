#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

# Editable parameters
DATA="${LONGMEMEVAL_DATA:-data/longmemeval/longmemeval_s_cleaned.json}"
MEMORY_ROOT="${LONGMEMEVAL_MEMORY_ROOT:-data/longmemeval/memory/_memory_cache}"
OUTDIR="${LONGMEMEVAL_OUTDIR:-results/longmemeval}"
START_IDX="${LONGMEMEVAL_START_IDX:-0}"
END_IDX="${LONGMEMEVAL_END_IDX:-}"
EMBEDDING_MODEL="${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}"
DENSE_DEVICES="${MEMPRO_DENSE_DEVICES:-cuda:0}"
NUM_WORKERS="${MEMPRO_NUM_WORKERS:-1}"
MEMORY_BUILD_WORKERS="${MEMORY_BUILD_WORKERS:-${MEMPRO_MEMORY_WORKERS:-1}}"
USE_BM25="${LONGMEMEVAL_USE_BM25:-1}"
USE_DENSE="${LONGMEMEVAL_USE_DENSE:-1}"
BUILD_MEMORY_IF_MISSING="${LONGMEMEVAL_BUILD_MEMORY_IF_MISSING:-1}"
FORCE_REBUILD_MEMORY="${LONGMEMEVAL_FORCE_REBUILD_MEMORY:-0}"
MEMORY_API_KEY="${MEMORY_API_KEY:-${OPENAI_API_KEY:-empty}}"
MEMORY_BASE_URL="${MEMORY_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
MEMORY_MODEL="${MEMORY_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
MEMORY_API_TYPE="${MEMORY_API_TYPE:-${OPENAI_API_TYPE:-openai}}"
MEMORY_TEMPERATURE="${MEMORY_TEMPERATURE:-0.3}"
MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-1024}"
RESEARCH_API_KEY="${RESEARCH_API_KEY:-${OPENAI_API_KEY:-empty}}"
RESEARCH_BASE_URL="${RESEARCH_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
RESEARCH_MODEL="${RESEARCH_MODEL:-${OPENAI_MODEL:-gpt-4o-mini}}"
RESEARCH_API_TYPE="${RESEARCH_API_TYPE:-${OPENAI_API_TYPE:-openai}}"
RESEARCH_TEMPERATURE="${RESEARCH_TEMPERATURE:-0.3}"
RESEARCH_MAX_TOKENS="${RESEARCH_MAX_TOKENS:-2048}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_TYPE="${JUDGE_API_TYPE:-}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-512}"
PYTHONPATH_PREFIX="${LONGMEMEVAL_PYTHONPATH_PREFIX:-best_versions/longmemeval}"
LOG_FILE="${LONGMEMEVAL_LOG_FILE:-logs/longmemeval_inference.log}"

# Example:
# EXTRA_ARGS+=(--start-idx 0 --end-idx 49)
EXTRA_ARGS=()

mkdir -p "$OUTDIR" logs

bm25_args=()
if [ "$USE_BM25" = "1" ]; then
  bm25_args+=(--use-bm25)
fi

dense_args=()
if [ "$USE_DENSE" = "1" ]; then
  dense_args+=(--use-dense)
fi

memory_build_args=()
if [ "$BUILD_MEMORY_IF_MISSING" = "1" ]; then
  memory_build_args+=(--build-memory-if-missing)
else
  memory_build_args+=(--no-build-memory-if-missing)
fi

if [ "$FORCE_REBUILD_MEMORY" = "1" ]; then
  memory_build_args+=(--force-rebuild-memory)
fi

range_args=(--start-idx "$START_IDX")
if [ -n "$END_IDX" ]; then
  range_args+=(--end-idx "$END_IDX")
fi

judge_args=()
if [ -n "$JUDGE_API_KEY" ]; then
  judge_args+=(--judge-api-key "$JUDGE_API_KEY")
fi
if [ -n "$JUDGE_BASE_URL" ]; then
  judge_args+=(--judge-base-url "$JUDGE_BASE_URL")
fi
if [ -n "$JUDGE_MODEL" ]; then
  judge_args+=(--judge-model "$JUDGE_MODEL")
fi
if [ -n "$JUDGE_API_TYPE" ]; then
  judge_args+=(--judge-api-type "$JUDGE_API_TYPE")
fi

PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
python -u eval/longmemeval_test.py \
  --data "$DATA" \
  --memory-root "$MEMORY_ROOT" \
  --outdir "$OUTDIR" \
  "${range_args[@]}" \
  "${bm25_args[@]}" \
  "${dense_args[@]}" \
  --dense-model "$EMBEDDING_MODEL" \
  --dense-devices "$DENSE_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --memory-build-workers "$MEMORY_BUILD_WORKERS" \
  "${memory_build_args[@]}" \
  --memory-api-key "$MEMORY_API_KEY" \
  --memory-base-url "$MEMORY_BASE_URL" \
  --memory-model "$MEMORY_MODEL" \
  --memory-api-type "$MEMORY_API_TYPE" \
  --memory-temperature "$MEMORY_TEMPERATURE" \
  --memory-max-tokens "$MEMORY_MAX_TOKENS" \
  --research-api-key "$RESEARCH_API_KEY" \
  --research-base-url "$RESEARCH_BASE_URL" \
  --research-model "$RESEARCH_MODEL" \
  --research-api-type "$RESEARCH_API_TYPE" \
  --research-temperature "$RESEARCH_TEMPERATURE" \
  --research-max-tokens "$RESEARCH_MAX_TOKENS" \
  --judge-temperature "$JUDGE_TEMPERATURE" \
  --judge-max-tokens "$JUDGE_MAX_TOKENS" \
  "${judge_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
