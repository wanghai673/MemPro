#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

mkdir -p results/longmemeval logs

if [ ! -d "${LONGMEMEVAL_MEMORY_ROOT:-data/longmemeval/memory/_memory_cache}" ]; then
  echo "[INFO] LongMemEval memory cache not found; building it first."
  bash scripts/build_longmemeval_memory.sh
fi

PYTHONPATH="best_versions/longmemeval:${PYTHONPATH:-}" \
python -u eval/longmemeval_test.py \
  --data "${LONGMEMEVAL_DATA:-data/longmemeval/longmemeval_s_cleaned.json}" \
  --memory-root "${LONGMEMEVAL_MEMORY_ROOT:-data/longmemeval/memory/_memory_cache}" \
  --outdir "${LONGMEMEVAL_OUTDIR:-results/longmemeval}" \
  --use-bm25 \
  --use-dense \
  --dense-model "${MEMPRO_EMBEDDING_MODEL:-BAAI/bge-m3}" \
  --dense-devices "${MEMPRO_DENSE_DEVICES:-cuda:0}" \
  --num-workers "${MEMPRO_NUM_WORKERS:-1}" \
  "$@" 2>&1 | tee logs/longmemeval_inference.log
