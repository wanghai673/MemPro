#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

# Editable parameters
DATA="${LOCOMO_DATA:-data/locomo/locomo10.json}"
OUTDIR="${LOCOMO_OUTDIR:-results/locomo}"
QUESTION_WORKERS="${MEMPRO_QUESTION_WORKERS:-32}"
PYTHONPATH_PREFIX="${LOCOMO_PYTHONPATH_PREFIX:-best_versions/locomo}"
LOG_FILE="${LOCOMO_LOG_FILE:-logs/locomo_inference.log}"

# Example:
# EXTRA_ARGS+=(--start-idx 0 --end-idx 0)
EXTRA_ARGS=()

mkdir -p "$OUTDIR" logs

PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
python -u eval/locomo_test.py \
  --data "$DATA" \
  --outdir "$OUTDIR" \
  --question-workers "$QUESTION_WORKERS" \
  "${EXTRA_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_FILE"
