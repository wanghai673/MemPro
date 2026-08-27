#!/usr/bin/env bash
set -euo pipefail

# Editable parameters
DATA="${LOCOMO_DATA:-data/locomo/locomo10.json}"
OUTDIR="${LOCOMO_OUTDIR:-results/locomo}"
PYTHONPATH_PREFIX="${LOCOMO_PYTHONPATH_PREFIX:-best_versions/locomo}"
LOG_FILE="${LOCOMO_LOG_FILE:-logs/locomo_inference.log}"

mkdir -p "$OUTDIR" logs

PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
python -u eval/locomo_test.py \
  --data "$DATA" \
  --outdir "$OUTDIR" \
  "$@" 2>&1 | tee "$LOG_FILE"
