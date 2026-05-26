#!/usr/bin/env bash
set -euo pipefail

set -a
[ -f .env ] && source .env
set +a

mkdir -p results/locomo logs

PYTHONPATH="best_versions/locomo:${PYTHONPATH:-}" \
python -u eval/locomo_test.py \
  --data "${LOCOMO_DATA:-data/locomo/locomo10.json}" \
  --outdir "${LOCOMO_OUTDIR:-results/locomo}" \
  --question-workers "${MEMPRO_QUESTION_WORKERS:-32}" \
  "$@" 2>&1 | tee logs/locomo_inference.log
