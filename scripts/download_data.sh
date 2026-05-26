#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/locomo data/hotpotqa data/longmemeval data/narrativeqa

python download_data/download_file.py \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  data/locomo/locomo10.json

python download_data/download_file.py \
  https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_400.json \
  data/hotpotqa/eval_400.json

python download_data/download_file.py \
  https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_1600.json \
  data/hotpotqa/eval_1600.json

python download_data/download_file.py \
  https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_3200.json \
  data/hotpotqa/eval_3200.json

python download_data/download_longmemeval.py --with-readme
python download_data/download_narrativeqa.py
