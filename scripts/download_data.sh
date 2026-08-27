#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/locomo

python download_data/download_file.py \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  data/locomo/locomo10.json
