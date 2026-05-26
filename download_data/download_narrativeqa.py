#!/usr/bin/env python
"""Download NarrativeQA and save splits as local parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NarrativeQA")
    parser.add_argument("--output-dir", type=Path, default=Path("data/narrativeqa"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("deepmind/narrativeqa")
    for split_name, split in dataset.items():
        out_path = args.output_dir / f"{split_name}.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] {out_path} already exists")
            continue
        print(f"[WRITE] {out_path}")
        split.to_parquet(str(out_path))


if __name__ == "__main__":
    main()
