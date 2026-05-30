#!/usr/bin/env python
"""Download NarrativeQA parquet shards directly from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_file import download_url


SPLIT_SHARDS = {
    "train": 24,
    "validation": 3,
    "test": 8,
}

REPO_BASE = "https://huggingface.co/datasets/deepmind/narrativeqa/resolve/main/data"


def shard_name(split: str, index: int, total: int) -> str:
    return f"{split}-{index:05d}-of-{total:05d}.parquet"


def download_split(split: str, output_dir: Path) -> None:
    total = SPLIT_SHARDS[split]
    for index in range(total):
        filename = shard_name(split, index, total)
        url = f"{REPO_BASE}/{filename}"
        download_url(url, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NarrativeQA")
    parser.add_argument("--output-dir", type=Path, default=Path("data/narrativeqa"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_SHARDS),
        default=["train", "validation", "test"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        download_split(split, args.output_dir)


if __name__ == "__main__":
    main()
