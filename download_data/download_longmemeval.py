#!/usr/bin/env python
"""Download the LongMemEval cleaned split used by MemPro."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


REPO_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
FILES = {
    "small": "longmemeval_s_cleaned.json",
    "medium": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
    "readme": "README.md",
}


def download_file(filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    part_path = out_path.with_suffix(out_path.suffix + ".part")

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] {out_path} already exists")
        return out_path

    if filename == "README.md":
        url = f"{REPO_BASE}/raw/main/{filename}"
    else:
        url = f"{REPO_BASE}/resolve/main/{filename}"

    print(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    part_path.replace(out_path)
    print(f"[OK] {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LongMemEval data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/longmemeval"))
    parser.add_argument("--split", choices=["small", "medium", "oracle"], default="small")
    parser.add_argument("--with-readme", action="store_true")
    args = parser.parse_args()

    if args.with_readme:
        download_file(FILES["readme"], args.output_dir)
    download_file(FILES[args.split], args.output_dir)


if __name__ == "__main__":
    main()
