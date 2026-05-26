#!/usr/bin/env python
"""Small downloader used by MemPro dataset scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one file")
    parser.add_argument("url")
    parser.add_argument("output")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"[SKIP] {output} already exists")
        return

    part_path = output.with_suffix(output.suffix + ".part")
    print(f"[DOWNLOAD] {args.url}")
    with requests.get(args.url, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part_path.replace(output)
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
