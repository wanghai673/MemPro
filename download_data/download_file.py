#!/usr/bin/env python
"""Small downloader used by MemPro dataset scripts."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import requests


DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
CHUNK_SIZE = 1024 * 1024
HF_MIRROR = "https://hf-mirror.com"


def _dedupe_preserve_order(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def candidate_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        return [url]

    path = parsed.path.lstrip("/")
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    candidates: list[str] = []
    if hf_endpoint:
        candidates.append(f"{hf_endpoint}/{path}")
    candidates.append(f"{HF_MIRROR}/{path}")
    candidates.append(url)
    return _dedupe_preserve_order(candidates)


def _download_with_requests(url: str, output: Path, timeout: int) -> None:
    part_path = output.with_suffix(output.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)

    part_path.replace(output)


def _download_with_wget(url: str, output: Path, timeout: int, retries: int) -> None:
    part_path = output.with_suffix(output.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    cmd = [
        "wget",
        "--tries",
        str(retries),
        "--waitretry",
        "2",
        "--timeout",
        str(timeout),
        "-O",
        str(part_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    part_path.replace(output)


def download_to_path(
    urls: Sequence[str] | str,
    output: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"[SKIP] {output} already exists")
        return output

    url_list = [urls] if isinstance(urls, str) else list(urls)
    errors: list[str] = []
    has_wget = shutil.which("wget") is not None

    for url in _dedupe_preserve_order(url_list):
        print(f"[DOWNLOAD] {url}")
        for attempt in range(1, retries + 1):
            try:
                _download_with_requests(url, output, timeout)
                print(f"[OK] {output}")
                return output
            except Exception as exc:
                errors.append(f"requests attempt {attempt} failed for {url}: {exc}")
                if has_wget:
                    try:
                        _download_with_wget(url, output, timeout, retries)
                        print(f"[OK] {output}")
                        return output
                    except Exception as wget_exc:
                        errors.append(f"wget attempt {attempt} failed for {url}: {wget_exc}")

    raise RuntimeError("\n".join(errors))


def download_url(
    url: str,
    output: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Path:
    return download_to_path(candidate_urls(url), output, timeout=timeout, retries=retries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one file")
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args()

    download_url(args.url, Path(args.output), timeout=args.timeout, retries=args.retries)


if __name__ == "__main__":
    main()
