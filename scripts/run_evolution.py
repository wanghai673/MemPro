#!/usr/bin/env python3
"""Launch Codex for MemPro benchmark evolution workspaces."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


BENCHMARKS = ("locomo", "longmemeval", "hotpotqa", "narrativeqa")


def build_prompt(benchmark: str) -> str:
    return (
        "Read AGENTS.md, registry/versions.json, and the local scripts first. "
        f"Continue MemPro framework evolution for {benchmark}: select a base "
        "version, create one coherent candidate direction, use targeted debug "
        "before any full benchmark, and avoid sample-specific shortcuts."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Codex for MemPro evolution.")
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--model", default="gpt-5.4-medium")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--execute", action="store_true", help="Run Codex instead of only printing the command.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    workspace = repo_root / "MemPro" / args.benchmark
    if not workspace.is_dir():
        raise SystemExit(f"Workspace not found: {workspace}")
    if not (workspace / "AGENTS.md").is_file():
        raise SystemExit(f"AGENTS.md not found in workspace: {workspace}")

    prompt = args.prompt or build_prompt(args.benchmark)
    command = [
        "codex",
        "--model",
        args.model,
        "-C",
        str(workspace),
        prompt,
    ]

    print(" ".join(command))
    if args.dry_run or not args.execute:
        return 0

    if shutil.which("codex") is None:
        raise SystemExit("Codex CLI not found on PATH.")
    return subprocess.run(command, cwd=str(repo_root), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
