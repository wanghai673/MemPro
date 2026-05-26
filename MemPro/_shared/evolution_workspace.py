#!/usr/bin/env python3
"""Utilities for public MemPro benchmark evolution workspaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CORE_VERSION_FILES = (
    "memory_agent.py",
    "research_agent.py",
    "memory_prompts.py",
    "research_prompts.py",
)

OPTIONAL_VERSION_FILES = (
    "final_summarize_prompts.py",
    "working_prompts.py",
)


def find_mempro_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "eval").is_dir() and (candidate / "best_versions").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate MemPro root from {start}")


def workspace_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[1]


def benchmark_name(workspace_root: Path) -> str:
    return workspace_root.name


def mempro_root(workspace_root: Path) -> Path:
    return find_mempro_root(workspace_root)


def registry_path(workspace_root: Path) -> Path:
    return workspace_root / "registry" / "versions.json"


def versions_dir(workspace_root: Path) -> Path:
    return workspace_root / "versions"


def runs_dir(workspace_root: Path) -> Path:
    return workspace_root / "runs"


def load_registry(workspace_root: Path) -> Dict[str, Any]:
    path = registry_path(workspace_root)
    if not path.exists():
        return {"benchmark": benchmark_name(workspace_root), "versions": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(workspace_root: Path, registry: Dict[str, Any]) -> None:
    path = registry_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        f.write("\n")


def version_ids(registry: Dict[str, Any]) -> List[str]:
    return [str(item["version_id"]) for item in registry.get("versions", [])]


def get_version(registry: Dict[str, Any], version_id: str) -> Optional[Dict[str, Any]]:
    for item in registry.get("versions", []):
        if item.get("version_id") == version_id:
            return item
    return None


def next_version_id(registry: Dict[str, Any]) -> str:
    used = set(version_ids(registry))
    index = 1
    while True:
        candidate = f"v{index:04d}"
        if candidate not in used:
            return candidate
        index += 1


def copy_version_files(src: Path, dst: Path) -> List[str]:
    if not src.is_dir():
        raise FileNotFoundError(f"Version directory does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in (*CORE_VERSION_FILES, *OPTIONAL_VERSION_FILES):
        source_file = src / name
        if source_file.exists():
            shutil.copy2(source_file, dst / name)
            copied.append(name)
    missing = [name for name in CORE_VERSION_FILES if name not in copied]
    if missing:
        raise FileNotFoundError(f"Missing required version files in {src}: {', '.join(missing)}")
    return copied


def register_version(
    workspace_root: Path,
    base_version: str,
    new_version: Optional[str],
    note: str,
    dry_run: bool,
) -> Dict[str, Any]:
    registry = load_registry(workspace_root)
    if get_version(registry, base_version) is None:
        raise ValueError(f"Base version is not registered: {base_version}")

    new_version_id = new_version or next_version_id(registry)
    if get_version(registry, new_version_id) is not None:
        raise ValueError(f"Version already registered: {new_version_id}")

    base_dir = versions_dir(workspace_root) / base_version
    new_dir = versions_dir(workspace_root) / new_version_id
    if new_dir.exists():
        raise FileExistsError(f"Version directory already exists: {new_dir}")

    copied = list_existing_version_files(base_dir)
    entry = {
        "version_id": new_version_id,
        "parent_version": base_version,
        "role": "candidate",
        "status": "created",
        "version_dir": f"versions/{new_version_id}",
        "files": copied,
        "metrics": {"train_score": None, "test_score": None},
        "notes": note,
    }

    if not dry_run:
        copy_version_files(base_dir, new_dir)
        registry.setdefault("versions", []).append(entry)
        save_registry(workspace_root, registry)

    return entry


def list_existing_version_files(version_dir: Path) -> List[str]:
    return [
        name
        for name in (*CORE_VERSION_FILES, *OPTIONAL_VERSION_FILES)
        if (version_dir / name).exists()
    ]


def score_for(entry: Dict[str, Any]) -> float:
    metrics = entry.get("metrics") or {}
    for key in ("train_score", "validation_score", "test_score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    role = entry.get("role")
    if role == "evolved_best":
        return -1.0
    if role == "initial_framework":
        return -2.0
    return -3.0


def select_base(workspace_root: Path) -> Dict[str, Any]:
    registry = load_registry(workspace_root)
    versions = registry.get("versions", [])
    if not versions:
        raise ValueError("No versions are registered.")
    return max(versions, key=score_for)


def print_versions(workspace_root: Path) -> None:
    registry = load_registry(workspace_root)
    print(f"Benchmark: {registry.get('benchmark', benchmark_name(workspace_root))}")
    print(f"Registry: {registry_path(workspace_root)}")
    print("")
    for entry in registry.get("versions", []):
        metrics = entry.get("metrics") or {}
        print(
            f"- {entry.get('version_id')} "
            f"role={entry.get('role')} "
            f"parent={entry.get('parent_version')} "
            f"train={metrics.get('train_score')} "
            f"test={metrics.get('test_score')}"
        )
    print("")
    base = select_base(workspace_root)
    print(f"Recommended base: {base.get('version_id')} ({base.get('role')})")


def materialize_runtime(workspace_root: Path, version_id: str, force: bool = False) -> Path:
    bench = benchmark_name(workspace_root)
    root = mempro_root(workspace_root)
    source_runtime = root / "best_versions" / bench / "mempro_memory"
    if not source_runtime.is_dir():
        raise FileNotFoundError(f"Runtime package not found: {source_runtime}")

    version_dir = versions_dir(workspace_root) / version_id
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")

    runtime_parent = runs_dir(workspace_root) / version_id / "runtime_parent"
    runtime_pkg = runtime_parent / "mempro_memory"
    if runtime_parent.exists() and force:
        shutil.rmtree(runtime_parent)
    if runtime_pkg.exists():
        return runtime_parent

    runtime_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_runtime, runtime_pkg, dirs_exist_ok=True)

    overlays = {
        "memory_agent.py": runtime_pkg / "agents" / "memory_agent.py",
        "research_agent.py": runtime_pkg / "agents" / "research_agent.py",
        "memory_prompts.py": runtime_pkg / "prompts" / "memory_prompts.py",
        "research_prompts.py": runtime_pkg / "prompts" / "research_prompts.py",
        "final_summarize_prompts.py": runtime_pkg / "prompts" / "final_summarize_prompts.py",
        "working_prompts.py": runtime_pkg / "prompts" / "working_prompts.py",
    }
    for name, target in overlays.items():
        source = version_dir / name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return runtime_parent


def update_registry_result(
    workspace_root: Path,
    version_id: str,
    split: str,
    output_dir: Path,
) -> None:
    registry = load_registry(workspace_root)
    entry = get_version(registry, version_id)
    if entry is None:
        return
    entry.setdefault("runs", {})[split] = os.path.relpath(output_dir, workspace_root)
    save_registry(workspace_root, registry)


def eval_script_path(workspace_root: Path) -> Path:
    bench = benchmark_name(workspace_root)
    return mempro_root(workspace_root) / "eval" / f"{bench}_test.py"


def default_split_args(bench: str, split: str, limit: Optional[int]) -> List[str]:
    start = 0
    if split == "train":
        defaults = {
            "locomo": 154,
            "longmemeval": 50,
            "hotpotqa": 90,
            "narrativeqa": 40,
        }
        end = defaults.get(bench)
    else:
        end = None
    if limit is not None:
        end = start + limit
    args = ["--start-idx", str(start)]
    if end is not None:
        args.extend(["--end-idx", str(end)])
    if bench == "narrativeqa" and split in {"train", "validation", "test"}:
        args.extend(["--split", "train" if split == "train" else "test"])
    return args


def run_eval(
    workspace_root: Path,
    version_id: str,
    split: str,
    limit: Optional[int],
    extra_args: Iterable[str],
    force_runtime: bool,
) -> int:
    bench = benchmark_name(workspace_root)
    script = eval_script_path(workspace_root)
    if not script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script}")

    runtime_parent = materialize_runtime(workspace_root, version_id, force=force_runtime)
    outdir = runs_dir(workspace_root) / version_id / split
    outdir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(script),
        "--outdir",
        str(outdir),
        *default_split_args(bench, split, limit),
        *list(extra_args),
    ]

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(runtime_parent)
        if not existing_pythonpath
        else str(runtime_parent) + os.pathsep + existing_pythonpath
    )
    print("Running:")
    print(" ".join(command))
    print(f"PYTHONPATH={runtime_parent}")
    result = subprocess.run(command, cwd=str(mempro_root(workspace_root)), env=env)
    if result.returncode == 0:
        update_registry_result(workspace_root, version_id, split, outdir)
    return int(result.returncode)


def main_register(script_file: str) -> None:
    workspace_root = workspace_root_from_script(script_file)
    parser = argparse.ArgumentParser(description="Register a new MemPro benchmark evolution version.")
    parser.add_argument("--base-version", default=None)
    parser.add_argument("--new-version", default=None)
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_version = args.base_version or select_base(workspace_root)["version_id"]
    entry = register_version(
        workspace_root=workspace_root,
        base_version=base_version,
        new_version=args.new_version,
        note=args.note,
        dry_run=args.dry_run,
    )
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def main_select(script_file: str) -> None:
    print_versions(workspace_root_from_script(script_file))


def main_run_eval(script_file: str) -> None:
    workspace_root = workspace_root_from_script(script_file)
    parser = argparse.ArgumentParser(description="Run one MemPro benchmark version.")
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-runtime", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    version_id = args.version or select_base(workspace_root)["version_id"]
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    raise_code = run_eval(
        workspace_root=workspace_root,
        version_id=version_id,
        split=args.split,
        limit=args.limit,
        extra_args=extra_args,
        force_runtime=args.force_runtime,
    )
    raise SystemExit(raise_code)
