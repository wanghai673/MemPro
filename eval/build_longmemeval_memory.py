#!/usr/bin/env python3
"""Build LongMemEval memory caches from the downloaded JSON file."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from mempro_memory import (
    InMemoryMemoryStore,
    InMemoryPageStore,
    MemoryAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def env_or_default(name: str, default: str | None = None) -> str | None:
    import os

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def role_env(role: str, suffix: str, default: str | None = None) -> str | None:
    return env_or_default(f"{role}_{suffix}", env_or_default(f"OPENAI_{suffix}", default))


def build_generator(args: argparse.Namespace):
    if args.memory_api_type == "openai":
        return OpenAIGenerator(
            OpenAIGeneratorConfig(
                model_name=args.memory_model,
                api_key=args.memory_api_key,
                base_url=args.memory_base_url,
                temperature=args.memory_temperature,
                max_tokens=args.memory_max_tokens,
                use_schema=False,
            ).__dict__
        )
    if args.memory_api_type == "vllm":
        return VLLMGenerator(
            VLLMGeneratorConfig(
                model_name=args.memory_model,
                api_key=args.memory_api_key,
                base_url=args.memory_base_url,
                temperature=args.memory_temperature,
                max_tokens=args.memory_max_tokens,
                use_schema=False,
            ).__dict__
        )
    raise ValueError(f"Unsupported memory API type: {args.memory_api_type}")


def session_to_text(session: Any, session_id: str, date: str, index: int) -> str:
    lines = [f"=== SESSION {index} - Dialogue Time: {date} - Session ID: {session_id} ==="]
    if isinstance(session, list):
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("speaker") or "unknown"
                content = turn.get("content") or turn.get("text") or ""
                lines.append(f"{role}: {content}")
            else:
                lines.append(str(turn))
    else:
        lines.append(str(session))
    return "\n".join(lines).strip()


def memory_key(sample: Dict[str, Any], sample_index: int) -> str:
    session_ids = sample.get("haystack_session_ids") or []
    if session_ids:
        raw = "\n".join(str(x) for x in session_ids)
    else:
        raw = f"{sample_index}:{sample.get('question_id', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_one(sample_index: int, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    key = memory_key(sample, sample_index)
    cache_dir = Path(args.memory_root) / key
    done_file = cache_dir / "_DONE"

    if done_file.exists() and not args.force:
        return {"sample_index": sample_index, "memory_key": key, "status": "skipped"}

    cache_dir.mkdir(parents=True, exist_ok=True)
    memory_store = InMemoryMemoryStore(dir_path=str(cache_dir))
    page_store = InMemoryPageStore(dir_path=str(cache_dir))
    generator = build_generator(args)
    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=generator,
    )

    sessions = sample.get("haystack_sessions") or []
    dates = sample.get("haystack_dates") or []
    session_ids = sample.get("haystack_session_ids") or []

    for i, session in enumerate(sessions):
        session_id = str(session_ids[i]) if i < len(session_ids) else f"session_{i}"
        date = str(dates[i]) if i < len(dates) else ""
        text = session_to_text(session, session_id, date, i)
        update = memory_agent.memorize(text)
        pages = page_store.load()
        if pages:
            pages[-1].meta.update(
                {
                    "sample_index": sample_index,
                    "question_id": sample.get("question_id"),
                    "session_index": i,
                    "session_id": session_id,
                    "date": date,
                }
            )
            page_store.save(pages)

    state = memory_store.load()
    meta = {
        "memory_key": key,
        "sample_index": sample_index,
        "question_id": sample.get("question_id"),
        "session_ids": [str(x) for x in session_ids],
        "num_sessions": len(sessions),
        "num_abstracts": len(state.abstracts),
    }
    dump_json(cache_dir / "memory_meta.json", meta)
    done_file.write_text("ok\n", encoding="utf-8")
    return {"sample_index": sample_index, "memory_key": key, "status": "built"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MemPro LongMemEval memory cache")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--memory-root", type=Path, default=Path("data/longmemeval/memory/_memory_cache"))
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-api-key", default=role_env("MEMORY", "API_KEY", "empty"))
    parser.add_argument("--memory-base-url", default=role_env("MEMORY", "BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--memory-model", default=role_env("MEMORY", "MODEL", "gpt-4o-mini"))
    parser.add_argument("--memory-api-type", choices=["openai", "vllm"], default=role_env("MEMORY", "API_TYPE", "openai"))
    parser.add_argument("--memory-temperature", type=float, default=0.3)
    parser.add_argument("--memory-max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples: List[Dict[str, Any]] = load_json(args.data)
    end_idx = args.end_idx if args.end_idx is not None else len(samples)
    indices = list(range(args.start_idx, min(end_idx, len(samples))))
    args.memory_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        for idx in tqdm(indices, desc="Build LongMemEval memory"):
            results.append(build_one(idx, samples[idx], args))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(build_one, idx, samples[idx], args): idx for idx in indices}
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Build LongMemEval memory"):
                results.append(future.result())

    dump_json(Path(args.memory_root).parent / "build_manifest.json", results)


if __name__ == "__main__":
    main()
