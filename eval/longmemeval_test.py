#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongMemEval test harness for MemPro.

This script can automatically build missing memory caches under:
  <memory_root>/<sample_cache_key>/

For each sample, it:
  1) ensures the required memory/page cache exists
  2) loads the cache
  3) runs ResearchAgent only
  4) records planner/search traces including retrieved page_ids
  5) judges whether research_summary contains the gold answer
  6) saves per-sample traces and a global results file
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import traceback
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from mempro_memory.agents.memory_agent import MemoryAgent
from mempro_memory.agents.research_agent import ResearchAgent
from mempro_memory.generator.openai_generator import OpenAIGenerator
from mempro_memory.generator.vllm_generator import VLLMGenerator
from mempro_memory.config.generator import OpenAIGeneratorConfig, VLLMGeneratorConfig
from mempro_memory.config.retriever import BM25RetrieverConfig, DenseRetrieverConfig
from mempro_memory.schemas.memory import InMemoryMemoryStore
from mempro_memory.schemas.page import InMemoryPageStore


def env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def role_env(role: str, suffix: str, default: Optional[str] = None) -> Optional[str]:
    return env_value(f"{role}_{suffix}", env_value(f"OPENAI_{suffix}", default))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_generator(
    api_type: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
    use_schema: bool = True,
    default_headers: Optional[Dict[str, str]] = None,
):
    if api_type == "openai":
        cfg = OpenAIGeneratorConfig(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            use_schema=use_schema,
        )
        return OpenAIGenerator.from_config(cfg)
    if api_type == "vllm":
        cfg = VLLMGeneratorConfig(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url or "http://localhost:8000/v1",
            default_headers=default_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            use_schema=use_schema,
        )
        return VLLMGenerator.from_config(cfg)
    raise ValueError(f"Unsupported api_type: {api_type}")


def build_retrievers(args: argparse.Namespace, index_dir: str, page_store: InMemoryPageStore) -> Dict[str, Any]:
    retrievers: Dict[str, Any] = {}

    if args.use_bm25:
        try:
            from mempro_memory.retriever.bm25 import BM25Retriever
            bm25_index_dir = os.path.join(index_dir, "bm25_index")
            if os.path.exists(bm25_index_dir):
                shutil.rmtree(bm25_index_dir)
            bm25_config = BM25RetrieverConfig(
                index_dir=bm25_index_dir,
                threads=args.bm25_threads,
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print("[OK] BM25 retriever created")
        except Exception as e:
            print(f"[WARN] BM25 retriever creation failed: {e}")

    if args.use_dense:
        try:
            from mempro_memory.retriever.dense_retriever import DenseRetriever
            dense_index_dir = os.path.join(index_dir, "dense_index")
            if os.path.exists(dense_index_dir):
                shutil.rmtree(dense_index_dir)
            dense_retriever_devices = parse_dense_devices(args.dense_devices)
            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                model_name=args.dense_model,
                api_url=args.dense_api_url,
                devices=dense_retriever_devices,
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print("[OK] Dense retriever created")
        except Exception as e:
            print(f"[WARN] Dense retriever creation failed: {e}")
            traceback.print_exc()

    return retrievers


def parse_dense_devices(dense_devices: Optional[str]) -> List[str]:
    if not dense_devices:
        return []
    return [item.strip() for item in dense_devices.split(",") if item.strip()]


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l : r + 1])
    except Exception:
        return None


def make_judge_prompt(question: str, gold_answer: str, summary: str) -> str:
    return f"""\
You are judging whether a research summary contains the correct answer to a question.

Question:
{question}

Gold answer:
{gold_answer}

Research summary:
{summary}

Return a JSON object with exactly these keys:
- "contains_answer": boolean
- "reason": short string

Rules:
- Mark true if the summary contains the gold answer, a close paraphrase, or the main answer entity/phrase.
- For entity answers, be lenient: if the gold entity is present anywhere in the summary, count it as correct even if extra context is missing.
- For short answers, do not require exact formatting if the meaning is clearly the same.
- Prefer semantic equivalence over exact wording.
- If the summary clearly supports the intended answer, even with extra context or minor wording differences, mark true.
- Mark false if the summary does not contain enough information to recover the gold answer.
- Do not output anything except JSON.
"""


def judge_summary(generator, question: str, gold_answer: str, summary: str) -> Dict[str, Any]:
    prompt = make_judge_prompt(question, gold_answer, summary)
    gold_norm = normalize_text(gold_answer)
    summary_norm = normalize_text(summary)
    try:
        resp = generator.generate_single(prompt=prompt)
        payload = resp.get("json") or safe_json_extract(resp.get("text", ""))
        if isinstance(payload, dict) and "contains_answer" in payload:
            contains_answer = bool(payload.get("contains_answer", False))
            if not contains_answer and gold_norm and gold_norm in summary_norm:
                contains_answer = True
            return {
                "contains_answer": contains_answer,
                "reason": str(payload.get("reason", "")),
                "raw": payload,
            }
    except Exception as e:
        return {
            "contains_answer": gold_norm in summary_norm,
            "reason": f"judge_error: {e}",
            "raw": None,
        }

    return {
        "contains_answer": gold_norm in summary_norm,
        "reason": "fallback heuristic",
        "raw": None,
    }


def load_samples(data_path: str) -> List[Dict[str, Any]]:
    data = load_json(data_path)
    if isinstance(data, list):
        return data
    raise ValueError("Expected the LongMemeEval JSON to be a list.")


ASSISTANT_CONTENT_CHAR_LIMIT = 4000
ASSISTANT_CONTENT_EDGE_CHARS = 2000


def truncate_assistant_content(role: Any, content: Any) -> str:
    content_text = str(content)
    if str(role).lower() != "assistant":
        return content_text
    if len(content_text) <= ASSISTANT_CONTENT_CHAR_LIMIT:
        return content_text
    return content_text[:ASSISTANT_CONTENT_EDGE_CHARS] + content_text[-ASSISTANT_CONTENT_EDGE_CHARS:]


def session_to_text(session: Any, session_id: str, date: str, index: int) -> str:
    items: List[Dict[str, str]] = [
        {
            "session_name": f"SESSION {index}",
            "session_time": date,
        }
    ]
    if isinstance(session, list):
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("speaker") or "unknown"
                content = turn.get("content") or turn.get("text") or ""
                items.append(
                    {
                        "role": str(role),
                        "content": truncate_assistant_content(role, content),
                        "session_time": date,
                    }
                )
            else:
                items.append(
                    {
                        "role": "unknown",
                        "content": str(turn),
                        "session_time": date,
                    }
                )
    else:
        items.append(
            {
                "role": "unknown",
                "content": str(session),
                "session_time": date,
            }
        )
    return json.dumps(items, ensure_ascii=False, indent=2)


def memory_key(sample: Dict[str, Any], sample_index: int) -> str:
    session_ids = sample.get("haystack_session_ids") or []
    if session_ids:
        raw = "\n".join(str(x) for x in session_ids)
    else:
        raw = f"{sample_index}:{sample.get('question_id', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_memory_generator(args: argparse.Namespace):
    return build_generator(
        api_type=args.memory_api_type,
        api_key=args.memory_api_key,
        base_url=args.memory_base_url,
        model_name=args.memory_model,
        max_tokens=args.memory_max_tokens,
        temperature=args.memory_temperature,
        use_schema=False,
        default_headers={"api-key": args.memory_api_key} if args.memory_api_key else None,
    )


def build_one_memory_cache(sample_index: int, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    key = memory_key(sample, sample_index)
    cache_dir = os.path.join(args.memory_root, key)
    done_file = os.path.join(cache_dir, "_DONE")

    if os.path.exists(done_file) and not args.force_rebuild_memory:
        return {"sample_index": sample_index, "memory_key": key, "status": "skipped"}

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    memory_store = InMemoryMemoryStore(dir_path=cache_dir)
    page_store = InMemoryPageStore(dir_path=cache_dir)
    generator = build_memory_generator(args)
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
        memory_agent.memorize(text)
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
    dump_json(os.path.join(cache_dir, "memory_meta.json"), meta)
    with open(done_file, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return {"sample_index": sample_index, "memory_key": key, "status": "built"}


def ensure_memory_caches(
    samples: List[Dict[str, Any]],
    sample_indices: List[int],
    args: argparse.Namespace,
) -> None:
    os.makedirs(args.memory_root, exist_ok=True)

    missing_indices = []
    for idx in sample_indices:
        cache_dir = os.path.join(args.memory_root, memory_key(samples[idx], idx))
        done_file = os.path.join(cache_dir, "_DONE")
        if args.force_rebuild_memory or not os.path.exists(done_file):
            missing_indices.append(idx)

    if not missing_indices:
        return

    if not args.build_memory_if_missing and not args.force_rebuild_memory:
        raise FileNotFoundError(
            f"Missing LongMemEval memory caches for {len(missing_indices)} samples under {args.memory_root}. "
            "Re-run with --build-memory-if-missing or provide complete caches."
        )

    results: List[Dict[str, Any]] = []
    desc = "Build LongMemEval memory"
    if args.memory_build_workers <= 1:
        for idx in tqdm(missing_indices, desc=desc):
            results.append(build_one_memory_cache(idx, samples[idx], args))
    else:
        with cf.ThreadPoolExecutor(max_workers=args.memory_build_workers) as executor:
            futures = {
                executor.submit(build_one_memory_cache, idx, samples[idx], args): idx
                for idx in missing_indices
            }
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
                results.append(future.result())

    results.sort(key=lambda r: int(r.get("sample_index", 10**9)))
    dump_json(os.path.join(os.path.dirname(args.memory_root), "build_manifest.json"), results)


def format_question_with_date(sample: Dict[str, Any]) -> str:
    question_date = str(sample.get("question_date", "")).strip()
    question = str(sample.get("question", "")).strip()
    if question_date:
        return f"[question_date]: {question_date}\n{question}"
    return question


def list_sorted_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing memory root: {root}")
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(dirs)


def build_answer_session_cache_index(memory_root: str) -> Dict[str, str]:
    """Map each session_id in memory_meta.json to its cache directory."""
    index: Dict[str, str] = {}
    for dirname in list_sorted_dirs(memory_root):
        cache_dir = os.path.join(memory_root, dirname)
        meta_path = os.path.join(cache_dir, "memory_meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        for session_id in meta.get("session_ids", []) or []:
            index[str(session_id)] = cache_dir
    return index


def resolve_cache_dir_for_sample(
    sample: Dict[str, Any],
    answer_session_cache_index: Dict[str, str],
    memory_root: str,
    sample_index: int,
) -> str:
    answer_session_ids = [str(x) for x in sample.get("answer_session_ids", []) or []]
    matched_dirs = {
        answer_session_cache_index[sid]
        for sid in answer_session_ids
        if sid in answer_session_cache_index
    }
    if len(matched_dirs) == 1:
        return next(iter(matched_dirs))
    if len(matched_dirs) > 1:
        raise ValueError(
            f"answer_session_ids map to multiple cache dirs for {sample.get('question_id')}: "
            f"{sorted(matched_dirs)}"
        )
    return os.path.join(memory_root, memory_key(sample, sample_index))


def extract_gold_page_ids(cache_dir: str, answer_session_ids: List[str]) -> List[str]:
    pages_path = os.path.join(cache_dir, "pages.json")
    pages = load_json(pages_path)
    gold_page_ids: List[str] = []
    answer_set = set(answer_session_ids)
    for idx, page in enumerate(pages):
        meta = page.get("meta", {}) if isinstance(page, dict) else {}
        session_id = meta.get("session_id")
        if session_id in answer_set:
            gold_page_ids.append(str(idx))
    return gold_page_ids


def add_page_index_to_headers(page_store: InMemoryPageStore) -> None:
    """Add visible page indices to headers without mutating the source cache files."""
    pages = page_store.load()
    for idx, page in enumerate(pages):
        prefix = f"[PAGE {idx}] "
        if not page.header.startswith(prefix):
            page.header = prefix + page.header
    if hasattr(page_store, "_pages"):
        page_store._pages = pages


def run_one_sample(
    sample: Dict[str, Any],
    cache_dir: str,
    research_generator,
    judge_generator,
    max_iters: int,
    args: argparse.Namespace,
    sample_dir: str,
) -> Dict[str, Any]:
    question = format_question_with_date(sample)
    gold_answer = sample.get("answer", "")
    answer_session_ids = sample.get("answer_session_ids", [])

    memory_store = InMemoryMemoryStore(dir_path=cache_dir)
    page_store = InMemoryPageStore(dir_path=cache_dir)
    add_page_index_to_headers(page_store)
    retrievers = build_retrievers(args, sample_dir, page_store)

    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=research_generator,
        max_iters=max_iters,
    )

    research_result = research_agent.research(question)
    research_summary = research_result.integrated_memory
    iterations = research_result.raw_memory.get("iterations", [])

    retrieved_page_ids: List[str] = []
    normalized_iterations: List[Dict[str, Any]] = []
    previous_sources: List[str] = []
    for it in iterations:
        temp_memory = it.get("temp_memory", {}) or {}
        sources = [str(x) for x in (temp_memory.get("sources", []) or []) if x is not None]
        new_sources = [x for x in sources if x not in previous_sources]
        retrieved_page_ids.extend(new_sources)
        normalized_iterations.append(
            {
                "step": it.get("step"),
                "plan": it.get("plan", {}),
                "planner_prompt": it.get("planner_prompt", ""),
                "search_trace": it.get("search_trace", {}),
                "integrate_prompt": it.get("integrate_prompt", ""),
                "retrieved_page_ids": new_sources,
                "cumulative_page_ids": sources,
                "temp_memory": temp_memory,
                "decision": it.get("decision", {}),
                "reflection_trace": it.get("reflection_trace", {}),
            }
        )
        previous_sources = sources

    gold_page_ids = extract_gold_page_ids(cache_dir, answer_session_ids)
    judge = judge_summary(judge_generator, question, gold_answer, research_summary)

    retrieved_gold_page = any(pid in set(gold_page_ids) for pid in retrieved_page_ids)

    return {
        "question_id": sample.get("question_id"),
        "question_type": sample.get("question_type"),
        "question": question,
        "gold_answer": gold_answer,
        "answer_session_ids": answer_session_ids,
        "gold_page_ids": gold_page_ids,
        "cache_dir": cache_dir,
        "research_summary": research_summary,
        "retrieved_page_ids": retrieved_page_ids,
        "retrieved_gold_page": retrieved_gold_page,
        "judge": judge,
        "iterations": normalized_iterations,
        "raw_memory": research_result.raw_memory,
    }


def build_generators(args: argparse.Namespace):
    research_generator = build_generator(
        api_type=args.research_api_type,
        api_key=args.research_api_key,
        base_url=args.research_base_url,
        model_name=args.research_model,
        max_tokens=args.research_max_tokens,
        temperature=args.research_temperature,
        use_schema=True,
        default_headers={"api-key": args.research_api_key} if args.research_api_key else None,
    )

    judge_api_type = args.judge_api_type or args.research_api_type
    judge_model = args.judge_model or args.research_model
    judge_generator = build_generator(
        api_type=judge_api_type,
        api_key=args.judge_api_key or args.research_api_key,
        base_url=args.judge_base_url or args.research_base_url,
        model_name=judge_model,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        use_schema=False,
        default_headers={"api-key": (args.judge_api_key or args.research_api_key)}
        if (args.judge_api_key or args.research_api_key)
        else None,
    )

    return research_generator, judge_generator


def process_sample(
    idx: int,
    sample: Dict[str, Any],
    answer_session_cache_index: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sample_dir = os.path.join(args.outdir, f"{idx:06d}_{sample.get('question_id', f'sample_{idx}')}")
    os.makedirs(sample_dir, exist_ok=True)

    try:
        cache_dir = resolve_cache_dir_for_sample(
            sample=sample,
            answer_session_cache_index=answer_session_cache_index,
            memory_root=args.memory_root,
            sample_index=idx,
        )
        research_generator, judge_generator = build_generators(args)
        result = run_one_sample(
            sample=sample,
            cache_dir=cache_dir,
            research_generator=research_generator,
            judge_generator=judge_generator,
            max_iters=args.max_iters,
            args=args,
            sample_dir=sample_dir,
        )
        result["sample_index"] = idx
        result["sample_dir"] = sample_dir
        dump_json(os.path.join(sample_dir, "research_trace.json"), result)
        return result
    except Exception as e:
        err = {
            "sample_index": idx,
            "question_id": sample.get("question_id"),
            "question": sample.get("question"),
            "question_with_date": format_question_with_date(sample),
            "error": str(e),
            "sample_dir": sample_dir,
        }
        dump_json(os.path.join(sample_dir, "error.json"), err)
        return err


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemeEval test for MemPro")
    parser.add_argument("--data", type=str, default="./data/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument(
        "--memory-root",
        type=str,
        default="./data/longmemeval/memory/_memory_cache",
        help="Root directory that contains one cache subdir per sample.",
    )
    parser.add_argument("--outdir", type=str, default="./results/longmemeval")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1, help="Number of questions to process in parallel.")
    parser.add_argument(
        "--build-memory-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically build missing memory caches before evaluation.",
    )
    parser.add_argument(
        "--force-rebuild-memory",
        action="store_true",
        help="Rebuild target-split memory caches even if they already exist.",
    )
    parser.add_argument(
        "--memory-build-workers",
        type=int,
        default=int(env_value("MEMORY_BUILD_WORKERS", env_value("MEMPRO_MEMORY_WORKERS", "1")) or "1"),
        help="Number of workers for memory-cache construction.",
    )

    parser.add_argument("--use-bm25", action="store_true", help="Enable BM25 keyword retrieval.")
    parser.add_argument("--bm25-threads", type=int, default=1)
    parser.add_argument("--use-dense", action="store_true", help="Enable dense vector retrieval.")
    parser.add_argument(
        "--dense-model",
        type=str,
        default="./models/bge-m3",
        help="Dense model name or local model path.",
    )
    parser.add_argument("--dense-api-url", type=str, default=None)
    parser.add_argument("--dense-devices", type=str, default="cuda:0")

    parser.add_argument("--memory-api-key", type=str, default=role_env("MEMORY", "API_KEY", "empty"))
    parser.add_argument("--memory-base-url", type=str, default=role_env("MEMORY", "BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--memory-model", type=str, default=role_env("MEMORY", "MODEL", "gpt-4o-mini"))
    parser.add_argument("--memory-api-type", type=str, default=role_env("MEMORY", "API_TYPE", "openai"), choices=["openai", "vllm"])
    parser.add_argument("--memory-temperature", type=float, default=0.3)
    parser.add_argument("--memory-max-tokens", type=int, default=1024)

    parser.add_argument("--research-api-key", type=str, default=role_env("RESEARCH", "API_KEY", "empty"))
    parser.add_argument("--research-base-url", type=str, default=role_env("RESEARCH", "BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--research-model", type=str, default=role_env("RESEARCH", "MODEL", "gpt-4o-mini"))
    parser.add_argument("--research-api-type", type=str, default=role_env("RESEARCH", "API_TYPE", "openai"), choices=["openai", "vllm"])
    parser.add_argument("--research-temperature", type=float, default=0.3)
    parser.add_argument("--research-max-tokens", type=int, default=2048)

    parser.add_argument("--judge-api-key", type=str, default=role_env("JUDGE", "API_KEY", None))
    parser.add_argument("--judge-base-url", type=str, default=role_env("JUDGE", "BASE_URL", None))
    parser.add_argument("--judge-model", type=str, default=role_env("JUDGE", "MODEL", None))
    parser.add_argument("--judge-api-type", type=str, default=role_env("JUDGE", "API_TYPE", None), choices=["openai", "vllm"])
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=512)

    args = parser.parse_args()

    samples = load_samples(args.data)
    end_idx = args.end_idx if args.end_idx is not None else len(samples)
    sample_indices = list(range(args.start_idx, min(end_idx, len(samples))))
    ensure_memory_caches(samples, sample_indices, args)
    answer_session_cache_index = build_answer_session_cache_index(args.memory_root)

    os.makedirs(args.outdir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        for idx in tqdm(sample_indices, desc="LongMemeEval"):
            results.append(
                process_sample(
                    idx=idx,
                    sample=samples[idx],
                    answer_session_cache_index=answer_session_cache_index,
                    args=args,
                )
            )
    else:
        with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_sample,
                    idx,
                    samples[idx],
                    answer_session_cache_index,
                    args,
                ): idx
                for idx in sample_indices
            }
            with tqdm(total=len(futures), desc="LongMemeEval") as pbar:
                for fut in cf.as_completed(futures):
                    results.append(fut.result())
                    pbar.update(1)

    results.sort(key=lambda r: int(r.get("sample_index", 10**9)))

    out_file = os.path.join(args.outdir, "results.json")
    dump_json(out_file, results)

    def normalized_category(sample: Dict[str, Any]) -> str:
        raw = str(sample.get("question_type", "")).strip().lower().replace("_", "-")
        mapping = {
            "single-session-user": "Single-User",
            "single-user": "Single-User",
            "single-session-assistant": "Single-Assistant",
            "single-assistant": "Single-Assistant",
            "single-session-preference": "Single-Preference",
            "single-preference": "Single-Preference",
            "multi-session": "Multi-Session",
            "temporal": "Temporal",
            "temporal-reasoning": "Temporal",
            "knowledge-update": "Knowledge-Update",
        }
        return mapping.get(raw, sample.get("question_type", "Unknown"))

    category_order = [
        "Temporal",
        "Multi-Session",
        "Knowledge-Update",
        "Single-User",
        "Single-Assistant",
        "Single-Preference",
    ]

    judged = [r for r in results if "judge" in r]
    correct = [r for r in judged if r.get("judge", {}).get("contains_answer")]
    by_category: Dict[str, Dict[str, int]] = {cat: {"total": 0, "correct": 0} for cat in category_order}
    for r in judged:
        cat = normalized_category(r)
        bucket = by_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if r.get("judge", {}).get("contains_answer"):
            bucket["correct"] += 1

    summary_rows = []
    for r in results:
        judge = r.get("judge", {}) if isinstance(r, dict) else {}
        summary_rows.append(
            {
                "sample_index": r.get("sample_index"),
                "question_id": r.get("question_id"),
                "question_type": r.get("question_type"),
                "category": normalized_category(r),
                "question": r.get("question"),
                "gold_answer": r.get("gold_answer"),
                "research_summary": r.get("research_summary"),
                "judge_contains_answer": judge.get("contains_answer"),
                "judge_reason": judge.get("reason"),
                "retrieved_gold_page": r.get("retrieved_gold_page"),
                "gold_page_ids": r.get("gold_page_ids"),
                "retrieved_page_ids": r.get("retrieved_page_ids"),
                "sample_dir": r.get("sample_dir"),
                "error": r.get("error"),
            }
        )

    summary = {
        "total": len(results),
        "judged": len(judged),
        "judge_correct": len(correct),
        "llm_judge_accuracy": (len(correct) / len(judged)) if judged else 0.0,
        "judge_accuracy": (len(correct) / len(judged)) if judged else 0.0,
        "retrieved_gold_page": sum(1 for r in judged if r.get("retrieved_gold_page")),
        "categories": {
            cat: {
                "total": by_category.get(cat, {"total": 0})["total"],
                "correct": by_category.get(cat, {"correct": 0})["correct"],
                "accuracy": (
                    by_category.get(cat, {"total": 0, "correct": 0})["correct"] / by_category.get(cat, {"total": 0, "correct": 0})["total"]
                    if by_category.get(cat, {"total": 0, "correct": 0})["total"]
                    else 0.0
                ),
            }
            for cat in category_order
        },
        "out_file": out_file,
        "rows_file": os.path.join(args.outdir, "summary_rows.json"),
    }
    dump_json(os.path.join(args.outdir, "summary_rows.json"), summary_rows)
    dump_json(os.path.join(args.outdir, f"summary_{args.start_idx}_{end_idx - 1}.json"), {**summary, "rows": summary_rows})
    dump_json(os.path.join(args.outdir, "summary.json"), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
