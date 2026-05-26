#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemPro 框架 + LoCoMo 数据集测试文件

结合 locomoqa_v3.py 的数据处理逻辑和 MemPro 框架，测试在多轮对话数据上的效果。
"""

import sys
import os
import re
import json
import math
import time
import shutil
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

from mempro_memory.utils.env import get_env_or_default, load_local_env
from mempro_memory.prompts import build_final_summarize_prompt

from mempro_memory import (
    MemoryAgent,
    ResearchAgent,
    InMemoryMemoryStore,
    InMemoryPageStore,
    MemoryState,
    Page,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGenerator,
    VLLMGeneratorConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== 数据加载：借鉴自 locomoqa_v3.py ==========

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return the list of samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")

def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    Extract sessions as (idx, timestamp, turns, optional_session_summary).
    """
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        original_idx = int(m.group(1))
        idx = original_idx - 1
        ts = conv_obj.get(f"session_{original_idx}_date_time", "")
        ssum = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions

def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    # 将时间信息放在最前面，使用更突出的格式
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")  # 空行分隔
    turn_time_prefix = f"[Session date: {ts}] " if ts else ""
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id  = turn.get("dia_id", "")
        text    = turn.get("text", "")
        lines.append(f"{turn_time_prefix}{speaker} ({dia_id}): {text}")
    
    if session_summary:
        lines.append("")
        lines.append(f"{turn_time_prefix}Session {idx} summary: {session_summary}")
    
    return "\n".join(lines).strip()

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Build session chunks from a sample."""
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect QA items from a sample."""
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for idx, q in enumerate(sample.get("qa", []), 1):
        qas.append({
            "sample_id": sid,
            "question_index": idx,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas

# ========== Prompt 设计：完全借鉴自 locomoqa_v3.py ==========

def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """尽量把模型输出（string/dict）解析成 dict，失败返回 None。"""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None

def _load_final_validator_hooks():
    try:
        from mempro_memory.prompts import final_summarize_prompts as final_hooks
        return (
            getattr(final_hooks, "build_final_answer_validator_prompt", None),
            getattr(final_hooks, "FINAL_ANSWER_VALIDATOR_SCHEMA", None),
            getattr(final_hooks, "postprocess_final_answer", None),
        )
    except Exception:
        return None, None, None


def _load_final_slot_router_hooks():
    try:
        from mempro_memory.prompts import final_summarize_prompts as final_hooks
        return (
            getattr(final_hooks, "build_final_slot_router_prompt", None),
            getattr(final_hooks, "FINAL_SLOT_ROUTER_SCHEMA", None),
            getattr(final_hooks, "build_slot_routed_final_prompt", None),
            getattr(final_hooks, "FINAL_SLOT_ANSWER_SCHEMA", None),
            getattr(final_hooks, "postprocess_final_answer", None),
        )
    except Exception:
        return None, None, None, None, None


def _run_slot_routed_answer(
    category: Optional[int],
    summary: str,
    question: str,
    generator,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    (
        router_prompt_builder,
        router_schema,
        answer_prompt_builder,
        answer_schema,
        postprocess_final_answer,
    ) = _load_final_slot_router_hooks()
    if not router_prompt_builder or not router_schema or not answer_prompt_builder:
        return None, None

    router_prompt = router_prompt_builder(question, summary, category)
    router_raw = generator.generate_single(prompt=router_prompt, schema=router_schema)
    router_text = router_raw.get("text", "")
    router_data = router_raw.get("json") or safe_json_extract(router_text) or {}

    answer_prompt = answer_prompt_builder(category, summary, question, router_data)
    if answer_schema:
        answer_raw = generator.generate_single(prompt=answer_prompt, schema=answer_schema)
        answer_text = answer_raw.get("text", "")
        answer_data = answer_raw.get("json") or safe_json_extract(answer_text) or {}
        final_answer = str(answer_data.get("final_answer") or "").strip()
    else:
        answer_raw = generator.generate_single(prompt=answer_prompt)
        answer_text = answer_raw.get("text", "")
        answer_data = {}
        final_answer = answer_text.strip()

    if postprocess_final_answer:
        final_answer = postprocess_final_answer(question, summary, final_answer, final_answer, router_data)

    trace = {
        "slot_router": {
            "prompt": router_prompt,
            "raw_text": router_text,
            "json": router_data,
        },
        "slot_answer": {
            "prompt": answer_prompt,
            "raw_text": answer_text,
            "json": answer_data,
        },
    }
    return final_answer, trace


def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    """根据category选择不同的prompt"""
    try:
        routed_answer, _ = _run_slot_routed_answer(category, summary, question, generator)
        if routed_answer:
            return routed_answer
    except Exception as e:
        print(f"[WARN] slot-routed final answer failed: {e}")

    prompt = build_final_summarize_prompt(category, summary, question)
    raw = generator.generate_single(prompt=prompt)
    draft_answer = raw.get("text", "").strip()
    validator_prompt_builder, validator_schema, postprocess_final_answer = _load_final_validator_hooks()
    if not validator_prompt_builder or not validator_schema:
        return draft_answer
    try:
        validator_prompt = validator_prompt_builder(question, summary, draft_answer)
        validator_raw = generator.generate_single(prompt=validator_prompt, schema=validator_schema)
        validator_text = validator_raw.get("text", "")
        validator_data = validator_raw.get("json") or safe_json_extract(validator_text) or {}
        final_answer = str(validator_data.get("final_answer") or draft_answer).strip()
        if postprocess_final_answer:
            final_answer = postprocess_final_answer(question, summary, draft_answer, final_answer, validator_data)
        return final_answer or draft_answer
    except Exception as e:
        print(f"[WARN] final answer validator failed: {e}")
        return draft_answer


def answer_with_summary_trace(category: Optional[int], summary: str, question: str, generator) -> Tuple[str, Dict[str, Any]]:
    try:
        routed_answer, routed_trace = _run_slot_routed_answer(category, summary, question, generator)
        if routed_answer:
            return routed_answer, routed_trace or {}
    except Exception as e:
        print(f"[WARN] slot-routed final answer failed: {e}")

    prompt = build_final_summarize_prompt(category, summary, question)
    raw = generator.generate_single(prompt=prompt)
    draft_answer = raw.get("text", "").strip()
    trace: Dict[str, Any] = {
        "draft_prompt": prompt,
        "draft_raw_text": raw.get("text", ""),
        "draft_answer": draft_answer,
        "validator": None,
    }
    validator_prompt_builder, validator_schema, postprocess_final_answer = _load_final_validator_hooks()
    if not validator_prompt_builder or not validator_schema:
        return draft_answer, trace
    try:
        validator_prompt = validator_prompt_builder(question, summary, draft_answer)
        validator_raw = generator.generate_single(prompt=validator_prompt, schema=validator_schema)
        validator_text = validator_raw.get("text", "")
        validator_data = validator_raw.get("json") or safe_json_extract(validator_text) or {}
        final_answer = str(validator_data.get("final_answer") or draft_answer).strip() or draft_answer
        postprocess_trace = None
        if postprocess_final_answer:
            postprocessed_answer = postprocess_final_answer(question, summary, draft_answer, final_answer, validator_data)
            if postprocessed_answer != final_answer:
                postprocess_trace = {
                    "before": final_answer,
                    "after": postprocessed_answer,
                }
                final_answer = postprocessed_answer
        trace["validator"] = {
            "prompt": validator_prompt,
            "raw_text": validator_text,
            "json": validator_data,
            "postprocess": postprocess_trace,
        }
        return final_answer, trace
    except Exception as e:
        trace["validator"] = {"error": str(e)}
        print(f"[WARN] final answer validator failed: {e}")
        return draft_answer, trace

# ========== 指标计算：借鉴自 eval_metric_locomo.py ==========

LOCOMO_CATEGORY_TYPES = {
    1: "Multi Hop",
    2: "Temporal",
    3: "Open Domain",
    4: "Single Hop",
    5: "Adversarial",
}

LOCOMO_PAPER_TYPE_ORDER = ["Single Hop", "Multi Hop", "Temporal", "Open Domain"]

JUDGE_PROMPT_TEMPLATE = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG. The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic. The generated answer might be much longer, but you should be generous with your grading. As long as it touches on the same topic as the gold answer, it should be counted as CORRECT. For time related questions, the gold answer will often be a specific date, month, year, or time period. The generated answer might be longer or use relative references like "last week" or "the week before", but you should be generous if it refers to the same date or time period as the gold answer. Return the label CORRECT or WRONG in JSON format with the key "label". Do not include both CORRECT and WRONG in your response. Question: {question} Gold answer: {gold_answer} Generated answer: {generated_answer}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["CORRECT", "WRONG"]
        }
    },
    "required": ["label"],
    "additionalProperties": False
}

def locomo_category_type(category: Any) -> str:
    """Map official LoCoMo category ids to paper-aligned type names."""
    try:
        category = int(category)
    except (TypeError, ValueError):
        return "Unknown"
    return LOCOMO_CATEGORY_TYPES.get(category, "Unknown")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

def normalize_evidence_sessions(evidence: Any) -> List[int]:
    """Convert LoCoMo evidence annotations like D1:3 to 0-based session indexes."""
    if evidence is None:
        return []

    raw_items = evidence if isinstance(evidence, list) else [evidence]
    session_indexes: List[int] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue

        # LoCoMo evidence uses the dialogue/session prefix, e.g. D3:7 means session 3.
        match = re.search(r"\bD\s*(\d+)\s*:\s*\d+\b", text, flags=re.IGNORECASE)
        if match:
            session_idx = int(match.group(1)) - 1
        elif text.isdigit():
            session_idx = int(text) - 1
        else:
            continue

        if session_idx >= 0:
            session_indexes.append(session_idx)

    # Preserve order while de-duplicating.
    deduped: List[int] = []
    seen = set()
    for idx in session_indexes:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def format_evidence_refs(evidence: Any) -> List[str]:
    """Format LoCoMo evidence annotations into readable Chinese descriptions."""
    if evidence is None:
        return []

    raw_items = evidence if isinstance(evidence, list) else [evidence]
    formatted: List[str] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue

        match = re.search(r"\bD\s*(\d+)\s*:\s*(\d+)\b", text, flags=re.IGNORECASE)
        if match:
            session_idx = int(match.group(1)) - 1
            turn_idx = int(match.group(2))
            formatted.append(f"答案来自第{session_idx}个session的第{turn_idx}轮对话")
            continue

        if text.isdigit():
            formatted.append(f"答案来自第{int(text) - 1}个session")
            continue

        formatted.append(text)
    return formatted

def llm_judge_score(label: str) -> float:
    return 1.0 if str(label).strip().upper() == "CORRECT" else 0.0


def approx_bytes(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return len(value.encode("utf-8"))


def approx_tokens_from_bytes(byte_count: int) -> int:
    # User requested an approximate conversion using bytes / 4.
    return max(0, round(byte_count / 4))


def make_usage_stats(input_payload: Any, output_payload: Any) -> Dict[str, int]:
    input_bytes = approx_bytes(input_payload)
    output_bytes = approx_bytes(output_payload)
    input_tokens = approx_tokens_from_bytes(input_bytes)
    output_tokens = approx_tokens_from_bytes(output_bytes)
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "input_tokens_approx": input_tokens,
        "output_tokens_approx": output_tokens,
        "total_tokens_approx": input_tokens + output_tokens,
    }

def call_llm_judge(
    client: OpenAI,
    question: str,
    gold_answer: str,
    generated_answer: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> str:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=str(generated_answer),
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_response",
                        "schema": JUDGE_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.0,
            )
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            return result_json.get("label", "WRONG")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error calling LLM judge (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to call LLM judge after {max_retries} attempts: {e}")
                return "WRONG"

def compute_metrics_by_category(items, pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg = defaultdict(list)
    type_agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        category_type = locomo_category_type(cat)
        gold = ex.get("gold_answer", "")
        pred = ""
        val = ex.get(pred_key, "")
        if isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        judge_label = ex.get("llm_as_judge", "WRONG")
        judge_score = float(ex.get("llm_as_judge_score", llm_judge_score(judge_label)))
        agg[cat].append((f1, b1, judge_score))
        type_agg[category_type].append((f1, b1, judge_score))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "category_type": category_type,
            "question": ex.get("question", ""),
            "gold_answer": str(gold),
            "evidence": format_evidence_refs(ex.get("evidence")),
            "research_summary": str(ex.get("research_summary", "")),
            "summary_answer": str(pred),
            "iterations": ex.get("iterations"),
            "research_trace_file": ex.get("research_trace_file"),
            "usage": ex.get("usage"),
            "research_agent_usage": ex.get("research_agent_usage"),
            "llm_as_judge": judge_label,
            "llm_as_judge_score": judge_score,
            "F1": f1,
            "BLEU1": b1
        })
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            f1_avg = sum(s[0] for s in scores)/len(scores)
            b1_avg = sum(s[1] for s in scores)/len(scores)
            judge_avg = sum(s[2] for s in scores)/len(scores)
            summary.append({
                "category": cat,
                "category_type": locomo_category_type(cat),
                "count": len(scores),
                "F1_avg": f1_avg,
                "BLEU1_avg": b1_avg,
                "llm_as_judge_avg": judge_avg,
            })

    type_summary = []
    ordered_types = [
        type_name
        for type_name in LOCOMO_PAPER_TYPE_ORDER
        if type_name in type_agg
    ]
    ordered_types.extend(
        sorted(
            type_name
            for type_name in type_agg.keys()
            if type_name not in LOCOMO_PAPER_TYPE_ORDER
        )
    )
    for type_name in ordered_types:
        scores = type_agg[type_name]
        if scores:
            f1_avg = sum(s[0] for s in scores)/len(scores)
            b1_avg = sum(s[1] for s in scores)/len(scores)
            judge_avg = sum(s[2] for s in scores)/len(scores)
            type_summary.append({
                "category_type": type_name,
                "count": len(scores),
                "F1_avg": f1_avg,
                "BLEU1_avg": b1_avg,
                "llm_as_judge_avg": judge_avg,
            })

    return summary, type_summary, rows

# ========== 核心处理逻辑 ==========

def process_sample(
    sample: Dict[str, Any], 
    sample_index: int, 
    outdir: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    judge_api_key: str,
    judge_base_url: str,
    judge_model: str,
    use_schema: bool = False,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai",
    question_workers: int = 4,
    force_rerun: bool = False,
    rebuild_memory: bool = False,
    selected_question_indices: Optional[List[int]] = None,
):
    """
    使用 MemPro 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答
    """
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建会话块
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"会话数: {len(session_chunks)}")
        if session_chunks:
            print(f"第一个会话预览:\n{session_chunks[0][:400]}...")
        
        # 创建输出目录
        sample_results_dir = os.path.join(outdir, sample_id)
        if force_rerun and os.path.exists(sample_results_dir):
            shutil.rmtree(sample_results_dir)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"输出目录: {sample_results_dir}")

        memory_state_path = os.path.join(sample_results_dir, "memory_state.json")
        pages_path = os.path.join(sample_results_dir, "pages.json")
        if rebuild_memory and not force_rerun:
            for artifact_path in (
                memory_state_path,
                pages_path,
                os.path.join(sample_results_dir, "page_index"),
                os.path.join(sample_results_dir, "bm25_index"),
                os.path.join(sample_results_dir, "dense_index"),
            ):
                if os.path.isdir(artifact_path):
                    shutil.rmtree(artifact_path)
                elif os.path.isfile(artifact_path):
                    os.remove(artifact_path)
        memory_artifacts_exist = os.path.exists(memory_state_path) and os.path.exists(pages_path)
        
        # 2. 创建共享存储
        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)
        
        # 3. 创建 Memory Generator
        print(f"\n步骤 1: 创建 Memory Generator")
        if memory_api_type == "openai":
            memory_generator_config = OpenAIGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=8192
            )
            memory_generator = OpenAIGenerator(memory_generator_config.__dict__)
        elif memory_api_type == "vllm":
            memory_generator_config = VLLMGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=8192
            )
            memory_generator = VLLMGenerator(memory_generator_config.__dict__)
        print(f"[OK] Memory Generator 创建完成")
        
        # 4. 使用 MemoryAgent 构建记忆（将每个 session 作为一条消息）
        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator
        )
        
        if force_rerun or rebuild_memory or not memory_artifacts_exist:
            for i, session_chunk in enumerate(session_chunks, 1):
                print(f"  处理会话 {i}/{len(session_chunks)}...")
                memory_update = memory_agent.memorize(session_chunk)
        else:
            print("[INFO] 复用已有 memory_state.json 与 pages.json，不重新生成 memory")
        
        # 查看构建的记忆
        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        
        # 显示记忆摘要
        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        
        # 保存记忆状态
        memory_state_file = memory_state_path
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        
        # 5. 创建检索器
        print(f"\n步骤 3: 创建检索器")
        retrievers = {}
        
        # 索引检索器
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(page_index_dir):
                shutil.rmtree(page_index_dir)
                print(f"[INFO] 清理已存在的页面索引目录: {page_index_dir}")
            
            index_config = IndexRetrieverConfig(
                index_dir=page_index_dir
            )
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            print(f"[OK] 索引检索器创建成功")
        except Exception as e:
            print(f"[WARN] 索引检索器创建失败: {e}")
        
        # BM25 检索器
        try:
            bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(bm25_index_dir):
                shutil.rmtree(bm25_index_dir)
                print(f"[INFO] 清理已存在的 BM25 索引目录: {bm25_index_dir}")
            
            bm25_config = BM25RetrieverConfig(
                index_dir=bm25_index_dir,
                threads=1
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print(f"[OK] BM25 检索器创建成功")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
        
        # Dense 检索器
        try:
            dense_index_dir = os.path.join(sample_results_dir, "dense_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(dense_index_dir):
                shutil.rmtree(dense_index_dir)
                print(f"[INFO] 清理已存在的 Dense 索引目录: {dense_index_dir}")

            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                model_name="BAAI/bge-m3"
            )

            # dense_config = DenseRetrieverConfig(
            #     index_dir=dense_index_dir,
            #     api_url="http://localhost:8001"  # API 模式：所有进程共享一个模型服务
            # )

            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print(f"[OK] Dense 检索器创建成功")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
        
        print(f"[INFO] 成功创建 {len(retrievers)} 个检索器")
        
        print(f"\n步骤 4: 创建 Research Generator 和 Working Generator")
        if research_api_type == "openai":
            research_generator_config = OpenAIGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=8192,
                use_schema=use_schema
            )
            research_generator = OpenAIGenerator(research_generator_config.__dict__)
        elif research_api_type == "vllm":
            research_generator_config = VLLMGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=8192,
                use_schema=use_schema
            )
            research_generator = VLLMGenerator(research_generator_config.__dict__)

        if working_api_type == "openai":
            working_generator_config = OpenAIGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=8192,
                use_schema=use_schema
            )
            working_generator = OpenAIGenerator(working_generator_config.__dict__)
        elif working_api_type == "vllm":
            working_generator_config = VLLMGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=8192,
                use_schema=use_schema
            )
            working_generator = VLLMGenerator(working_generator_config.__dict__)
        print(f"[OK] Research Generator 和 Working Generator 创建完成")


        # 6. ResearchAgent 将在每个问题的 worker 中单独创建，避免并行时共享状态
        print(f"\n步骤 5: 准备 ResearchAgent 依赖")
        print(f"[OK] ResearchAgent 依赖准备完成")
        
        # 7. 进行问答
        print(f"\n步骤 6: 进行问答")
        qas = collect_qa_items_for_sample(sample)
        if selected_question_indices is not None:
            selected_question_index_set = set(selected_question_indices)
            qas = [qa for qa in qas if qa.get("question_index") in selected_question_index_set]
        print(f"共有 {len(qas)} 个问题需要回答")
        if selected_question_indices is not None:
            print(f"过滤后的问题索引: {sorted(selected_question_indices)}")

        if not qas:
            print("[WARN] 过滤后没有可运行的问题，跳过该样本")
            return []
        
        # 定义处理单个问题的worker函数
        def process_question(qi_with_index):
            """处理单个问题的worker函数"""
            i, qi = qi_with_index
            q = qi.get("question") or ""
            gold = qi.get("answer")
            cat = qi.get("category")
            question_index = qi.get("question_index", i)
            
            print(f"\n--- 问题 {question_index} ({i}/{len(qas)}) ---")
            print(f"问题: {q}")
            print(f"标准答案: {gold}")
            print(f"分类: {cat}")
            
            if cat == 5:
                return None

            try:
                research_agent = ResearchAgent(
                    page_store=page_store,
                    memory_store=memory_store,
                    retrievers=retrievers,
                    generator=research_generator,
                    max_iters=3,
                    build_retrievers=False
                )

                # 使用 ResearchAgent 进行研究
                print(f"[问题 {i}] 正在进行深度研究...")
                result = research_agent.research(q)
                research_summary = result.integrated_memory
                iterations = result.raw_memory.get("iterations", [])
                print(f"[问题 {i}] [OK] 研究完成！迭代次数: {len(iterations)}")
                print(f"[问题 {i}] 研究摘要: {research_summary[:200]}...")

                formatted_iterations = []
                for item in iterations:
                    formatted_iterations.append({
                        "step": item.get("step"),
                        "route": item.get("route"),
                        "selected_page_ids_before_step": item.get("selected_page_ids_before_step"),
                        "retrieved_page_ids": item.get("retrieved_page_ids"),
                        "new_page_ids": item.get("new_page_ids"),
                        "repeated_page_ids": item.get("repeated_page_ids"),
                        "plan": item.get("plan"),
                        "temp_memory": item.get("temp_memory"),
                        "decision": item.get("decision"),
                        "llm_calls": item.get("llm_calls"),
                        "usage": item.get("usage"),
                    })

                research_trace = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "question_index": question_index,
                    "question": q,
                    "ground_truth": gold,
                    "router": result.raw_memory.get("router"),
                    "route": result.raw_memory.get("route"),
                    "selected_page_ids": result.raw_memory.get("selected_page_ids"),
                    "iterations": formatted_iterations,
                    "integrated_memory": result.integrated_memory,
                }
                
                # 基于研究结果生成答案（根据category选择不同prompt）
                print(f"[问题 {i}] 生成答案...")
                summary_answer, answer_trace = answer_with_summary_trace(cat, research_summary, q, working_generator)
                f1 = f1_score(summary_answer, gold)
                b1 = bleu1_score(summary_answer, gold)
                judge_client = OpenAI(
                    api_key=judge_api_key,
                    base_url=judge_base_url.rstrip("/") if judge_base_url else None,
                )
                judge_label = call_llm_judge(
                    client=judge_client,
                    question=q,
                    gold_answer=str(gold),
                    generated_answer=str(summary_answer),
                    model=judge_model,
                )
                judge_score = llm_judge_score(judge_label)

                research_usage = result.raw_memory.get("usage") or make_usage_stats(q, research_summary)
                answer_usage = make_usage_stats(
                    {
                        "question": q,
                        "research_summary": research_summary,
                        "answer_generation": answer_trace,
                    },
                    summary_answer,
                )
                judge_usage = make_usage_stats(
                    {
                        "question": q,
                        "gold_answer": str(gold),
                        "generated_answer": str(summary_answer),
                    },
                    judge_label,
                )
                question_usage = {
                    "research": research_usage,
                    "answer": answer_usage,
                    "judge": judge_usage,
                    "input_bytes": research_usage["input_bytes"] + answer_usage["input_bytes"] + judge_usage["input_bytes"],
                    "output_bytes": research_usage["output_bytes"] + answer_usage["output_bytes"] + judge_usage["output_bytes"],
                    "input_tokens_approx": research_usage["input_tokens_approx"] + answer_usage["input_tokens_approx"] + judge_usage["input_tokens_approx"],
                    "output_tokens_approx": research_usage["output_tokens_approx"] + answer_usage["output_tokens_approx"] + judge_usage["output_tokens_approx"],
                }
                question_usage["total_tokens_approx"] = question_usage["input_tokens_approx"] + question_usage["output_tokens_approx"]
                
                print(f"[问题 {i}] 预测答案: {summary_answer}")
                print(f"[问题 {i}] F1: {f1:.4f}, BLEU1: {b1:.4f}, LLM_JUDGE: {judge_label}")

                research_trace["summary_answer"] = summary_answer
                research_trace["answer_generation"] = answer_trace
                research_trace["f1"] = f1
                research_trace["b1"] = b1
                research_trace["llm_as_judge"] = judge_label
                research_trace["llm_as_judge_score"] = judge_score
                research_trace["usage"] = question_usage
                research_trace["research_agent_usage"] = result.raw_memory.get("usage")

                # 保存单个问题的研究轨迹
                trace_file = os.path.join(sample_results_dir, f"research_trace_q{question_index}.json")
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[问题 {i}] [INFO] 研究轨迹已保存: {trace_file}")
                
                qa_result = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "question_index": question_index,
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "evidence": format_evidence_refs(qi.get("evidence")),
                    "research_summary": research_summary,
                    "summary_answer": summary_answer,
                    "f1": f1,
                    "b1": b1,
                    "llm_as_judge": judge_label,
                    "llm_as_judge_score": judge_score,
                    "usage": question_usage,
                    "research_agent_usage": result.raw_memory.get("usage"),
                    "iterations": len(result.raw_memory.get("iterations", [])),
                    "research_trace_file": trace_file
                }
                return qa_result
            
            except Exception as e:
                print(f"[问题 {i}] [ERROR] 处理问题失败: {e}")
                import traceback
                traceback.print_exc()
                qa_result = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "question_index": question_index,
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "evidence": format_evidence_refs(qi.get("evidence")),
                    "error": str(e)
                }
                return qa_result
        
        # 处理所有问题
        qa_items_with_index = [(i, qi) for i, qi in enumerate(qas, 1)]

        question_workers = max(1, min(question_workers, len(qa_items_with_index))) if qa_items_with_index else 1
        print(f"开始并行处理 {len(qa_items_with_index)} 个问题，worker 数: {question_workers}")

        ordered_results = {}
        with ThreadPoolExecutor(max_workers=question_workers) as executor:
            future_to_index = {
                executor.submit(process_question, qa_item): qa_item[0]
                for qa_item in qa_items_with_index
            }

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="处理问题"):
                question_index = future_to_index[future]
                try:
                    result = future.result()
                    # 过滤掉 None 结果（category == 5 的问题）
                    if result is not None:
                        ordered_results[question_index] = result
                except Exception as e:
                    print(f"[问题 {question_index}] [ERROR] Future 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    ordered_results[question_index] = {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "question_index": qas[question_index - 1].get("question_index"),
                        "question": qas[question_index - 1].get("question", ""),
                        "gold_answer": qas[question_index - 1].get("answer"),
                        "category": qas[question_index - 1].get("category"),
                        "evidence": format_evidence_refs(qas[question_index - 1].get("evidence")),
                        "error": str(e)
                    }

        qa_results = [ordered_results[i] for i in sorted(ordered_results.keys())]
        
        # 保存结果
        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        
        # 保存所有研究轨迹的汇总
        all_research_traces = []
        for i, qa_result in enumerate(qa_results, 1):
            if "research_trace_file" in qa_result:
                trace_file = qa_result["research_trace_file"]
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                        all_research_traces.append({
                            "question_index": qa_result.get("question_index", i),
                            "question": qa_result["question"],
                            "category": qa_result["category"],
                            "research_trace": trace_data
                        })
        
        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] 所有研究轨迹汇总已保存到: {traces_summary_file}")
        
        # 总结
        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"样本ID: {sample_id}")
        print(f"会话数: {len(session_chunks)}")
        print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"处理问题数: {len(qa_results)}")
        print(f"研究轨迹文件数: {len(all_research_traces)}")
        print(f"结果保存到: {sample_results_dir}")
        print(f"  - QA结果: qa_results.json")
        print(f"  - 记忆状态: memory_state.json")
        print(f"  - 研究轨迹汇总: all_research_traces.json")
        print(f"  - 单个研究轨迹: research_trace_q*.json")
        
        return qa_results
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []


# ========== 主函数 ==========

def main():
    import argparse

    load_local_env()

    parser = argparse.ArgumentParser(description="MemPro 框架 + LoCoMo 数据集测试")
    parser.add_argument("--data", type=str, default="/path/to/locomo/dataset.json", 
                        help="LoCoMo 数据集路径")
    parser.add_argument("--outdir", type=str, default="./results/locomo",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本索引（不包含），None表示处理所有样本")
    
    # Memory Generator 配置
    parser.add_argument("--memory-api-key", type=str, default=get_env_or_default("MEMORY_API_KEY", get_env_or_default("OPENAI_API_KEY", "empty")), help="Memory 模型 API Key")
    parser.add_argument("--memory-base-url", type=str, default=get_env_or_default("MEMORY_BASE_URL", get_env_or_default("OPENAI_BASE_URL", "https://api.openai.com/v1")), help="Memory 模型 Base URL")
    parser.add_argument("--memory-model", type=str, default=get_env_or_default("MEMORY_MODEL", get_env_or_default("OPENAI_MODEL", "gpt-4o-mini")), help="Memory 模型名称")
    parser.add_argument("--memory-api-type", type=str, default=get_env_or_default("MEMORY_API_TYPE", get_env_or_default("OPENAI_API_TYPE", "openai")), choices=["openai", "vllm"], help="Memory 模型 API 类型")
    
    # Research Generator 配置
    parser.add_argument("--research-api-key", type=str, default=get_env_or_default("RESEARCH_API_KEY", get_env_or_default("OPENAI_API_KEY", "empty")), help="Research 模型 API Key")
    parser.add_argument("--research-base-url", type=str, default=get_env_or_default("RESEARCH_BASE_URL", get_env_or_default("OPENAI_BASE_URL", "https://api.openai.com/v1")), help="Research 模型 Base URL")
    parser.add_argument("--research-model", type=str, default=get_env_or_default("RESEARCH_MODEL", get_env_or_default("OPENAI_MODEL", "gpt-4o-mini")), help="Research 模型名称")
    parser.add_argument("--research-api-type", type=str, default=get_env_or_default("RESEARCH_API_TYPE", get_env_or_default("OPENAI_API_TYPE", "openai")), choices=["openai", "vllm"], help="Research 模型 API 类型")
    parser.add_argument("--use-schema", type=bool, default=True, help="是否使用 schema")

    # Working Generator 配置
    parser.add_argument("--working-api-key", type=str, default=get_env_or_default("WORKING_API_KEY", get_env_or_default("OPENAI_API_KEY", "empty")), help="Working 模型 API Key")
    parser.add_argument("--working-base-url", type=str, default=get_env_or_default("WORKING_BASE_URL", get_env_or_default("OPENAI_BASE_URL", "https://api.openai.com/v1")), help="Working 模型 Base URL")
    parser.add_argument("--working-model", type=str, default=get_env_or_default("WORKING_MODEL", get_env_or_default("OPENAI_MODEL", "gpt-4o-mini")), help="Working 模型名称")
    parser.add_argument("--working-api-type", type=str, default=get_env_or_default("WORKING_API_TYPE", get_env_or_default("OPENAI_API_TYPE", "openai")), choices=["openai", "vllm"], help="Working 模型 API 类型")
    parser.add_argument("--judge-api-key", type=str, default=get_env_or_default("JUDGE_API_KEY", get_env_or_default("OPENAI_API_KEY", "empty")), help="Judge 模型 API Key")
    parser.add_argument("--judge-base-url", type=str, default=get_env_or_default("JUDGE_BASE_URL", get_env_or_default("OPENAI_BASE_URL", "https://api.openai.com/v1")), help="Judge 模型 Base URL")
    parser.add_argument("--judge-model", type=str, default=get_env_or_default("JUDGE_MODEL", get_env_or_default("OPENAI_MODEL", "gpt-4o-mini")), help="Judge 模型名称")
    parser.add_argument("--question-workers", type=int, default=32, help="单个样本内问题并行 worker 数")
    parser.add_argument("--conv-id", type=str, default=None, help="只运行指定 sample_id / conv-id")
    parser.add_argument("--question-index", type=int, default=None, help="只运行指定 question 索引（1-based）")
    parser.add_argument("--rebuild-memory", action="store_true", help="重建 memory/pages/indexes，但保留样本目录中的其他结果文件")
    parser.add_argument("--force-rerun", action="store_true", help="强制清空样本目录并重建记忆与索引")

    args = parser.parse_args()
    
    print("=" * 60)
    print("MemPro 框架 + LoCoMo 数据集测试")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'} (共 {args.end_idx - args.start_idx if args.end_idx else '全部'} 个样本)")
    if args.conv_id:
        print(f"仅运行 conv_id: {args.conv_id}")
    if args.question_index is not None:
        print(f"仅运行 question_index: {args.question_index}")
    print(f"rebuild_memory: {args.rebuild_memory}")
    print("=" * 60)
    
    # 加载数据
    samples = load_locomo(args.data)
    print(f"共加载 {len(samples)} 个样本")
    
    # 重新设置结束索引（在加载数据后）
    if args.end_idx is None:
        args.end_idx = len(samples)
    
    print(f"实际处理范围: {args.start_idx} 到 {args.end_idx-1} (共 {args.end_idx - args.start_idx} 个样本)")
    
    # 验证索引范围
    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"错误: 开始样本索引 {args.start_idx} 超出范围 (总样本数: {len(samples)})")
        return
    
    if args.end_idx > len(samples):
        print(f"警告: 结束样本索引 {args.end_idx} 超出范围，调整为 {len(samples)}")
        args.end_idx = len(samples)
    
    if args.start_idx >= args.end_idx:
        print(f"错误: 开始索引 {args.start_idx} 必须小于结束索引 {args.end_idx}")
        return
    
    # 顺序处理每个样本
    sample_indices = list(range(args.start_idx, args.end_idx))
    if args.conv_id:
        sample_indices = [
            idx for idx in sample_indices
            if str(samples[idx].get("sample_id", f"conv-{idx}")) == args.conv_id
        ]
        if not sample_indices:
            print(f"错误: 在索引范围 {args.start_idx}:{args.end_idx} 内没有找到 conv_id={args.conv_id}")
            return
    
    print(f"将顺序处理 {len(sample_indices)} 个样本...")
    
    all_results = []
    
    # 顺序处理每个样本
    for sample_idx in tqdm(sample_indices, desc="处理样本"):
        sample = samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"开始处理样本 {sample_idx}/{len(samples)-1} (范围: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            results = process_sample(
                sample, 
                sample_idx, 
                args.outdir,
                args.memory_api_key,
                args.memory_base_url,
                args.memory_model,
                args.research_api_key,
                args.research_base_url,
                args.research_model,
                args.working_api_key,
                args.working_base_url,
                args.working_model,
                args.judge_api_key,
                args.judge_base_url,
                args.judge_model,
                args.use_schema,
                args.memory_api_type,
                args.research_api_type,
                args.working_api_type,
                args.question_workers,
                args.force_rerun,
                args.rebuild_memory,
                [args.question_index] if args.question_index is not None else None,
            )
            print(f"[OK] 样本 {sample_idx} 处理完成")
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] 样本 {sample_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算并保存批量统计；逐 conversation 的 qa_results.json 已包含原始结果。
    if all_results:
        # 计算指标
        print(f"\n{'='*60}")
        print("开始计算指标...")
        print(f"{'='*60}")
        
        # 计算 summary_answer 的指标
        pred_key = "summary_answer"
        pred_field = "answer"
        
        print(f"\n# LoCoMo Metrics for pred_key='{pred_key}', pred_field='{pred_field}'")
        summary, type_summary, details = compute_metrics_by_category(all_results, pred_key=pred_key, pred_field=pred_field)
        
        # 打印统计信息
        print(f"\n按类别统计:")
        for r in summary:
            print(f"Category {r['category']} ({r['category_type']}): n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}, llm_as_judge_avg={r['llm_as_judge_avg']:.4f}")

        print(f"\n按论文类型统计:")
        for r in type_summary:
            print(f"{r['category_type']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}, llm_as_judge_avg={r['llm_as_judge_avg']:.4f}")
        
        # 计算整体平均指标
        all_f1_scores = [row["F1"] for row in details]
        all_bleu1_scores = [row["BLEU1"] for row in details]
        all_judge_scores = [row["llm_as_judge_score"] for row in details]
        all_input_tokens = [row.get("usage", {}).get("input_tokens_approx", 0) for row in details]
        all_output_tokens = [row.get("usage", {}).get("output_tokens_approx", 0) for row in details]
        all_total_tokens = [row.get("usage", {}).get("total_tokens_approx", 0) for row in details]
        all_research_input_tokens = [row.get("research_agent_usage", {}).get("input_tokens_approx", 0) for row in details]
        all_research_output_tokens = [row.get("research_agent_usage", {}).get("output_tokens_approx", 0) for row in details]
        all_research_total_tokens = [row.get("research_agent_usage", {}).get("total_tokens_approx", 0) for row in details]
        overall_f1_avg = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        overall_bleu1_avg = sum(all_bleu1_scores) / len(all_bleu1_scores) if all_bleu1_scores else 0.0
        overall_llm_as_judge_avg = sum(all_judge_scores) / len(all_judge_scores) if all_judge_scores else 0.0
        avg_input_tokens_approx = sum(all_input_tokens) / len(all_input_tokens) if all_input_tokens else 0.0
        avg_output_tokens_approx = sum(all_output_tokens) / len(all_output_tokens) if all_output_tokens else 0.0
        avg_total_tokens_approx = sum(all_total_tokens) / len(all_total_tokens) if all_total_tokens else 0.0
        avg_research_input_tokens_approx = sum(all_research_input_tokens) / len(all_research_input_tokens) if all_research_input_tokens else 0.0
        avg_research_output_tokens_approx = sum(all_research_output_tokens) / len(all_research_output_tokens) if all_research_output_tokens else 0.0
        avg_research_total_tokens_approx = sum(all_research_total_tokens) / len(all_research_total_tokens) if all_research_total_tokens else 0.0
        
        print(f"\n整体统计:")
        print(f"总问题数: {len(all_results)}")
        print(f"整体平均 F1: {overall_f1_avg:.4f}")
        print(f"整体平均 BLEU1: {overall_bleu1_avg:.4f}")
        print(f"整体平均 llm_as_judge: {overall_llm_as_judge_avg:.4f}")
        print(f"平均输入 tokens(approx): {avg_input_tokens_approx:.2f}")
        print(f"平均输出 tokens(approx): {avg_output_tokens_approx:.2f}")
        print(f"平均总 tokens(approx): {avg_total_tokens_approx:.2f}")
        print(f"平均 ResearchAgent 输入 tokens(approx): {avg_research_input_tokens_approx:.2f}")
        print(f"平均 ResearchAgent 输出 tokens(approx): {avg_research_output_tokens_approx:.2f}")
        print(f"平均 ResearchAgent 总 tokens(approx): {avg_research_total_tokens_approx:.2f}")
        
        # 保存统计信息到 JSON 文件（类似 hotpotqa_test.py）
        processed_sample_ids = sorted({row.get("sample_id") for row in all_results if row.get("sample_id")})
        processed_sample_indices = sorted({row.get("sample_index") for row in all_results if row.get("sample_index") is not None})
        statistics = {
            "total_samples": len(processed_sample_ids) if processed_sample_ids else args.end_idx - args.start_idx,
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1_avg,
            "overall_bleu1_avg": overall_bleu1_avg,
            "overall_llm_as_judge_avg": overall_llm_as_judge_avg,
            "avg_input_tokens_approx": avg_input_tokens_approx,
            "avg_output_tokens_approx": avg_output_tokens_approx,
            "avg_total_tokens_approx": avg_total_tokens_approx,
            "avg_research_input_tokens_approx": avg_research_input_tokens_approx,
            "avg_research_output_tokens_approx": avg_research_output_tokens_approx,
            "avg_research_total_tokens_approx": avg_research_total_tokens_approx,
            "by_category": summary,
            "details": details,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1,
            "processed_sample_ids": processed_sample_ids,
            "processed_sample_indices": processed_sample_indices,
        }
        
        stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"\n指标结果已保存到: {stats_file}")
    
    print(f"\n{'='*60}")
    print("[OK] 批量测试完成！")
    print(f"处理样本数: {args.end_idx - args.start_idx}")
    print(f"成功处理: {len(all_results)} 个问题")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
