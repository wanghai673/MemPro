#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemPro 框架 + HotpotQA 数据集测试文件

基于 test_mempro_longbench.py，适配 HotpotQA 数据集格式。
HotpotQA 数据格式：
- context: str - 长文本上下文
- input: str - 问题
- index: int - 索引
- answers: [] - 答案列表（可能有多个正确答案）
"""

import string
import sys
import os
import re
import json
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Counter, Dict, List, Optional, Tuple
from tqdm import tqdm

from mempro_memory.utils.env import load_local_env

from mempro_memory import (
    MemoryAgent,
    ResearchAgent,
    Page,
    VLLMGenerator,
    OpenAIGeneratorConfig,
    OpenAIGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)


def _env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _shared_profile_value(suffix: str, default: str) -> str:
    backend = (_env_value("MEMPRO_MODEL_BACKEND", "api") or "api").lower()
    if backend == "local":
        value = _env_value(f"LOCAL_{suffix}")
        if value is not None:
            return value
    elif backend == "api":
        value = _env_value(f"API_{suffix}")
        if value is not None:
            return value

    openai_value = _env_value(f"OPENAI_{suffix}")
    if openai_value is not None:
        return openai_value
    return default


def _role_profile_value(role: str, suffix: str, default: str) -> str:
    return _env_value(f"{role}_{suffix}", _shared_profile_value(suffix, default)) or default


def _int_env_value(name: str, default: int) -> int:
    value = _env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ========== 数据加载 ==========

def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    """
    加载 HotpotQA JSON 数据集
    
    Args:
        json_path: 数据集 JSON 文件路径
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 提取所有属性
    data_all = [
        {
            "index": item.get("index", idx),
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": item.get("answers", []),
            "_id": f"hotpotqa-{item.get('index', idx)}"  # 生成唯一ID
        }
        for idx, item in enumerate(dataset)
    ]
    
    return data_all

# ========== 长文本切分 ==========

def build_context_chunks_for_sample(
    sample: Dict[str, Any], 
    max_tokens: int = 2000, 
    embedding_model_path: Optional[str] = None
) -> List[str]:
    """
    将 context 文本按 token 数量分割成多个会话块
    使用智能切分：优先在边界处切分
    
    Args:
        sample: 样本数据，包含 'context' 字段
        max_tokens: 每个会话块的最大 token 数量
        embedding_model_path: embedding 模型路径，如果提供则使用该模型进行精确 token 计算
    """
    context_text = sample.get("context") or ""
    
    if not context_text:
        return []
    
    # 优先尝试使用 embedding 模型进行精确的 token 切分
    if embedding_model_path:
        try:
            chunks = _split_with_embedding_model(context_text, max_tokens, embedding_model_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"Warning: Embedding model splitting failed: {e}, falling back to tiktoken")
    
    # 使用 tiktoken 进行精确的 token 切分
    try:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = tokenizer.encode(context_text, disallowed_special=())
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{context_text}"]
        
        # 智能切分：按 token 数量切分
        chunks = _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
        return chunks
        
    except ImportError:
        print("Warning: tiktoken not available, falling back to character-based splitting")
        return _fallback_char_split(context_text, max_tokens)

def _split_with_embedding_model(text: str, max_tokens: int, model_path: str) -> List[str]:
    """
    使用 embedding 模型进行精确的 token 切分
    """
    try:
        from transformers import AutoTokenizer
        
        # 使用指定的模型 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 编码文本获取 tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{text}"]
        
        # 智能切分
        chunks = _smart_split_by_tokens(text, tokens, max_tokens, tokenizer)
        return chunks
        
    except Exception as e:
        print(f"Error using embedding model: {e}")
        return []

def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    """
    按 token 数量简单切分：不进行智能边界查找，直接按 max_tokens 切分
    """
    chunks = []
    
    # 如果文本不超过最大 token 数，直接返回
    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]
    
    # 直接按照 token 索引切分
    session_id = 0
    start_idx = 0
    
    while start_idx < len(tokens):
        # 计算当前块的结束 token 索引
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # 将 tokens 解码回文本
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        if chunk_text.strip():
            chunks.append(f"[Session {session_id}]\n{chunk_text.strip()}")
            session_id += 1
        
        start_idx = end_idx
    
    return chunks

def _fallback_char_split(text: str, max_tokens: int) -> List[str]:
    """
    字符切分的 fallback 方法
    """
    # 粗略估计：1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    
    chunks = []
    current_start = 0
    session_id = 0
    
    while current_start < len(text):
        current_end = min(current_start + max_chars, len(text))
        
        # 尝试在单词边界切分
        if current_end < len(text):
            # 寻找最后一个换行符
            last_newline = text.rfind('\n', current_start, current_end)
            if last_newline > current_start:
                current_end = last_newline
            else:
                # 寻找最后一个空格
                last_space = text.rfind(' ', current_start, current_end)
                if last_space > current_start:
                    current_end = last_space
        
        chunk_text = text[current_start:current_end].strip()
        if chunk_text:
            chunks.append(f"[Session {session_id}]\n{chunk_text}")
            session_id += 1
        
        current_start = current_end
    
    return chunks

# ========== Prompt 设计 ==========

def _load_versioned_make_prompt():
    prompt_path = os.getenv("HOTPOTQA_WORKING_PROMPTS_PATH")
    if not prompt_path:
        return None
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"HOTPOTQA_WORKING_PROMPTS_PATH does not exist: {prompt_path}")
    spec = importlib.util.spec_from_file_location("hotpotqa_versioned_working_prompts", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HotpotQA working prompts from: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make_prompt_fn = getattr(module, "make_prompt", None)
    if make_prompt_fn is None:
        raise AttributeError(f"{prompt_path} must define make_prompt(summary: str, question: str)")
    return make_prompt_fn


def make_prompt(summary: str, question: str) -> str:
    """创建统一的 Prompt（开放问答格式）"""
    versioned_make_prompt = _load_versioned_make_prompt()
    if versioned_make_prompt is not None:
        return versioned_make_prompt(summary, question)
    prompt = f"""You are a careful multi-hop reading assistant. 
Use the given Context. 
Answer with ONLY the final answer string; no extra words.

Question:
{question}

Context:
{summary}

Answer:
"""
    return prompt

# ========== 答案提取和评估 ==========
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def _calculate_f1(pred_answer: str, gold_answers: List[str]) -> float:
    # 计算与每个标准答案的 F1，取最大值
    max_f1 = 0.0
    for gold_answer in gold_answers:
        max_f1 = max(max_f1, qa_f1_score(pred_answer, gold_answer))
    return max_f1


def build_memory_parallel(
    memory_agent: MemoryAgent,
    context_chunks: List[str],
    memory_workers: int = 32,
) -> None:
    """
    Build HotpotQA memory by parallelizing LLM decoration, then persisting in
    chunk order. Store writes remain sequential because the JSON-backed stores
    are not thread-safe.
    """
    if not context_chunks:
        return

    worker_count = max(1, min(memory_workers, len(context_chunks)))
    if worker_count == 1:
        for i, context_chunk in enumerate(context_chunks, 1):
            print(f"  处理上下文块 {i}/{len(context_chunks)}...")
            memory_agent.memorize(context_chunk)
        return

    initial_state = memory_agent.memory_store.load()
    decorated_items: List[Optional[Tuple[int, str, str, str, str]]] = [None] * len(context_chunks)

    def decorate_one(item: Tuple[int, str]) -> Tuple[int, str, str, str, str]:
        chunk_index, context_chunk = item
        last_error: Optional[Exception] = None
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                print(
                    f"  并行生成上下文块摘要 {chunk_index}/{len(context_chunks)}"
                    f" (attempt {attempt}/{max_attempts})..."
                )
                abstract, header, decorated_new_page = memory_agent._decorate(context_chunk, initial_state)
                return chunk_index, context_chunk, abstract, header, decorated_new_page
            except RuntimeError as exc:
                last_error = exc
                if "Empty abstract returned by generator" not in str(exc) or attempt == max_attempts:
                    raise
                wait_s = min(2.0 * attempt, 12.0)
                print(f"  [WARN] 上下文块 {chunk_index} 摘要为空，{wait_s:.1f}s 后重试...")
                time.sleep(wait_s)
        raise last_error or RuntimeError("Memory decoration failed")

    print(f"  使用 {worker_count} 个 memory worker 并行生成 {len(context_chunks)} 个摘要...")
    indexed_chunks = list(enumerate(context_chunks, 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(decorate_one, item): item[0]
            for item in indexed_chunks
        }
        for future in as_completed(future_to_index):
            chunk_index = future_to_index[future]
            decorated_items[chunk_index - 1] = future.result()
            print(f"  [OK] 上下文块摘要完成 {chunk_index}/{len(context_chunks)}")

    print("  按原始上下文块顺序写入 memory/page store...")
    for item in decorated_items:
        if item is None:
            raise RuntimeError("Missing decorated memory item")
        _, context_chunk, abstract, header, decorated_new_page = item
        memory_agent.memory_store.add(abstract)
        page = Page(header=header, content=context_chunk, meta={"decorated": decorated_new_page})
        memory_agent.page_store.add(page)


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
    max_tokens: int = 2000,
    embedding_model_path: Optional[str] = None,
    memory_workers: int = 32,
    use_schema: bool = False,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai"
):
    """
    使用 MemPro 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答
    """
    sample_id = sample.get("_id", f"sample-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:
        # 1. 构建上下文块
        context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
        print(f"上下文块数: {len(context_chunks)}")
        if context_chunks:
            print(f"第一个上下文块预览:\n{context_chunks[0][:400]}...")
        
        # 创建输出目录
        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"输出目录: {sample_results_dir}")
        
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
                max_tokens=256
            )
            memory_generator = OpenAIGenerator(memory_generator_config.__dict__)
        elif memory_api_type == "vllm":
            memory_generator_config = VLLMGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=256
            )
            memory_generator = VLLMGenerator(memory_generator_config.__dict__)
        
        print(f"[OK] Memory Generator 创建完成")
        
        # 4. 使用 MemoryAgent 构建记忆（将每个 context chunk 作为一条消息）
        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator,
        )

        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            build_memory_parallel(memory_agent, context_chunks, memory_workers=memory_workers)
        
        # 查看构建的记忆
        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        
        # 显示记忆摘要
        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        
        # 保存记忆状态
        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        
        # 5. 创建检索器（用于 ResearchAgent）
        print(f"\n步骤 3: 创建检索器（用于 ResearchAgent）")
        retrievers = {}
        
        # 索引检索器
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            # 如果索引目录已存在，先删除它（避免 "Directory not empty" 错误）
            if os.path.exists(page_index_dir):
                import shutil
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
                import shutil
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
                import shutil
                shutil.rmtree(dense_index_dir)
                print(f"[INFO] 清理已存在的 Dense 索引目录: {dense_index_dir}")
            
            dense_retriever_devices = [
                item.strip()
                for item in _env_value("DENSE_RETRIEVER_DEVICES", "cuda:0").split(",")
                if item.strip()
            ]
            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                model_name="BAAI/bge-m3",
                devices=dense_retriever_devices,
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
        
        # 6. 创建 Research Generator 和 Working Generator
        print(f"\n步骤 4: 创建 Research Generator 和 Working Generator")
        research_max_output_tokens = _int_env_value("HOTPOTQA_RESEARCH_MAX_OUTPUT_TOKENS", 2048)
        if research_api_type == "openai":
            research_generator_config = OpenAIGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=research_max_output_tokens,
                use_schema=use_schema
            )
            research_generator = OpenAIGenerator(research_generator_config.__dict__)
        elif research_api_type == "vllm":
            research_generator_config = VLLMGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=research_max_output_tokens,
                use_schema=use_schema
            )
            research_generator = VLLMGenerator(research_generator_config.__dict__)
            
        if working_api_type == "openai":
            working_generator_config = OpenAIGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=256
            )
            working_generator = OpenAIGenerator(working_generator_config.__dict__)
        elif working_api_type == "vllm":
            working_generator_config = VLLMGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=256
            )
            working_generator = VLLMGenerator(working_generator_config.__dict__)
        print(f"[OK] Research Generator 和 Working Generator 创建完成")
        
        # 7. 创建 ResearchAgent
        print(f"\n步骤 5: 创建 ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=3
        )
        print(f"[OK] ResearchAgent 创建完成")
        
        # 8. 进行问答
        print(f"\n步骤 6: 进行问答")
        
        # 提取问题信息
        question = sample.get("input", "")
        gold_answers = sample.get("answers", [])
        
        print(f"问题: {question}")
        print(f"标准答案: {gold_answers}")
        
        # 保存所有数据属性
        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "question": question,
            "answers": gold_answers,
            "gold_answers": gold_answers,  # 保留 gold_answers 以便兼容
        }

        try:
            # 使用 ResearchAgent 进行研究
            print("正在进行深度研究...")
            research_result = research_agent.research(question)
            research_summary = research_result.integrated_memory
            print(f"[OK] 研究完成！迭代次数: {len(research_result.raw_memory.get('iterations', []))}")
            print(f"研究摘要: {research_summary[:200]}...")
            
            # 保存研究轨迹
            research_trace = {
                "question": question,
                "raw_memory": research_result.raw_memory,
                "integrated_memory": research_result.integrated_memory,
                "iterations": research_result.raw_memory.get("iterations", []),
                "search_plans": research_result.raw_memory.get("search_plans", []),
                "reflections": research_result.raw_memory.get("reflections", [])
            }
            
            trace_file = os.path.join(sample_results_dir, "research_trace.json")
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(research_trace, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 研究轨迹已保存: {trace_file}")
            
            result["research_summary"] = research_summary
            result["research_trace_file"] = trace_file
            
            # 使用统一的 prompt 格式生成答案
            print("生成答案...")
            prompt = make_prompt(research_summary, question)
            response = working_generator.generate_single(prompt=prompt)
            answer_text = response.get("text", "").strip()
            
            print(f"模型响应: {answer_text[:200]}...")
            
            # 提取答案
            pred_answer = answer_text
            result["response"] = answer_text
            result["pred"] = pred_answer
            
            # 计算 F1 分数
            f1_score = _calculate_f1(pred_answer, gold_answers) if pred_answer else 0.0
            result["f1"] = f1_score
            
            print(f"预测答案: {pred_answer}")
            print(f"标准答案: {gold_answers}")
            print(f"F1 分数: {f1_score:.4f}")
            
        except Exception as e:
            print(f"[ERROR] 处理问题失败: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        # 保存结果
        results_file = os.path.join(sample_results_dir, "qa_result.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        
        # 总结
        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"样本ID: {sample_id}")
        print(f"上下文块数: {len(context_chunks)}")
        if final_state:
            print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"预测答案: {result.get('pred', 'N/A')}")
        print(f"标准答案: {gold_answers}")
        print(f"F1 分数: {result.get('f1', 0.0):.4f}")
        print(f"结果保存到: {sample_results_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "error": error_msg
        }


# ========== 主函数 ==========

def main():
    import argparse

    load_local_env()

    parser = argparse.ArgumentParser(description="MemPro 框架 + HotpotQA 数据集测试")
    parser.add_argument("--data", type=str, default="/path/to/hotpotqa/eval.json", 
                        help="HotpotQA 数据集路径")
    parser.add_argument("--outdir", type=str, default="./results/hotpotqa",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本索引（不包含），None表示处理所有样本")
    parser.add_argument("--max-tokens", type=int, default=2048, help="每个上下文块的最大 token 数量")
    parser.add_argument("--memory-workers", type=int, default=32, help="并行构建 HotpotQA memory 的 worker 数")
    parser.add_argument("--embedding-model-path", type=str, default=None, 
                        help="Embedding 模型路径，用于精确 token 计算（可选）")
    
    # Memory Generator 配置
    parser.add_argument("--memory-api-key", type=str, default=_role_profile_value("MEMORY", "API_KEY", "empty"), help="Memory 模型 API Key")
    parser.add_argument("--memory-base-url", type=str, default=_role_profile_value("MEMORY", "BASE_URL", "https://api.openai.com/v1"), help="Memory 模型 Base URL")
    parser.add_argument("--memory-model", type=str, default=_role_profile_value("MEMORY", "MODEL", "gpt-4o-mini"), help="Memory 模型名称")
    parser.add_argument("--memory-api-type", type=str, default=_role_profile_value("MEMORY", "API_TYPE", "openai"), choices=["openai", "vllm"], help="Memory 模型 API 类型")
    
    # Research Generator 配置
    parser.add_argument("--research-api-key", type=str, default=_role_profile_value("RESEARCH", "API_KEY", "empty"), help="Research 模型 API Key")
    parser.add_argument("--research-base-url", type=str, default=_role_profile_value("RESEARCH", "BASE_URL", "https://api.openai.com/v1"), help="Research 模型 Base URL")
    parser.add_argument("--research-model", type=str, default=_role_profile_value("RESEARCH", "MODEL", "gpt-4o-mini"), help="Research 模型名称")
    parser.add_argument("--research-api-type", type=str, default=_role_profile_value("RESEARCH", "API_TYPE", "openai"), choices=["openai", "vllm"], help="Research 模型 API 类型")
    parser.add_argument("--use-schema", type=bool, default=False, help="是否使用 schema")

    # Working Generator 配置
    parser.add_argument("--working-api-key", type=str, default=_role_profile_value("WORKING", "API_KEY", "empty"), help="Working 模型 API Key")
    parser.add_argument("--working-base-url", type=str, default=_role_profile_value("WORKING", "BASE_URL", "https://api.openai.com/v1"), help="Working 模型 Base URL")
    parser.add_argument("--working-model", type=str, default=_role_profile_value("WORKING", "MODEL", "gpt-4o-mini"), help="Working 模型名称")
    parser.add_argument("--working-api-type", type=str, default=_role_profile_value("WORKING", "API_TYPE", "openai"), choices=["openai", "vllm"], help="Working 模型 API 类型")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MemPro 框架 + HotpotQA 数据集测试")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'}")
    print(f"最大 token 数: {args.max_tokens}")
    print(f"memory workers: {args.memory_workers}")
    print("=" * 60)
    
    # 加载数据
    all_samples = load_hotpotqa(args.data)
    print(f"共加载 {len(all_samples)} 个样本")
    
    # 重新设置结束索引（在加载数据后）
    if args.end_idx is None:
        args.end_idx = len(all_samples)
    
    print(f"实际处理范围: {args.start_idx} 到 {args.end_idx-1} (共 {args.end_idx - args.start_idx} 个样本)")
    
    # 验证索引范围
    if args.start_idx < 0 or args.start_idx >= len(all_samples):
        print(f"错误: 开始样本索引 {args.start_idx} 超出范围 (总样本数: {len(all_samples)})")
        return
    
    if args.end_idx > len(all_samples):
        print(f"警告: 结束样本索引 {args.end_idx} 超出范围，调整为 {len(all_samples)}")
        args.end_idx = len(all_samples)
    
    if args.start_idx >= args.end_idx:
        print(f"错误: 开始索引 {args.start_idx} 必须小于结束索引 {args.end_idx}")
        return
    
    # 串行批量处理样本
    sample_indices = list(range(args.start_idx, args.end_idx))
    
    print(f"开始串行处理样本...")
    
    all_results = []
    for sample_idx in tqdm(sample_indices, desc="处理样本"):
        sample = all_samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"开始处理样本 {sample_idx}/{len(all_samples)-1} (范围: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            result = process_sample(
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
                max_tokens=args.max_tokens,
                embedding_model_path=args.embedding_model_path,
                memory_workers=args.memory_workers,
                use_schema=args.use_schema,
                memory_api_type=args.memory_api_type,
                research_api_type=args.research_api_type,
                working_api_type=args.working_api_type
            )
            print(f"[OK] 样本 {sample_idx} 处理完成")
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] 样本 {sample_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "sample_id": sample.get("_id", f"sample-{sample_idx}"),
                "error": str(e)
            })
    
    # 统计结果
    f1_scores = []
    
    for result in all_results:
        if "f1" in result:
            f1_scores.append(result["f1"])
    
    # 保存所有结果汇总
    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 批量结果汇总已保存: {summary_file}")
        
        # 计算平均 F1 分数
        if len(f1_scores) > 0:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            total_samples = args.end_idx - args.start_idx
            success_count = len(f1_scores)
            
            # 构建统计信息
            statistics = {
                "total_samples": total_samples,
                "success_count": success_count,
                "failed_count": total_samples - success_count,
                "success_rate": success_count / total_samples if total_samples > 0 else 0.0,
                "avg_f1": avg_f1,
                "f1_scores": f1_scores,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx - 1
            }
            
            # 保存统计信息到文件
            stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            print(f"[OK] 批量测试统计已保存: {stats_file}")
            
            # 打印统计信息
            print(f"\n{'='*60}")
            print("批量测试统计")
            print(f"{'='*60}")
            print(f"处理样本数: {total_samples}")
            print(f"成功回答问题数: {success_count}")
            print(f"失败问题数: {total_samples - success_count}")
            print(f"成功率: {statistics['success_rate']:.2%}")
            print(f"平均 F1 分数: {avg_f1:.4f}")
            print(f"{'='*60}")

if __name__ == "__main__":
    main()
