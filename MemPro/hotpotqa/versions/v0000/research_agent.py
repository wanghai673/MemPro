# research_agent.py
# -*- coding: utf-8 -*-
"""
ResearchAgent Module

This module defines the ResearchAgent for the MemPro (MemPro) framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import json

from mempro_memory.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from mempro_memory.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from mempro_memory.generator import AbsGenerator

class ResearchAgent:
    RETRIEVAL_TOP_K = 4
    MAX_LLM_RETRIES = 5
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        max_iters: int = 3,
        dir_path: Optional[str] = None,  # 新增：文件系统存储路径
        system_prompts: Optional[Dict[str, str]] = None,  # 新增：system prompts字典
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        self._last_planning_prompt = ""
        self._last_search_trace: Dict[str, Any] = {}
        self._last_integration_prompt = ""
        self._last_reflection_trace: Dict[str, Any] = {}
        
        # 初始化 system_prompts，默认值为空字符串
        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # 合并用户提供的 prompts 和默认值
            self.system_prompts = {**default_system_prompts, **system_prompts}

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                # 调用 retriever 的 build 方法，传递 page_store
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        # 在开始研究前，确保检索器索引是最新的
        self._update_retrievers()
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "planner_prompt": self._last_planning_prompt,
                "search_trace": self._last_search_trace,
                "integrate_prompt": self._last_integration_prompt,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
                "reflection_trace": self._last_reflection_trace,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _update_retrievers(self):
        """确保检索器索引是最新的"""
        # 检查是否有新的页面需要更新索引
        current_page_count = len(self.page_store.load())
        
        # 如果页面数量发生变化，更新所有检索器索引
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"检测到页面数量变化 ({self._last_page_count} -> {current_page_count})，更新检索器索引...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        # 更新页面计数
        self._last_page_count = current_page_count

    # ---- Internal ----
    def _build_retry_prompt(self, base_prompt: str, stage_name: str, attempt: int, error_message: str) -> str:
        return (
            f"{base_prompt}\n\n"
            f"FORMAT RETRY NOTICE ({stage_name}, attempt {attempt}/{self.MAX_LLM_RETRIES}): "
            f"Your previous response could not be accepted because: {error_message}. "
            "Return exactly one valid JSON object that satisfies the required schema. "
            "Do not add commentary, Markdown, code fences, or extra text outside the JSON object."
        )

    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        data = response.get("json")
        if data is not None:
            if not isinstance(data, dict):
                raise ValueError("structured response is not a JSON object")
            return data

        text = (response.get("text") or "").strip()
        if not text:
            raise ValueError("empty text response")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("no JSON object found in text response")

        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("parsed response is not a JSON object")
        return data

    def _validate_planning_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        required_keys = ["info_needs", "tools", "keyword_collection", "vector_queries", "page_index"]
        for key in required_keys:
            if key not in data:
                return False, f"missing key: {key}"

        list_keys = ["info_needs", "tools", "keyword_collection", "vector_queries", "page_index"]
        for key in list_keys:
            if not isinstance(data.get(key), list):
                return False, f"{key} must be a list"

        allowed_tools = {"keyword", "vector", "page_index"}
        if any(tool not in allowed_tools for tool in data.get("tools", [])):
            return False, "tools contains unsupported values"

        if not any(data.get(key) for key in required_keys):
            return False, "planning output is fully empty"

        return True, ""

    def _validate_integrate_data(self, data: Dict[str, Any], require_non_empty_content: bool) -> Tuple[bool, str]:
        if "content" not in data:
            return False, "missing key: content"
        if "sources" not in data:
            return False, "missing key: sources"
        if not isinstance(data.get("content"), str):
            return False, "content must be a string"
        if not isinstance(data.get("sources"), list):
            return False, "sources must be a list"
        if require_non_empty_content and not data.get("content", "").strip():
            return False, "content is empty despite having retrieval evidence"
        return True, ""

    def _validate_info_check_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        if "enough" not in data:
            return False, "missing key: enough"
        if not isinstance(data.get("enough"), bool):
            return False, "enough must be a boolean"
        return True, ""

    def _validate_generate_requests_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        if "new_requests" not in data:
            return False, "missing key: new_requests"
        if not isinstance(data.get("new_requests"), list):
            return False, "new_requests must be a list"
        if any(not isinstance(item, str) for item in data.get("new_requests", [])):
            return False, "new_requests must contain only strings"
        return True, ""

    def _generate_structured_with_retry(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        validator: Callable[[Dict[str, Any]], Tuple[bool, str]],
        stage_name: str,
    ) -> Dict[str, Any]:
        current_prompt = prompt
        last_error = "unknown error"

        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                response = self.generator.generate_single(prompt=current_prompt, schema=schema)
                data = self._parse_json_response(response)
                is_valid, error_message = validator(data)
                if is_valid:
                    return data
                last_error = error_message
            except Exception as e:
                last_error = str(e)

            print(f"[WARN] {stage_name} attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {last_error}")
            if attempt < self.MAX_LLM_RETRIES:
                current_prompt = self._build_retry_prompt(prompt, stage_name, attempt + 1, last_error)

        raise ValueError(f"{stage_name} failed after {self.MAX_LLM_RETRIES} attempts: {last_error}")

    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        self._last_planning_prompt = prompt
        
        # 调试：打印prompt长度
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # 粗略估算：1 token ≈ 4 字符
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            data = self._generate_structured_with_retry(
                prompt=prompt,
                schema=PLANNING_SCHEMA,
                validator=self._validate_planning_data,
                stage_name="planning",
            )
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        all_hits: List[Hit] = []
        tool_hits: Dict[str, List[Dict[str, Any]]] = {
            "keyword": [],
            "vector": [],
            "page_index": [],
        }

        # Always execute all core retrieval tools, regardless of planner output
        for tool in ["keyword", "vector", "page_index"]:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=self.RETRIEVAL_TOP_K)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    tool_hits["keyword"].extend([h.model_dump() for h in hits])
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    # 对每个向量查询都进行独立的搜索，然后在retriever层面聚合得分
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=self.RETRIEVAL_TOP_K)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    tool_hits["vector"].extend([h.model_dump() for h in hits])
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index[:self.RETRIEVAL_TOP_K])
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    tool_hits["page_index"].extend([h.model_dump() for h in hits])
                    all_hits.extend(hits)

        # Deduplicate hits by page_id
        if not all_hits:
            self._last_search_trace = {
                "tool_hits": tool_hits,
                "tool_page_ids": {k: [] for k in tool_hits},
                "all_page_ids": [],
                "dedup_page_ids": [],
                "sorted_hits": [],
            }
            self._last_integration_prompt = ""
            return result
        
        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.page_id:
                # 如果这个 page_id 还没出现过，或者当前 hit 的得分更高（如果有的话），则更新
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # 如果已有该 page_id 的 hit，比较得分（如果有的话），保留得分更高的
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # 没有 page_id 的 hits 也保留
                hits_without_id.append(hit)
        
        # 合并有 page_id 和没有 page_id 的 hits，按得分排序
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        self._last_search_trace = {
            "tool_hits": tool_hits,
            "tool_page_ids": {
                name: [h.get("page_id") for h in hits if h.get("page_id") is not None]
                for name, hits in tool_hits.items()
            },
            "all_page_ids": [hit.page_id for hit in all_hits if hit.page_id is not None],
            "dedup_page_ids": [hit.page_id for hit in sorted_hits if hit.page_id is not None],
            "sorted_hits": [hit.model_dump() for hit in sorted_hits],
        }
        
        # 统一进行一次 integrate，最多保留前 5 个候选页
        # sorted_hits = sorted_hits[:5]
        self._last_search_trace["dedup_page_ids"] = [hit.page_id for hit in sorted_hits if hit.page_id is not None]
        self._last_search_trace["sorted_hits"] = [hit.model_dump() for hit in sorted_hits]
        return self._integrate(sorted_hits, result, question)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: List[Hit] = []

        # Always execute all core retrieval tools, regardless of planner output
        for tool in ["keyword", "vector", "page_index"]:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=self.RETRIEVAL_TOP_K)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    # 对每个向量查询都进行独立的搜索，然后在retriever层面聚合得分
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=self.RETRIEVAL_TOP_K)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index[:self.RETRIEVAL_TOP_K])
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result
        
        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.page_id:
                # 如果这个 page_id 还没出现过，或者当前 hit 的得分更高（如果有的话），则更新
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # 如果已有该 page_id 的 hit，比较得分（如果有的话），保留得分更高的
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # 没有 page_id 的 hits 也保留
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        # 按得分排序（如果有的话），然后格式化
        # 合并有 page_id 和没有 page_id 的 hits
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        for i, hit in enumerate(sorted_hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            # Collect unique sources
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        self._last_integration_prompt = prompt

        try:
            data = self._generate_structured_with_retry(
                prompt=prompt,
                schema=INTEGRATE_SCHEMA,
                validator=lambda data: self._validate_integrate_data(data, require_non_empty_content=bool(hits)),
                stage_name="integration",
            )
            
            # 处理 sources：确保是字符串列表（如果LLM返回的是整数，转换为字符串）
            llm_sources = data.get("sources", sources)
            if llm_sources:
                # 将整数或混合类型转换为字符串列表
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever 返回 List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                # IndexRetriever 现在期望 List[str]，将 page_index 转换为逗号分隔的字符串
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=min(len(page_index), self.RETRIEVAL_TOP_K))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        # fallback: 直接通过 page_store 获取页面
        out: List[Hit] = []
        for idx in page_index[:self.RETRIEVAL_TOP_K]:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # 包装成 List[List[Hit]] 格式
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        
        try:
            system_prompt = self.system_prompts.get("reflection")
            
            # 调试：打印reflection prompt长度
            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            print(f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)")
            
            # Step 1: Check for completeness of information
            template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            self._last_reflection_trace = {
                "check_prompt": check_prompt,
                "generate_prompt": "",
            }
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_data = self._generate_structured_with_retry(
                prompt=check_prompt,
                schema=INFO_CHECK_SCHEMA,
                validator=self._validate_info_check_data,
                stage_name="reflection_check",
            )
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate a list of new requests
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            self._last_reflection_trace = {
                "check_prompt": check_prompt,
                "generate_prompt": generate_prompt,
            }
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_data = self._generate_structured_with_retry(
                prompt=generate_prompt,
                schema=GENERATE_REQUESTS_SCHEMA,
                validator=self._validate_generate_requests_data,
                stage_name="reflection_generate",
            )
            
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            self._last_reflection_trace = {
                "check_prompt": check_prompt if 'check_prompt' in locals() else "",
                "generate_prompt": generate_prompt if 'generate_prompt' in locals() else "",
                "error": str(e),
            }
            return ReflectionDecision(enough=False, new_request=None)
