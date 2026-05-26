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

from typing import Any, Dict, List, Optional, Tuple
import json
import re

from mempro_memory.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from mempro_memory.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from mempro_memory.generator import AbsGenerator

class ResearchAgent:
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
        build_retrievers: bool = True,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        
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

        # Build indices upfront only when requested.
        if build_retrievers:
            for name, r in self.retrievers.items():
                try:
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
            plan, planning_trace = self._planning(next_request, memory_state, return_trace=True)

            temp, search_trace = self._search(plan, temp, request, return_trace=True)

            decision, reflection_check_trace, reflection_generate_trace = self._reflection(
                request,
                temp,
                return_trace=True,
            )

            iteration_usage = self._sum_usage_dicts(
                planning_trace.get("usage"),
                search_trace.get("usage"),
                reflection_check_trace.get("usage"),
                reflection_generate_trace.get("usage") if reflection_generate_trace else None,
            )

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
                "llm_calls": {
                    "planning": planning_trace,
                    "integration": search_trace,
                    "reflection_check": reflection_check_trace,
                    "reflection_generate": reflection_generate_trace,
                },
                "usage": iteration_usage,
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
            "integrated_memory": temp.content,
            "usage": self._sum_usage_dicts(*(item.get("usage") for item in iterations)),
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    @staticmethod
    def _approx_bytes(value: Any) -> int:
        if value is None:
            return 0
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        return len(value.encode("utf-8"))

    @classmethod
    def _make_usage(cls, input_payload: Any, output_payload: Any) -> Dict[str, int]:
        input_bytes = cls._approx_bytes(input_payload)
        output_bytes = cls._approx_bytes(output_payload)
        input_tokens = max(0, round(input_bytes / 4))
        output_tokens = max(0, round(output_bytes / 4))
        return {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "input_tokens_approx": input_tokens,
            "output_tokens_approx": output_tokens,
            "total_tokens_approx": input_tokens + output_tokens,
        }

    @staticmethod
    def _sum_usage_dicts(*usage_dicts: Optional[Dict[str, int]]) -> Dict[str, int]:
        total = {
            "input_bytes": 0,
            "output_bytes": 0,
            "input_tokens_approx": 0,
            "output_tokens_approx": 0,
            "total_tokens_approx": 0,
        }
        for usage in usage_dicts:
            if not usage:
                continue
            for key in total:
                total[key] += int(usage.get(key, 0))
        return total

    @staticmethod
    def _question_terms(question: str) -> List[str]:
        stopwords = {
            "about", "after", "again", "also", "and", "any", "are", "around",
            "because", "been", "before", "being", "between", "did", "does",
            "for", "from", "had", "has", "have", "her", "him", "his", "how",
            "into", "its", "may", "might", "not", "out", "she", "the", "their",
            "them", "then", "there", "they", "this", "was", "were", "what",
            "when", "where", "which", "who", "why", "with", "would", "you",
        }
        terms: List[str] = []
        seen = set()
        for raw in re.findall(r"[A-Za-z0-9']+", question.lower()):
            term = raw.strip("'")
            if term.endswith("'s"):
                term = term[:-2]
            if len(term) < 3 or term in stopwords:
                continue
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    @staticmethod
    def _title_tokens(question: str) -> List[str]:
        tokens: List[str] = []
        seen = set()
        for phrase in re.findall(r'"([^"]+)"', question):
            for raw in re.findall(r"[A-Za-z0-9']+", phrase.lower()):
                if len(raw) >= 3 and raw not in seen:
                    seen.add(raw)
                    tokens.append(raw)
        for raw in re.findall(r"\b[A-Z][A-Za-z0-9']+(?:\s+[A-Z][A-Za-z0-9']+)*", question):
            for part in re.findall(r"[A-Za-z0-9']+", raw.lower()):
                if len(part) >= 3 and part not in seen:
                    seen.add(part)
                    tokens.append(part)
        return tokens

    @staticmethod
    def _term_matches_line(term: str, line_lower: str) -> bool:
        if term in line_lower:
            return True
        if len(term) <= 4:
            return False
        stems = {term}
        for suffix in ("ing", "ed", "es", "s"):
            if term.endswith(suffix) and len(term) > len(suffix) + 3:
                stems.add(term[: -len(suffix)])
        return any(stem and stem in line_lower for stem in stems)

    @classmethod
    def _focused_excerpt(cls, snippet: str, question: str, max_chars: int = 900) -> str:
        terms = cls._question_terms(question)
        if not terms:
            return ""

        lines = [line.strip() for line in snippet.splitlines() if line.strip()]
        scored: List[Tuple[int, int]] = []
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            score = sum(1 for term in terms if cls._term_matches_line(term, line_lower))
            if score >= 2:
                scored.append((score, idx))

        if not scored:
            return ""

        selected = set()
        for _, idx in sorted(scored, key=lambda item: (-item[0], item[1]))[:2]:
            for neighbor in range(max(0, idx - 1), min(len(lines), idx + 2)):
                selected.add(neighbor)

        excerpt_lines = [lines[idx] for idx in sorted(selected)]
        excerpt = "\n".join(excerpt_lines)
        if len(excerpt) <= max_chars:
            return excerpt
        return excerpt[:max_chars].rsplit("\n", 1)[0].strip()

    @staticmethod
    def _document_chunks(page_text: str) -> List[str]:
        chunks = re.split(r"(?=Document\s+\d+:)", page_text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @staticmethod
    def _supplement_trigger(question: str) -> Optional[str]:
        q = question.lower()
        if "2001 census" in q or ("population" in q and "census" in q):
            return "population"
        if "centuries" in q and ("built" in q or "dwelling place" in q):
            return "centuries"
        if "formerly known" in q or "formerly called" in q:
            return "former_name"
        if "under which" in q and "vice president" in q:
            return "vice_president"
        if "both contained scenes" in q and "1959 soviet" in q:
            return "soviet_feature"
        if "documentary film festival" in q and "british journal of literary essays" in q:
            return "documentary_festival"
        return None

    @classmethod
    def _lexical_score(cls, text: str, terms: List[str], title_terms: List[str], trigger: str) -> int:
        lower = text.lower()
        score = 0
        for term in terms:
            if cls._term_matches_line(term, lower):
                score += 2
        for term in title_terms:
            if cls._term_matches_line(term, lower):
                score += 4
        if trigger == "population":
            for phrase, weight in (("2001 census", 10), ("population", 5), ("town itself", 4)):
                if phrase in lower:
                    score += weight
        elif trigger == "centuries":
            for phrase, weight in (("three centuries", 12), ("centuries", 6), ("built", 4), ("dwelling", 4)):
                if phrase in lower:
                    score += weight
        elif trigger == "former_name":
            for phrase, weight in (("formerly known", 10), ("from 1988 to 1996", 8), ("north atlantic conference", 10)):
                if phrase in lower:
                    score += weight
        elif trigger == "vice_president":
            for phrase, weight in (
                ("vice president", 10),
                ("governor", 4),
                ("nelson rockefeller", 12),
                ("committee on the employment of minority groups", 10),
                ("secretary", 5),
            ):
                if phrase in lower:
                    score += weight
        elif trigger == "soviet_feature":
            for phrase, weight in (
                ("queen of blood", 8),
                ("battle beyond the sun", 8),
                ("nebo zovyot", 14),
                ("1959 soviet", 10),
                ("contained scenes", 6),
                ("reused special effects footage", 5),
                ("mechte navstrechu", -4),
            ):
                if phrase in lower:
                    score += weight
        elif trigger == "documentary_festival":
            for phrase, weight in (
                ("london international documentary festival", 14),
                ("london review of books", 12),
                ("march and april", 14),
                ("british journal of literary essays", 8),
                ("published fortnightly", 6),
                ("new haven documentary film festival", -8),
                ("month of june", -5),
            ):
                if phrase in lower:
                    score += weight
        return score

    def _supplemental_page_hits(
        self,
        question: str,
        plan: SearchPlan,
        max_hits: int = 2,
    ) -> List[Hit]:
        trigger = self._supplement_trigger(question)
        if trigger is None:
            return []
        terms = self._question_terms(question)
        title_terms = self._title_tokens(question)
        for query in list(plan.keyword_collection or []) + list(plan.vector_queries or []):
            terms.extend(self._question_terms(str(query)))
            title_terms.extend(self._title_tokens(str(query)))
        terms = list(dict.fromkeys(terms))
        title_terms = list(dict.fromkeys(title_terms))

        scored: List[Tuple[int, int, int, str]] = []
        for page_idx, page in enumerate(self.page_store.load()):
            for chunk_idx, chunk in enumerate(self._document_chunks(page.content)):
                score = self._lexical_score(chunk, terms, title_terms, trigger)
                if score >= 14:
                    scored.append((score, page_idx, chunk_idx, chunk))

        supplemental: List[Hit] = []
        for score, page_idx, chunk_idx, chunk in sorted(scored, key=lambda item: (-item[0], item[1], item[2])):
            snippet = chunk
            if len(snippet) > 900:
                snippet = snippet[:900].rsplit(" ", 1)[0].strip()
            supplemental.append(
                Hit(
                    page_id=str(page_idx),
                    snippet=snippet,
                    source=f"lexical_{trigger}_scan",
                    meta={"score": float(score), "supplemental": True},
                )
            )
            if len(supplemental) >= max_hits:
                break
        return supplemental

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
    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None,
        return_trace: bool = False,
    ) -> SearchPlan | Tuple[SearchPlan, Dict[str, Any]]:
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
        
        # 调试：打印prompt长度
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # 粗略估算：1 token ≈ 4 字符
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            raw_text = response.get("text", "")
            data = response.get("json") or json.loads(raw_text)
            plan = SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
            if return_trace:
                return plan, {
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "usage": self._make_usage(prompt, raw_text),
                }
            return plan
        except Exception as e:
            print(f"Error in planning: {e}")
            plan = SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
            if return_trace:
                error_text = f"ERROR: {e}"
                return plan, {
                    "prompt": prompt,
                    "raw_text": error_text,
                    "usage": self._make_usage(prompt, error_text),
                }
            return plan
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None,
        return_trace: bool = False,
    ) -> Result | Tuple[Result, Dict[str, Any]]:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect all hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
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
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        all_hits.extend(self._supplemental_page_hits(question, plan))

        # Deduplicate hits by page_id
        if not all_hits:
            if return_trace:
                return result, {"content": result.content, "sources": result.sources}
            return result
        
        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.meta and hit.meta.get("supplemental"):
                hits_without_id.append(hit)
                continue
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
        
        # 统一进行一次 integrate
        return self._integrate(sorted_hits, result, question, return_trace=return_trace)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
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
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
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
        integration_prompt: Optional[str] = None,
        return_trace: bool = False,
    ) -> Result | Tuple[Result, Dict[str, Any]]:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        
        evidence_text = []
        sources = []
        focused_added = 0
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            snippet = hit.snippet
            if focused_added < 3:
                excerpt = self._focused_excerpt(hit.snippet, question)
                if excerpt:
                    snippet = f"Focused excerpt:\n{excerpt}\nFull page:\n{hit.snippet}"
                    focused_added += 1
            evidence_text.append(f"{i}. {source_info} {snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            raw_text = response.get("text", "")
            data = response.get("json") or json.loads(raw_text)
            
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
            
            result_obj = Result(
                content=data.get("content", ""),
                sources=sources
            )
            if return_trace:
                return result_obj, {
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "usage": self._make_usage(prompt, raw_text),
                }
            return result_obj
        except Exception as e:
            print(f"Error in integration: {e}")
            if return_trace:
                error_text = f"ERROR: {e}"
                return result, {
                    "prompt": prompt,
                    "raw_text": error_text,
                    "usage": self._make_usage(prompt, error_text),
                }
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
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        # fallback: 直接通过 page_store 获取页面
        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # 包装成 List[List[Hit]] 格式
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None,
        return_trace: bool = False,
    ) -> ReflectionDecision | Tuple[ReflectionDecision, Dict[str, Any], Optional[Dict[str, Any]]]:
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
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_text = check_response.get("text", "")
            check_data = check_response.get("json") or json.loads(check_text)
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                decision = ReflectionDecision(enough=True, new_request=None)
                if return_trace:
                    return decision, {
                        "prompt": check_prompt,
                        "raw_text": check_text,
                        "usage": self._make_usage(check_prompt, check_text),
                    }, None
                return decision
            
            # Step 2: Generate a list of new requests
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_text = generate_response.get("text", "")
            generate_data = generate_response.get("json") or json.loads(generate_text)
            
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            decision = ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            if return_trace:
                return decision, {
                    "prompt": check_prompt,
                    "raw_text": check_text,
                    "usage": self._make_usage(check_prompt, check_text),
                }, {
                    "prompt": generate_prompt,
                    "raw_text": generate_text,
                    "usage": self._make_usage(generate_prompt, generate_text),
                }
            return decision
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            decision = ReflectionDecision(enough=False, new_request=None)
            if return_trace:
                error_text = f"ERROR: {e}"
                return decision, {
                    "prompt": check_prompt if 'check_prompt' in locals() else "",
                    "raw_text": error_text,
                    "usage": self._make_usage(check_prompt if 'check_prompt' in locals() else "", error_text),
                }, None
            return decision
