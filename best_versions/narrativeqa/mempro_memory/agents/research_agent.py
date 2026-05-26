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
from datetime import datetime, timedelta

from mempro_memory.prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from mempro_memory.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from mempro_memory.generator import AbsGenerator

QUESTION_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "did", "does", "do", "has", "have", "had", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "from", "with", "about", "into", "that", "this", "these", "those", "both",
    "kind", "type", "tell", "mention", "mentioned", "make", "made", "go",
    "went", "going", "get", "got", "give", "given", "her", "his", "their",
    "they", "them", "she", "he", "it", "its", "him", "hers", "himself",
    "herself", "after", "before", "during", "since", "last", "next", "recent",
    "recently", "there", "here", "someone", "something",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    deduped = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

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
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
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
        
        # 调试：打印prompt长度
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # 粗略估算：1 token ≈ 4 字符
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
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
        plan = self._augment_plan(plan, question)

        # Execute each planned tool and collect all hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    keyword_results = self._search_by_keyword(plan.keyword_collection, top_k=8)
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
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=8)
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

        lexical_hits = self._lexical_fallback_hits(question, top_k=5)
        all_hits.extend(lexical_hits)

        # Deduplicate hits by page_id
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
        
        # 合并有 page_id 和没有 page_id 的 hits，按问题相关度排序
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = self._rank_hits_for_question(all_unique_hits, question)
        
        # 统一进行一次 integrate
        return self._integrate(sorted_hits, result, question)

    def _augment_plan(self, plan: SearchPlan, question: str) -> SearchPlan:
        """Add deterministic, question-grounded retrieval queries as a safety net."""
        tools = list(plan.tools or [])
        if "keyword" not in tools:
            tools.append("keyword")
        if "vector" not in tools:
            tools.append("vector")

        keyword_queries = list(plan.keyword_collection or [])
        vector_queries = list(plan.vector_queries or [])
        keyword_queries.extend(self._question_keyword_queries(question))
        vector_queries.append(question)

        return SearchPlan(
            info_needs=plan.info_needs or [question],
            tools=_dedupe_keep_order(tools),
            keyword_collection=_dedupe_keep_order(keyword_queries)[:12],
            vector_queries=_dedupe_keep_order(vector_queries)[:8],
            page_index=_dedupe_keep_order(plan.page_index or [])[:5],
        )

    def _question_keyword_queries(self, question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9'+-]*|\d{1,4}", question)
        content_tokens = [
            t for t in tokens
            if len(t) > 2 and t.lower().strip("'") not in QUESTION_STOPWORDS
        ]
        capitalized = [
            t for t in tokens
            if len(t) > 1 and (t[0].isupper() or t.isupper()) and t.lower() not in QUESTION_STOPWORDS
        ]

        queries = [question]
        if content_tokens:
            queries.append(" ".join(content_tokens))
        if capitalized:
            queries.append(" ".join(capitalized))

        for n in (4, 3, 2):
            for i in range(max(0, len(content_tokens) - n + 1)):
                queries.append(" ".join(content_tokens[i:i + n]))

        return _dedupe_keep_order(queries)

    def _lexical_fallback_hits(self, question: str, top_k: int = 8) -> List[Hit]:
        """Score raw pages with question terms so exact dialogue details are not missed."""
        pages = self.page_store.load()
        terms = [
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z0-9'+-]*|\d{1,4}", question)
            if len(t) > 2 and t.lower().strip("'") not in QUESTION_STOPWORDS
        ]
        if not terms:
            return []

        hits: List[Hit] = []
        for idx, page in enumerate(pages):
            text = f"{page.header}\n{page.content}"
            lower_text = text.lower()
            score = 0.0
            matched = 0
            for term in terms:
                count = lower_text.count(term)
                if count:
                    matched += 1
                    score += min(count, 3)
            min_matched = 2 if len(terms) >= 3 else 1
            if matched < min_matched:
                continue

            phrase_bonus = 0.0
            for n in (4, 3, 2):
                for i in range(max(0, len(terms) - n + 1)):
                    phrase = " ".join(terms[i:i + n])
                    if phrase in lower_text:
                        phrase_bonus += n * 1.5
            score += phrase_bonus + (matched / max(len(terms), 1))

            hits.append(Hit(
                page_id=str(idx),
                snippet=page.content,
                source="lexical",
                meta={"score": score, "matched_terms": matched}
            ))

        hits.sort(key=lambda h: h.meta.get("score", 0), reverse=True)
        return hits[:top_k]

    def _rank_hits_for_question(self, hits: List[Hit], question: str, max_hits: int = 10) -> List[Hit]:
        """Favor pages that match question entities/phrases, then trusted retrieval sources."""
        question_terms = [
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z0-9'+-]*|\d{1,4}", question)
            if len(t) > 2 and t.lower().strip("'") not in QUESTION_STOPWORDS
        ]
        source_bonus = {"page_index": 4.0, "keyword": 3.0, "lexical": 2.0, "vector": 1.0}
        ranked = []
        for hit in hits:
            text = hit.snippet.lower()
            matched = sum(1 for term in set(question_terms) if term in text)
            phrase_bonus = 0.0
            for n in (4, 3, 2):
                for i in range(max(0, len(question_terms) - n + 1)):
                    if " ".join(question_terms[i:i + n]) in text:
                        phrase_bonus += n
            raw_score = hit.meta.get("score", 0) if hit.meta else 0
            score = (
                source_bonus.get(hit.source, 0.0)
                + matched * 1.5
                + phrase_bonus
                + min(float(raw_score), 10.0) * 0.05
            )
            ranked.append((score, hit))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in ranked[:max_hits]]

    def _extract_session_date(self, text: str) -> Optional[datetime]:
        match = re.search(
            r"Dialogue Time\(available to answer questions\):.*?on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        day = int(match.group(1))
        month = MONTHS.get(match.group(2).lower())
        year = int(match.group(3))
        if not month:
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    def _format_date(self, dt: datetime) -> str:
        return f"{dt.day} {dt.strftime('%B')} {dt.year}"

    def _relative_time_note(self, text: str) -> str:
        session_date = self._extract_session_date(text)
        if not session_date:
            return ""
        lower = text.lower()
        notes: List[str] = []

        if "yesterday" in lower:
            notes.append(f"'yesterday' = {self._format_date(session_date - timedelta(days=1))}")
        if "tomorrow" in lower:
            notes.append(f"'tomorrow' = {self._format_date(session_date + timedelta(days=1))}")
        if "last week" in lower:
            notes.append(f"'last week' means the week before {self._format_date(session_date)}")
        if "this month" in lower:
            notes.append(f"'this month' means {session_date.strftime('%B')} {session_date.year}")
        if "last month" in lower:
            prev_month = session_date.month - 1 or 12
            prev_year = session_date.year - 1 if session_date.month == 1 else session_date.year
            notes.append(f"'last month' means {datetime(prev_year, prev_month, 1).strftime('%B')} {prev_year}")

        for phrase, weekday_idx in WEEKDAYS.items():
            if re.search(rf"\blast\s+{re.escape(phrase)}\b", lower):
                days_back = (session_date.weekday() - weekday_idx) % 7
                days_back = days_back or 7
                resolved = session_date - timedelta(days=days_back)
                notes.append(f"'last {phrase}' = {self._format_date(resolved)}")

        if not notes:
            return ""
        return f"Session date: {self._format_date(session_date)}. Relative time notes: " + "; ".join(_dedupe_keep_order(notes)) + "."

    def _format_hit_evidence(self, hit: Hit, idx: int) -> str:
        source_info = f"[{hit.source}]"
        if hit.page_id:
            source_info = f"[{hit.source}]({hit.page_id})"
        note = self._relative_time_note(hit.snippet)
        if note:
            return f"{idx}. {source_info} {note}\n{hit.snippet}"
        return f"{idx}. {source_info} {hit.snippet}"

    def _fallback_evidence_summary(self, question: str, hits: List[Hit], max_hits: int = 5) -> str:
        """Use raw retrieved pages only when the integrator found no direct summary."""
        if not hits:
            return ""
        lines = [f"No direct answer was found. Relevant evidence for inference about: {question}"]
        for i, hit in enumerate(hits[:max_hits], 1):
            snippet = re.sub(r"\s+", " ", hit.snippet).strip()
            if len(snippet) > 900:
                snippet = snippet[:900].rsplit(" ", 1)[0] + "..."
            source = f"page {hit.page_id}" if hit.page_id else hit.source
            lines.append(f"{i}. {source}: {snippet}")
        return "\n".join(lines)

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
            evidence_text.append(self._format_hit_evidence(hit, i))
            
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
            evidence_text.append(self._format_hit_evidence(hit, i))
            
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
            data = response.get("json") or json.loads(response["text"])
            
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
            
            content = data.get("content", "")
            if not str(content).strip():
                content = self._fallback_evidence_summary(question, hits)

            return Result(
                content=content,
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
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            
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
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])
            
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
            return ReflectionDecision(enough=False, new_request=None)
