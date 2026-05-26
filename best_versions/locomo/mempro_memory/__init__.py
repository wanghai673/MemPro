# -*- coding: utf-8 -*-
"""
MemPro Framework

A dual-agent architecture for building long-term memory with deep research capabilities.

Key Components:
- MemoryAgent: Builds structured memory from raw messages
- ResearchAgent: Performs multi-iteration research with reflection
"""

from __future__ import annotations

# Core agents
from mempro_memory.agents import MemoryAgent, ResearchAgent

# Generators
from mempro_memory.generator import AbsGenerator, OpenAIGenerator, VLLMGenerator

# Retrievers
from mempro_memory.retriever import AbsRetriever, IndexRetriever

# 尝试导入可选检索器
try:
    from mempro_memory.retriever import BM25Retriever
except Exception:
    BM25Retriever = None  # type: ignore

try:
    from mempro_memory.retriever import DenseRetriever
except Exception:
    DenseRetriever = None  # type: ignore

# Configurations
from mempro_memory.config import (
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig
)

# Schemas
from mempro_memory.schemas import (
    MemoryState,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    EnoughDecision,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore
)

__version__ = "0.1.0"
__all__ = [
    # Core agents
    "MemoryAgent",
    "ResearchAgent",
    
    # Generators
    "AbsGenerator",
    "OpenAIGenerator",
    "VLLMGenerator",
    
    # Retrievers
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",
    
    # Configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",
    
    # Schemas
    "MemoryState",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "EnoughDecision",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
]
