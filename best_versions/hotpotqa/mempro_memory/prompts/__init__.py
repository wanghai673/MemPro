# -*- coding: utf-8 -*-
"""
Prompts Module

This module contains all prompt templates and prompt-related tools for the MemPro (MemPro) framework.
Prompts are used to guide the behavior and responses of different agents (e.g., MemoryAgent, ResearchAgent) for various tasks, such as memory management and research reasoning.

Available Prompts:
- memory_prompts: Templates for memory management and updating.
- research_prompts: Templates for research, reasoning, and scientific inquiry.
"""
from .memory_prompts import MemoryAgent_PROMPT
from .research_prompts import Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from .final_summarize_prompts import (
    make_final_summarize_prompt,
    make_final_summarize_prompt_category3,
    build_final_summarize_prompt,
)

try:
    from .final_summarize_prompts import (
        FINAL_ANSWER_VALIDATOR_SCHEMA,
        build_final_answer_validator_prompt,
    )
except ImportError:
    FINAL_ANSWER_VALIDATOR_SCHEMA = None
    build_final_answer_validator_prompt = None

try:
    from .final_summarize_prompts import (
        FINAL_SLOT_ROUTER_SCHEMA,
        FINAL_SLOT_ANSWER_SCHEMA,
        build_final_slot_router_prompt,
        build_slot_routed_final_prompt,
    )
except ImportError:
    FINAL_SLOT_ROUTER_SCHEMA = None
    FINAL_SLOT_ANSWER_SCHEMA = None
    build_final_slot_router_prompt = None
    build_slot_routed_final_prompt = None

__all__ = [
    "MemoryAgent_PROMPT",
    "Planning_PROMPT",
    "Integrate_PROMPT",
    "InfoCheck_PROMPT",
    "GenerateRequests_PROMPT",
    "make_final_summarize_prompt",
    "make_final_summarize_prompt_category3",
    "build_final_summarize_prompt",
    "FINAL_ANSWER_VALIDATOR_SCHEMA",
    "build_final_answer_validator_prompt",
    "FINAL_SLOT_ROUTER_SCHEMA",
    "FINAL_SLOT_ANSWER_SCHEMA",
    "build_final_slot_router_prompt",
    "build_slot_routed_final_prompt",
]
