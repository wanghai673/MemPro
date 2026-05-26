def make_prompt(summary: str, question: str) -> str:
    """Build the final HotpotQA answer prompt from research_summary."""
    return f"""You are a careful multi-hop reading assistant.
Use the given Context.
Answer with ONLY the final answer string; no extra words.

Question:
{question}

Context:
{summary}

Answer:
"""
