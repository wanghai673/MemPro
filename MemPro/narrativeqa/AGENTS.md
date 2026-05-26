# AGENTS.md

This directory contains the MemPro evolution workspace for NarrativeQA. The
goal is to evolve memory-framework versions for long-document narrative
question answering through repeated edit, debug, benchmark, and analysis.

## Mission

Continuously improve NarrativeQA F1 by evolving MemPro memory construction,
retrieval, integration, and final answer behavior.

Primary objective:

- improve NarrativeQA F1 on the selected evolution split
- retrieve and integrate plot events, character relations, motivations, and
  causal chains needed for concise answers
- reduce failures caused by overly generic summaries or distractor events

Secondary objective:

- keep answers short enough for token-level F1
- prefer lower token usage when F1 is tied or nearly tied
- avoid changes that only help one story or one answer style

Diagnostic signals:

- per-example F1
- retrieved pages and memory state
- whether evidence contains the answer-bearing event
- final answer length and lexical match
- regressions on previously strong examples

The optimization target is MemPro framework behavior. Do not edit data, labels,
reference answers, or F1 computation to improve scores.

## Workspace Layout

NarrativeQA evolution artifacts live under this directory:

- `analysis_notes/`
- `registry/versions.json`
- `runs/`
- `scripts/`
- `versions/`

The final inference runtime lives outside this workspace:

- `../../best_versions/narrativeqa/mempro_memory/`

Editable version snapshots live under:

- `versions/<version_id>/memory_agent.py`
- `versions/<version_id>/research_agent.py`
- `versions/<version_id>/memory_prompts.py`
- `versions/<version_id>/research_prompts.py`
- `versions/<version_id>/working_prompts.py` when present
- `versions/<version_id>/final_summarize_prompts.py` when present

Important:

- edit only the active version directory
- do not directly edit `../../best_versions/narrativeqa/mempro_memory/` during an
  experiment
- `scripts/run_eval.py` materializes a temporary runtime under
  `runs/<version_id>/runtime_parent/`
- generated outputs belong under `runs/<version_id>/`

## Registered Versions

The public workspace starts with:

- `v0000`: initial MemPro framework snapshot
- `v_best`: best evolved NarrativeQA framework snapshot

Use `registry/versions.json` as the source of truth for parent pointers,
version roles, and metrics.

## Required Reading

At the start of a NarrativeQA evolution session, inspect:

- `../../eval/narrativeqa_test.py`
- `../../scripts/eval_narrativeqa.sh`
- `../../best_versions/narrativeqa/mempro_memory/agents/memory_agent.py`
- `../../best_versions/narrativeqa/mempro_memory/agents/research_agent.py`
- `../../best_versions/narrativeqa/mempro_memory/prompts/working_prompts.py`
- `scripts/common.py`
- `scripts/select_base.py`
- `scripts/register_version.py`
- `scripts/run_eval.py`
- `registry/versions.json`
- recent `runs/<version_id>/` traces when available

Use these files to understand document chunking, memory construction,
retrieval, integration, answer generation, and F1 scoring.

## Continuous Iteration Loop

### 1. Select A Base

Read the registry and choose a base by considering:

- average F1
- low-F1 example patterns
- broad regressions
- branch history
- compatibility with the planned change

Helper command:

```bash
python scripts/select_base.py
```

### 2. Create A Child Version

Create a candidate from the selected base:

```bash
python scripts/register_version.py \
  --base-version v_best \
  --new-version v0001 \
  --note "one coherent change direction"
```

Good single-change directions:

- preserve named characters, aliases, relationships, and motivations
- improve retrieval queries for plot events and causal chains
- make integration distinguish similar events
- foreground direct answer-bearing excerpts before long summaries
- make final answers concise noun phrases when appropriate

Avoid mixing unrelated memory-writing, retrieval, and final-answer changes in
one version unless they form one clear mechanism.

### 3. Debug Before Benchmarking

Run targeted small checks before full benchmarking:

```bash
python scripts/run_eval.py --version v0001 --split train --limit 5 -- \
  --data-dir path/to/narrativeqa/parquet_dir \
  --max-tokens 2048 \
  --num-workers 1
```

Inspect outputs under `runs/v0001/train/`:

- per-example results
- research traces
- retrieved evidence
- memory state
- final answer strings

Memory reuse rule:

- rebuild memory if `memory_agent.py` or `memory_prompts.py` changes
- reuse compatible memory only for retrieval, integration, reflection, or final
  answering changes
- never compare a candidate with memory artifacts that do not match its memory
  construction logic

### 4. Benchmark And Analyze

After targeted debug shows stable behavior, run the full evolution split.
Compare parent and child by:

- average F1
- examples improved from zero or near-zero F1
- answer-length changes
- evidence retrieval quality
- regressions on previously strong examples

Write concise notes under `analysis_notes/` when a version reveals a reusable
failure pattern or tradeoff.

## Safety Rules

- Do not use held-out labels for edit selection.
- Do not encode sample IDs, gold answers, story-specific answer lexicons, or
  reference strings.
- Do not add API keys, personal paths, private URLs, or machine-specific
  configuration to tracked files.
- Do not import code from outside this MemPro repository.
- Keep benchmark data, metrics, and evaluation scripts fixed except for
  explicitly requested harness maintenance.
