# AGENTS.md

This directory contains the MemPro evolution workspace for HotpotQA. The goal
is to evolve runnable memory-framework versions through repeated edit, debug,
benchmark, and analysis cycles.

## Mission

Continuously improve HotpotQA answer F1 by evolving MemPro prompts and agent
framework behavior.

Primary objective:

- improve HotpotQA F1 on the selected evolution split
- prefer changes that improve many low-F1 examples without broad regressions
- strengthen multi-hop evidence retrieval, bridge-entity tracking, and final
  answer extraction

Secondary objective:

- prefer lower token usage only when F1 is tied or nearly tied
- keep answer formatting concise and compatible with token-level F1

Diagnostic signals:

- per-sample F1
- missing bridge evidence
- distractor evidence selected before answer-bearing evidence
- retrieval traces
- memory summaries and page excerpts
- final answer string shape

The optimization target is the MemPro framework: agent code, prompt files,
retrieval planning behavior, evidence integration, and answer-interface logic.
Do not edit data, labels, judge logic, or metric computation to improve scores.

## Workspace Layout

HotpotQA evolution artifacts live under this directory:

- `analysis_notes/`
- `registry/versions.json`
- `runs/`
- `scripts/`
- `versions/`

The final inference runtime lives outside this workspace:

- `../../best_versions/hotpotqa/mempro_memory/`

Editable version snapshots live under:

- `versions/<version_id>/memory_agent.py`
- `versions/<version_id>/research_agent.py`
- `versions/<version_id>/memory_prompts.py`
- `versions/<version_id>/research_prompts.py`
- `versions/<version_id>/working_prompts.py` when present
- `versions/<version_id>/final_summarize_prompts.py` when present

Important:

- edit only the active version directory
- do not directly edit `../../best_versions/hotpotqa/mempro_memory/` for an
  experiment
- `scripts/run_eval.py` builds a temporary runtime in
  `runs/<version_id>/runtime_parent/` and overlays the selected version files
- do not run conflicting benchmark jobs that write to the same version output
  directory at the same time

## Registered Versions

The public workspace starts with:

- `v0000`: initial MemPro framework snapshot
- `v_best`: best evolved HotpotQA framework snapshot

Use `registry/versions.json` as the source of truth for parent pointers,
version roles, and metrics.

## Required Reading

At the start of a HotpotQA evolution session, inspect:

- `../../eval/hotpotqa_test.py`
- `../../scripts/eval_hotpotqa.sh`
- `../../best_versions/hotpotqa/mempro_memory/agents/memory_agent.py`
- `../../best_versions/hotpotqa/mempro_memory/agents/research_agent.py`
- `../../best_versions/hotpotqa/mempro_memory/prompts/working_prompts.py`
- `scripts/common.py`
- `scripts/select_base.py`
- `scripts/register_version.py`
- `scripts/run_eval.py`
- `registry/versions.json`
- recent `runs/<version_id>/` traces when available

Use these files to understand context chunking, memory construction, retrieval
tools, integration, final answer generation, and F1 scoring.

## Continuous Iteration Loop

### 1. Select A Base

Read the registry and choose a base using:

- F1 and stability
- improved low-F1 examples
- regressed examples
- compatibility with the next change direction
- branch history

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

- improve bridge-entity query formation
- preserve article titles and aliases in memory
- prioritize answer-bearing excerpts before distractors
- reduce verbose final answers
- make integration explicitly connect two supporting facts

Avoid mixing unrelated retrieval, memory-writing, and final-answer changes in
one version unless they are part of one clear mechanism.

### 3. Debug Before Benchmarking

Run targeted small checks before full benchmarking:

```bash
python scripts/run_eval.py --version v0001 --split train --limit 5 -- \
  --data path/to/hotpotqa.json \
  --max-tokens 2048 \
  --memory-workers 4
```

Inspect output under `runs/v0001/train/`:

- batch statistics and per-sample results
- research traces
- retrieved pages
- memory state
- final answer strings

Memory reuse rule:

- rebuild memory if `memory_agent.py` or `memory_prompts.py` changes
- reuse compatible memory only for retrieval, integration, reflection, or
  answer-interface changes
- never compare a candidate with memory artifacts that do not match its memory
  construction logic

### 4. Benchmark And Analyze

Run the full evolution split only after targeted debug suggests a stable
effect. Compare parent and child by:

- average F1
- low-F1 improvements
- regressions on previously strong examples
- evidence trace quality
- answer string precision

Record concise analysis notes when a version teaches a reusable lesson.

## Safety Rules

- Do not use held-out labels for edit selection.
- Do not encode sample IDs, gold answers, title lists, or answer lexicons.
- Do not add private paths, API keys, or machine-specific configuration.
- Do not import code from outside this MemPro repository.
- Keep benchmark data, metrics, and evaluation scripts fixed except for
  explicitly requested harness maintenance.
