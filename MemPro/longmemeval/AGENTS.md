# AGENTS.md

This directory contains the MemPro evolution workspace for LongMemEval. The
goal is to evolve framework versions that improve long-term memory question
answering while preserving a clean, reproducible version tree.

## Mission

Continuously improve LongMemEval performance by evolving MemPro memory writing,
retrieval, integration, and answer-support behavior.

Primary objective:

- improve LongMemEval judge accuracy on the selected evolution split
- preserve specific user facts, assistant facts, preferences, temporal
  relations, and knowledge updates
- reduce failures where compressed memory loses exact answer entities

Secondary objective:

- keep generated summaries compact enough for efficient retrieval
- prefer lower token usage when accuracy is tied or nearly tied
- avoid overfitting to any single category

Diagnostic signals:

- category-level judge accuracy
- whether the research summary contains the gold answer
- retrieved page ids
- memory cache contents
- research traces
- missing or stale facts after knowledge updates

The optimization target is MemPro framework behavior, not the dataset or judge.
Do not edit data, labels, judge prompts, or metric computation to improve
scores.

## Workspace Layout

LongMemEval evolution artifacts live under this directory:

- `analysis_notes/`
- `registry/versions.json`
- `runs/`
- `scripts/`
- `versions/`

The final inference runtime lives outside this workspace:

- `../../best_versions/longmemeval/mempro_memory/`

Editable version snapshots live under:

- `versions/<version_id>/memory_agent.py`
- `versions/<version_id>/research_agent.py`
- `versions/<version_id>/memory_prompts.py`
- `versions/<version_id>/research_prompts.py`
- task-specific prompt files when present

Important:

- edit only files under the active `versions/<version_id>/`
- do not directly edit `../../best_versions/longmemeval/mempro_memory/`
- `scripts/run_eval.py` materializes a temporary runtime under
  `runs/<version_id>/runtime_parent/`
- LongMemEval evaluation auto-builds missing memory cache before question-time
  research evaluation

## Registered Versions

The public workspace starts with:

- `v0000`: initial MemPro framework snapshot
- `v_best`: best evolved LongMemEval framework snapshot

Use `registry/versions.json` as the source of truth for parent pointers,
version roles, and metrics.

## Required Reading

At the start of a LongMemEval evolution session, inspect:

- `../../eval/longmemeval_test.py`
- `../../scripts/eval_longmemeval.sh`
- `../../best_versions/longmemeval/mempro_memory/agents/memory_agent.py`
- `../../best_versions/longmemeval/mempro_memory/agents/research_agent.py`
- `scripts/common.py`
- `scripts/select_base.py`
- `scripts/register_version.py`
- `scripts/run_eval.py`
- `registry/versions.json`
- recent memory caches and traces under `runs/` when available

Use these files to understand how memory is auto-built when needed, how cached
memory is loaded, how question-time retrieval works, and how the judge
determines whether the research summary contains the answer.

## Continuous Iteration Loop

### 1. Select A Base

Choose a base by considering:

- overall judge accuracy
- category-level weakness
- failures caused by stale or overwritten facts
- regressions on previously stable examples
- compatibility with the next planned change

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

- preserve exact names, dates, locations, and preferences in memory abstracts
- improve retrieval requests for temporal or updated facts
- separate older facts from corrected newer facts
- make integration retain answer-bearing evidence rather than only general
  summaries
- reduce unsupported final research conclusions

### 3. Build Or Reuse Memory

Memory construction changes require rebuilding the cache. Question-time-only
changes can reuse compatible memory cache if the run records that reuse
clearly.

For memory-building runs, use the unified evaluation entrypoint with rebuild
flags:

```bash
../../scripts/eval_longmemeval.sh
```

For research evaluation through this workspace:

```bash
python scripts/run_eval.py --version v0001 --split train --limit 5 -- \
  --data path/to/longmemeval.json \
  --memory-root path/to/prebuilt_memory_root \
  --use-bm25
```

Inspect outputs under `runs/v0001/train/` and verify that cache paths match the
candidate's memory construction behavior.

### 4. Benchmark And Analyze

After debug behavior is stable, run the full evolution split. Compare:

- judge accuracy
- category-level gains and regressions
- retrieved page ids
- missing answer entities
- stale facts after updates
- token and runtime cost if available

Record concise analysis notes for reusable failure patterns.

## Safety Rules

- Do not use held-out labels for edit selection.
- Do not encode sample IDs, gold strings, or answer lexicons.
- Do not put API keys, personal paths, private URLs, or machine-specific
  configuration in tracked files.
- Do not import code from outside this MemPro repository.
- Keep memory-cache provenance explicit when comparing versions.
