# AGENTS.md

This directory contains the MemPro evolution workspace for LoCoMo. The goal is
not to make a one-off prompt edit. An evolution agent should repeatedly select
a base version, create a child, debug representative failures, benchmark the
candidate, analyze regressions, and keep the version tree usable for later
work.

## Mission

Continuously improve LoCoMo performance by evolving MemPro memory prompts and
agent framework behavior.

Primary objective:

- improve held-out LoCoMo accuracy after selecting versions only from the
  evolution split
- prefer changes that fix recurring failures across Single Hop, Multi Hop,
  Temporal, and Open Domain questions
- preserve exact entities, dates, user preferences, and session-specific
  evidence when those details are answer-bearing

Secondary objective:

- prefer lower token usage when accuracy is tied or nearly tied
- avoid changes that improve one category by causing broad regressions in
  previously stable categories

Diagnostic signals:

- category-level accuracy
- low-scoring question traces
- retrieved pages and memory abstracts
- final assembled context
- judge labels and reasons
- token usage when available

The optimization target is the MemPro framework itself: `memory_agent.py`,
`research_agent.py`, memory prompts, research prompts, and task-specific final
answer prompts. Do not edit data, labels, judge prompts, or evaluation metrics
to improve scores.

## Workspace Layout

LoCoMo evolution artifacts live under this directory:

- `analysis_notes/`
- `registry/versions.json`
- `runs/`
- `scripts/`
- `versions/`

The final inference runtime lives outside this workspace:

- `../../best_versions/locomo/mempro_memory/`

Editable version snapshots live under:

- `versions/<version_id>/memory_agent.py`
- `versions/<version_id>/research_agent.py`
- `versions/<version_id>/memory_prompts.py`
- `versions/<version_id>/research_prompts.py`
- `versions/<version_id>/final_summarize_prompts.py` when present
- `versions/<version_id>/working_prompts.py` when present

Important:

- edit only files under the active `versions/<version_id>/` directory
- do not directly edit `../../best_versions/locomo/mempro_memory/` during an
  experiment
- `scripts/run_eval.py` materializes a temporary runtime under
  `runs/<version_id>/runtime_parent/` and overlays the selected version files
- generated outputs belong under `runs/<version_id>/`

## Registered Versions

The public workspace starts with:

- `v0000`: initial MemPro framework snapshot
- `v_best`: best evolved LoCoMo framework snapshot

Use the registry as the source of truth for parent pointers, version roles,
and recorded metrics.

## Required Reading

At the start of a LoCoMo evolution session, inspect:

- `../../eval/locomo_test.py`
- `../../scripts/eval_locomo.sh`
- `../../best_versions/locomo/mempro_memory/agents/memory_agent.py`
- `../../best_versions/locomo/mempro_memory/agents/research_agent.py`
- `scripts/common.py`
- `scripts/select_base.py`
- `scripts/register_version.py`
- `scripts/run_eval.py`
- `registry/versions.json`
- at least one recent `runs/<version_id>/` trace if available

Use these files to understand how LoCoMo conversations are segmented, how
session time is represented, how memory is built, how retrieval and integration
run, and how final answers are judged.

## Continuous Iteration Loop

### 1. Select A Base

Read `registry/versions.json` and choose the most promising base. Consider:

- overall score
- category-level weaknesses
- branch history
- whether prior descendants regressed
- whether the next proposed change is compatible with the base

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

Use one coherent direction per child. Good directions include:

- improve temporal evidence grounding
- preserve exact session dates and relative-time expressions
- refine retrieval planning for multi-hop user facts
- make integration keep conflicting evidence separate
- make final answers shorter and more directly aligned with the question

Avoid mixing unrelated changes in the same version.

### 3. Debug Representative Failures

Run small targeted checks before a full train benchmark:

```bash
python scripts/run_eval.py --version v0001 --split train --limit 5 -- \
  --data path/to/locomo.json \
  --question-workers 8 \
  --force-rerun
```

Inspect generated files under `runs/v0001/train/`, especially question results,
research traces, pages, memory state, and batch statistics when present.

Memory reuse rule:

- if the candidate changes memory construction, rebuild memory
- if it only changes retrieval, integration, reflection, or final answering,
  reuse compatible generated memory when the evaluation harness supports it
- never report a candidate using memory artifacts inconsistent with its memory
  construction logic

### 4. Benchmark And Analyze

After debug behavior is stable, run the full intended evolution split. Compare
the candidate against its parent by:

- aggregate accuracy
- category-level accuracy
- improved and regressed examples
- trace-level failure causes
- cost or runtime changes if available

Write concise notes under `analysis_notes/` when useful and update registry
metrics only from actual runs.

## Safety Rules

- Do not use held-out labels to design edits.
- Do not encode sample IDs, gold strings, answer lexicons, or benchmark-specific
  shortcuts.
- Do not put API keys, personal paths, private URLs, or machine-specific
  configuration in tracked files.
- Do not import code from outside this MemPro repository.
- Do not rename public commands unless README and scripts are updated together.
