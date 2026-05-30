# 🧬 MemPro

MemPro is a failure-driven framework for evolving agent memory systems. Instead
of treating memory as only stored content, MemPro treats the full memory
construction--retrieval pipeline as an executable program that can evolve,
including memory writing, retrieval, evidence integration, context
construction, and answer generation.

Given benchmark failures, MemPro maintains a version tree of runnable memory
framework variants, selects promising parent versions, edits prompts and
framework code, debugs candidate descendants, and keeps improvements that
generalize under a fixed evaluation protocol. This repository provides the best
evolved MemPro versions for evaluation, together with benchmark-specific
evolution workspaces for continuing the framework-evolution process.

The repository has two main parts:

1. **Evaluation**: run the best evolved MemPro versions and reproduce benchmark
   performance. This is the main path most users should follow.
2. **Evolution**: inspect and continue benchmark-specific framework evolution
   with Codex and the `AGENTS.md` instructions under `MemPro/`.

## 🏗️ Project Structure

```text
MemPro/
├── best_versions/              # Best evolved runnable MemPro frameworks
│   ├── locomo/
│   ├── longmemeval/
│   ├── hotpotqa/
│   └── narrativeqa/
├── eval/             # Benchmark evaluation drivers
│   ├── locomo_test.py
│   ├── longmemeval_test.py
│   ├── hotpotqa_test.py
│   └── narrativeqa_test.py
├── MemPro/                      # Evolution workspaces
│   ├── locomo/AGENTS.md
│   ├── longmemeval/AGENTS.md
│   ├── hotpotqa/AGENTS.md
│   └── narrativeqa/AGENTS.md
├── initial_framework/          # Initial MemPro framework package
├── scripts/                    # Download, evaluation, and evolution helpers
├── download_data/              # Dataset download utilities
├── data/                       # Generated or downloaded by local setup; not tracked
├── results/                    # Evaluation outputs written by local runs; not tracked
└── logs/                       # Runtime logs written by local runs; not tracked
```

`best_versions/` records the strongest evolved framework for each benchmark.
The shell scripts under `scripts/` automatically point Python to the correct
best-version runtime, so you do not need to edit `PYTHONPATH` by hand.

`MemPro/` contains the version-tree workspaces used for evolution. Each
benchmark has its own `AGENTS.md`, registry, scripts, and version snapshots.

## 🎯 Quick Start

Follow these steps slowly the first time. The evaluation scripts expect data,
dependencies, and model credentials to be ready before running.

### 1. Clone

```bash
git clone https://github.com/wanghai673/MemPro.git
cd MemPro
```

### 2. Create A Conda Environment

```bash
conda create -n mempro python=3.10 -y
conda activate mempro
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the initial `mempro_memory` package from
`initial_framework/`. Evaluation scripts override it with the best evolved
runtime for each benchmark.

### 4. Download Data

```bash
bash scripts/download_data.sh
```

This creates benchmark files under `data/`, including:

```text
data/locomo/locomo10.json
data/longmemeval/longmemeval_s_cleaned.json
data/hotpotqa/eval_400.json
data/hotpotqa/eval_1600.json
data/hotpotqa/eval_3200.json
data/narrativeqa/*.parquet
```

LongMemEval also needs a memory cache. The LongMemEval evaluation script now
builds missing cache entries automatically before evaluation.

### 5. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` with your OpenAI-compatible endpoint:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_TYPE=openai
```

You can also set role-specific variables such as `MEMORY_MODEL`,
`RESEARCH_MODEL`, `WORKING_MODEL`, and `JUDGE_MODEL`.

Keep `.env` local because it contains credentials. The repository already
excludes it from version control.

## 🧪 Part 1: Evaluation

This is the main reproducibility path. Each script loads `.env`, points
`PYTHONPATH` to the corresponding runtime under `best_versions/`, writes
outputs to `results/`, and writes logs to `logs/`.

General model settings can be shared by all benchmarks:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_TYPE=openai
```

For benchmarks with separate memory, research, working, or judge calls, you can
override the shared settings with role-specific variables:

```bash
MEMORY_MODEL=gpt-4o-mini
RESEARCH_MODEL=gpt-4o-mini
WORKING_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
```

Common runtime knobs:

```bash
MEMPRO_NUM_WORKERS=16
MEMPRO_QUESTION_WORKERS=32
MEMPRO_EMBEDDING_MODEL=BAAI/bge-m3
MEMPRO_DENSE_DEVICES=cuda:0
```

### LoCoMo

Run the default LoCoMo evaluation:

```bash
bash scripts/eval_locomo.sh
```

Default paths:

```text
data:    data/locomo/locomo10.json
outputs: results/locomo
log:     logs/locomo_inference.log
runtime: best_versions/locomo
```

Useful overrides:

```bash
LOCOMO_DATA=data/locomo/locomo10.json \
LOCOMO_OUTDIR=results/locomo \
MEMPRO_QUESTION_WORKERS=32 \
bash scripts/eval_locomo.sh
```

You can pass evaluation-driver arguments after the script name:

```bash
bash scripts/eval_locomo.sh --start-idx 0 --end-idx 10
```

### LongMemEval

Run the default LongMemEval evaluation:

```bash
bash scripts/eval_longmemeval.sh
```

Default paths:

```text
data:         data/longmemeval/longmemeval_s_cleaned.json
memory cache: data/longmemeval/memory/_memory_cache
outputs:      results/longmemeval
log:          logs/longmemeval_inference.log
runtime:      best_versions/longmemeval
```

The script builds missing memory cache entries automatically by default. If you
already have a complete cache and want to forbid rebuilding, set
`LONGMEMEVAL_BUILD_MEMORY_IF_MISSING=0`.

Useful overrides:

```bash
LONGMEMEVAL_START_IDX=0 \
LONGMEMEVAL_END_IDX=10 \
LONGMEMEVAL_OUTDIR=results/longmemeval \
MEMORY_BUILD_WORKERS=4 \
MEMPRO_NUM_WORKERS=16 \
MEMPRO_DENSE_DEVICES=cuda:0 \
bash scripts/eval_longmemeval.sh
```

Retrieval options:

```bash
LONGMEMEVAL_USE_BM25=1 \
LONGMEMEVAL_USE_DENSE=1 \
MEMPRO_EMBEDDING_MODEL=BAAI/bge-m3 \
bash scripts/eval_longmemeval.sh
```

Cache options:

```bash
LONGMEMEVAL_BUILD_MEMORY_IF_MISSING=1 \
LONGMEMEVAL_FORCE_REBUILD_MEMORY=0 \
LONGMEMEVAL_MEMORY_ROOT=data/longmemeval/memory/_memory_cache \
bash scripts/eval_longmemeval.sh
```

### HotpotQA

Run the default HotpotQA evaluation:

```bash
bash scripts/eval_hotpotqa.sh
```

Default paths:

```text
data:    data/hotpotqa/eval_400.json
outputs: results/hotpotqa
log:     logs/hotpotqa_inference.log
runtime: best_versions/hotpotqa
```

Choose a different context-length file:

```bash
HOTPOTQA_DATA=data/hotpotqa/eval_1600.json \
bash scripts/eval_hotpotqa.sh
```

Useful overrides:

```bash
HOTPOTQA_OUTDIR=results/hotpotqa \
HOTPOTQA_NUM_WORKERS=16 \
HOTPOTQA_MEMORY_WORKERS=1 \
HOTPOTQA_MAX_TOKENS=2048 \
bash scripts/eval_hotpotqa.sh
```

### NarrativeQA

Run the default NarrativeQA evaluation:

```bash
bash scripts/eval_narrativeqa.sh
```

Default paths and range:

```text
data dir: data/narrativeqa
split:    test
range:    0..300
outputs:  results/narrativeqa
log:      logs/narrativeqa_inference.log
runtime:  best_versions/narrativeqa
```

Useful overrides:

```bash
NARRATIVEQA_START_IDX=0 \
NARRATIVEQA_END_IDX=50 \
NARRATIVEQA_SPLIT=test \
NARRATIVEQA_OUTDIR=results/narrativeqa \
MEMPRO_NUM_WORKERS=16 \
MEMPRO_DENSE_DEVICES=cuda:0 \
bash scripts/eval_narrativeqa.sh
```

## 🧬 Part 2: Evolution

The `MemPro/` directory contains benchmark-specific evolution workspaces. Each
workspace has:

- `AGENTS.md`: detailed instructions for Codex
- `versions/`: editable framework snapshots
- `registry/versions.json`: version-tree metadata
- `scripts/`: base selection, version registration, and evaluation helpers
- `runs/`: generated evolution outputs

To inspect a workspace manually:

```bash
cd MemPro/hotpotqa
python scripts/select_base.py
python scripts/register_version.py --dry-run --note "try one coherent change"
```

### Launch Codex For Evolution

The helper below selects a benchmark workspace and asks Codex to read
`AGENTS.md` before continuing evolution.

Dry run first:

```bash
python scripts/run_evolution.py hotpotqa --dry-run
```

Execute with Codex:

```bash
python scripts/run_evolution.py hotpotqa --execute
```

Choose another benchmark:

```bash
python scripts/run_evolution.py locomo --execute
python scripts/run_evolution.py longmemeval --execute
python scripts/run_evolution.py narrativeqa --execute
```

By default, the launcher uses `gpt-5.4-medium`:

```bash
python scripts/run_evolution.py hotpotqa --model gpt-5.4-medium --execute
```

The evolution workflow is intentionally separate from evaluation. Most users
only need Part 1; Part 2 is for developing new MemPro versions.

## 🙏 Acknowledgments

We thank the authors of the following datasets:

- [HotpotQA](https://github.com/hotpotqa/hotpot)
- [NarrativeQA](https://github.com/google-deepmind/narrativeqa)
- [LoCoMo](https://github.com/snap-research/locomo)
- [LongMemEval](https://github.com/xiaowu0162/longmemeval)

MemPro also builds on ideas and code from prior memory-framework research:

- [General Agentic Memory (GAM)](https://github.com/VectorSpaceLab/general-agentic-memory)
- [LightMem](https://github.com/zjunlp/LightMem)

## 📄 License

This project is released under the MIT License.
