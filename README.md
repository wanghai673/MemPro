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

### 1. Clone

```bash
git clone https://github.com/wanghai673/MemPro.git
cd MemPro
```

### 2. Create Environment And Install Dependencies

```bash
conda create -n mempro python=3.10 -y
conda activate mempro
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the initial `mempro_memory` package from
`initial_framework/`. Evaluation scripts override it with the best evolved
runtime for each benchmark.

### 3. Download Data

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

### 4. Configure `.env`

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

Each evaluation script loads `.env`, uses the corresponding runtime under
`best_versions/`, writes outputs to `results/`, and writes logs to `logs/`.
The default worker count is `1`; increase it with environment variables only
when your machine and API quota can support parallel requests.

### LoCoMo

```bash
bash scripts/eval_locomo.sh
```

### LongMemEval

```bash
bash scripts/eval_longmemeval.sh
```

### HotpotQA

```bash
bash scripts/eval_hotpotqa.sh
HOTPOTQA_DATA=data/hotpotqa/eval_1600.json bash scripts/eval_hotpotqa.sh
HOTPOTQA_DATA=data/hotpotqa/eval_3200.json bash scripts/eval_hotpotqa.sh
```

### NarrativeQA

```bash
bash scripts/eval_narrativeqa.sh
```

## 🧬 Part 2: Evolution

The `MemPro/` directory contains benchmark-specific evolution workspaces. To
continue evolution with Codex, choose a benchmark:

```bash
python scripts/run_evolution.py hotpotqa --execute
python scripts/run_evolution.py locomo --execute
python scripts/run_evolution.py longmemeval --execute
python scripts/run_evolution.py narrativeqa --execute
```

Choose a model if needed:

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
