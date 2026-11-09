<div align="center">

<h1>✨ MemPro: Agentic Memory Systems as Evolvable Programs</h1>

<h5>If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-EMNLP%202026-blue.svg)](https://aclanthology.org/2026.emnlp-main.1364/)
[![DOI](https://img.shields.io/badge/DOI-10.18653%2F2026.emnlp--main.1364-blue.svg)](https://doi.org/10.18653/2026.emnlp-main.1364)
[![arXiv](https://img.shields.io/badge/arXiv-2606.00619-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2606.00619)
[![HuggingFace](https://img.shields.io/badge/🤗%20Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2606.00619)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/wanghai673/MemPro)

</div>
</div>

## 📣 Latest News

- **[Aug 21, 2026]**: 🎉 MemPro has been accepted to the **Main Conference of EMNLP 2026**!
- **[May 30, 2026]**: 📄 Our paper is now available on [arXiv](https://arxiv.org/abs/2606.00619) and [Hugging Face](https://huggingface.co/papers/2606.00619).

## 💡 Method Overview

We introduce MemPro, a framework for automatically evolving agentic memory systems. By treating the entire memory construction–retrieval (MCR) pipeline, including its prompts and code, as an evolvable program, MemPro iteratively selects promising versions, expands them through targeted editing and debugging, and evaluates new candidates within a version tree.

<p align="center">
  <img src="figs/main.png" width="100%">
</p>

## 📈 Evolution Trajectory

MemPro evolution trajectory on LoCoMo. MemPro progressively improves through prompt- and code-level updates, outperforming GEPA and GAM.

<p align="center">
  <img src="figs/evolution_trajectory.png" width="100%">
</p>

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/wanghai673/MemPro.git
cd MemPro

conda create -n mempro python=3.10 openjdk=21 -y
conda activate mempro

pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data and API

```bash
bash scripts/download_data.sh
cp .env.example .env
```

Edit `.env` with your API credentials and model:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_TYPE=openai
```

- `OPENAI_API_KEY`: your API provider's key.
- `OPENAI_BASE_URL`: the provider's OpenAI-compatible API base URL, usually ending in `/v1`.
- `OPENAI_MODEL`: the model ID available at that endpoint.
- `OPENAI_API_TYPE`: keep `openai` for OpenAI-compatible APIs.

Leave the role-specific fields blank to use the shared configuration. You can optionally set separate API keys, base URLs, and models through `MEMORY_*` (memory construction), `RESEARCH_*` (retrieval reasoning), `WORKING_*` (answer generation), and `JUDGE_*` (scoring). Run the evaluation scripts from the repository root so they can load `.env`.

### 3. Evaluation

```bash
# LoCoMo
bash scripts/eval_locomo.sh

# LongMemEval
bash scripts/eval_longmemeval.sh

# HotpotQA
bash scripts/eval_hotpotqa.sh

# NarrativeQA
bash scripts/eval_narrativeqa.sh
```

### 4. Evolution

```bash
python scripts/run_evolution.py locomo --execute
```

## 📁 Repository Structure

```
MemPro/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── figs/                       # README figures
├── best_versions/              # Best evolved runnable MemPro frameworks
│   ├── locomo/
│   ├── longmemeval/
│   ├── hotpotqa/
│   └── narrativeqa/
├── eval/                       # Benchmark evaluation drivers
│   ├── locomo_test.py
│   ├── longmemeval_test.py
│   ├── hotpotqa_test.py
│   └── narrativeqa_test.py
├── MemPro/                     # Evolution workspaces
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

## 📄 Acknowledgement

Our initial framework is adapted from [General Agentic Memory (GAM)](https://github.com/VectorSpaceLab/general-agentic-memory), and we thank the authors for their valuable open-source contribution.

## 🥰 Citation

We appreciate your citations if you find our paper relevant and useful to your research!

```bibtex
@inproceedings{liu-etal-2026-mempro,
  title={{M}em{P}ro: Agentic Memory Systems as Evolvable Programs},
  author={Liu, Qingshan and Wang, Guoqing and Wu, Wen and Huang, Jingqi and Tao, Xinqi and Song, Dejia and Zhou, Jie and He, Liang},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026},
  publisher={Association for Computational Linguistics},
  pages={31124--31143},
  url={https://aclanthology.org/2026.emnlp-main.1364/},
  doi={10.18653/2026.emnlp-main.1364}
}
```

## 📧 Contact

For questions, suggestions, or bug reports, please contact:

```
51285901015@stu.ecnu.edu.cn
```
