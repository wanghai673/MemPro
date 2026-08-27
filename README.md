<div align="center">

<h1>🧬 MemPro: Agentic Memory Systems as Evolvable Programs</h1>

<h5>If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2606.00619)
[![HuggingFace](https://img.shields.io/badge/🤗%20Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2606.00619)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/wanghai673/MemPro)

</div>
</div>

## 📣 Latest News

- **[Aug 21, 2026]**: 🎉 MemPro has been accepted to the **Main Conference of EMNLP 2026**!
- **[May 30, 2026]**: 📄 Our paper is now available on [arXiv](https://arxiv.org/abs/2606.00619) and [Hugging Face](https://huggingface.co/papers/2606.00619).

## 💡 Method Overview

MemPro treats the entire memory construction–retrieval (MCR) pipeline—including prompts and executable logic for memory construction, retrieval, evidence integration, reflection, and answer generation—as an **evolvable program**. It maintains a **version tree** of runnable memory systems, where an Evolving Agent selects promising versions, diagnoses recurring failure modes, and creates improved child versions through targeted edit–debug refinement. This tree-based evolution preserves strong historical versions while enabling continued system-level improvement.

<p align="center">
  <img src="figs/main.png" width="100%">
</p>

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/wanghai673/MemPro.git
cd MemPro

conda create -n mempro python=3.10 -y
conda activate mempro

pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data and API

```bash
bash scripts/download_data.sh
cp .env.example .env
```

Set your `OPENAI_API_KEY` and, if needed, the OpenAI-compatible endpoint and model in `.env`.
Dense retrieval runs on CPU by default. With a CUDA-enabled PyTorch build, set `MEMPRO_DENSE_DEVICES=cuda:0` to use a GPU.

### 3. Evaluation

```bash
bash scripts/eval_locomo.sh
bash scripts/eval_longmemeval.sh
bash scripts/eval_hotpotqa.sh
bash scripts/eval_narrativeqa.sh
```

### 4. Evolution

```bash
python scripts/run_evolution.py <benchmark> --execute
```

Supported benchmarks: `locomo`, `longmemeval`, `hotpotqa`, and `narrativeqa`.

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

Our work is built on the following datasets and codebases, and we are deeply grateful for their contributions.

- [HotpotQA](https://github.com/hotpotqa/hotpot): Multi-hop question answering benchmark.
- [NarrativeQA](https://github.com/google-deepmind/narrativeqa): Reading comprehension benchmark over narratives.
- [LoCoMo](https://github.com/snap-research/locomo): Long-context multi-session conversation benchmark.
- [LongMemEval](https://github.com/xiaowu0162/longmemeval): Long-term memory evaluation benchmark.
- [General Agentic Memory (GAM)](https://github.com/VectorSpaceLab/general-agentic-memory): Prior memory-framework research we build upon.

## 🥰 Citation

We appreciate your citations if you find our paper relevant and useful to your research!

```bibtex
@article{liu2026mempro,
  title={MemPro: Agentic Memory Systems as Evolvable Programs},
  author={Liu, Qingshan and Wang, Guoqing and Wu, Wen and Huang, Jingqi and Tao, Xinqi and Song, Dejia and Zhou, Jie and He, Liang},
  journal={arXiv preprint arXiv:2606.00619},
  year={2026}
}
```

## 📧 Contact

For questions, suggestions, or bug reports, please contact:

```
51285901015@stu.ecnu.edu.cn
```
