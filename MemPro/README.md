# MemPro Benchmark Evolution Workspaces

This directory contains clean public evolution workspaces for the four MemPro
benchmarks: LoCoMo, LongMemEval, HotpotQA, and NarrativeQA.

Each benchmark directory contains:

- `AGENTS.md`: instructions for evolving that benchmark framework.
- `versions/`: editable framework snapshots.
- `registry/versions.json`: version-tree metadata.
- `scripts/`: utilities for selecting a base, registering a child version, and running evaluation.
- `runs/`: generated runtime packages and evaluation outputs.
- `analysis_notes/`: optional human-readable notes.

The workspaces are intentionally lightweight. Historical private runs, backups,
and raw analysis logs are not included. All paths are relative to this MemPro
repository.

## Basic Commands

From a benchmark directory:

```bash
python scripts/select_base.py
python scripts/register_version.py --base-version v0000 --new-version v0001 --note "short change note"
python scripts/run_eval.py --version v0001 --split train --limit 5 -- --data path/to/data.json
```

`run_eval.py` materializes a temporary `mempro_memory` package under
`runs/<version>/runtime_parent/` and sets `PYTHONPATH` to that package. It does
not overwrite the inference runtimes under `best_versions/`.
