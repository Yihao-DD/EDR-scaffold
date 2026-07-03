# Rep2 Reproduction Package

This package contains the clean Phase 0 `main rep2` line only. It excludes all
rep1 quarantine and repair artifacts.

## Contents

- `scripts/`: frozen Phase 0 training, evaluation, sibling-arena, and judge code.
- `data/`: frozen distillation rows, capped replay rows, arena files, pass@16
  partitions, and episode-id lists.
- `artifacts/main_rep2_seed*/`: the five main rep2 adapters, train/eval JSONs,
  and logs.
- `MANIFEST.md` and `MANIFEST.json`: SHA256, size, seed, config, and dry-run
  reconciliation metadata.
- `run_repro.sh`: one-command seed reproduction/evaluation launcher.

## Environment

Install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base model is `Qwen/Qwen2.5-7B-Instruct` from Hugging Face. The model
weights are intentionally not stored in this repository.

CUDA note: Phase 0 rep2 was trained with QLoRA on CUDA GPUs; Round 2 M1-anchor
training merges A1 in memory before attaching A2 and should be run on a larger
GPU, preferably 49GB or higher.

## Reproduce M1

The designated M1 adapter is selected by the preregistered rule: among the five
main rep2 seeds, choose the median heldout repair count; ties choose the smaller
seed. See `MANIFEST.md` for the exact seed, SHA256, and reference metrics.

Run a smoke reconciliation:

```bash
python3 round2/reconcile.py --skip-pytest
```

Run a full local acceptance check after dependencies are installed:

```bash
python3 round2/reconcile.py
```

## Re-train One Rep2 Seed

```bash
./repro_rep2/run_repro.sh 20260704
```

The script writes outputs under `repro_rep2/reruns/seed20260704/`.
