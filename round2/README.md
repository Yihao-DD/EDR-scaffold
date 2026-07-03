# Phase 2 Round-2 Iteration

This directory contains the cold-start scripts for the second iteration after
Phase 0 rep2. It assumes M1 is the designated `main rep2` adapter recorded in
`repro_rep2/MANIFEST.md`.

## Locked Decisions

- Base stack for collection: `M0 + A1(M1)`.
- New teacher loop: one NL-evo round, validation-only AST acceptance.
- T2 construction: train split only, T=0 x1 plus T=0.8 x4 with optional x8
  completion, AST filtering, dedupe, and leak asserts.
- Replay2: M1 T=0 train successes, 2:1 replay ratio, all eval surfaces excluded.
- A2 training recipe: rank 16, lr 5e-5, epoch 3, replay 2:1, KL lambda 2.
- KL anchor: frozen M1, not raw M0.
- Seeds: `20260708..20260712`.

## Pipeline

```bash
python3 round2/collect_failures.py \
  --m1-adapter repro_rep2/artifacts/main_rep2_seed20260704/adapter \
  --dry-run --limit 20

python3 round2/loop/evolution_loop_round2.py \
  --m1-adapter repro_rep2/artifacts/main_rep2_seed20260704/adapter \
  --f2 round2_outputs/f2_failures.json \
  --dry-run

python3 round2/build_t2.py \
  --teacher2-samples round2_outputs/teacher2_samples.jsonl

python3 round2/build_replay2.py \
  --m1-success-jsonl round2_outputs/m1_train_success.jsonl

python3 round2/train_round2.py \
  --m1-adapter repro_rep2/artifacts/main_rep2_seed20260704/adapter \
  --train-jsonl round2_outputs/t2.jsonl \
  --replay-jsonl round2_outputs/replay2.jsonl \
  --seed 20260708 \
  --output-adapter-dir round2_outputs/a2_seed20260708 \
  --output-json round2_outputs/a2_seed20260708.json \
  --dry-run

python3 round2/eval_round2.py \
  --m1-adapter repro_rep2/artifacts/main_rep2_seed20260704/adapter \
  --a2-adapter round2_outputs/a2_seed20260708 \
  --dry-run

python3 round2/reconcile.py --skip-pytest
```

Full GPU execution should remove `--dry-run` after data generation files exist.

## Review Hooks

`docs/REVIEW.md` records smoke commands, local checks, and the explicit manual
review of the M1 KL-anchor line.
