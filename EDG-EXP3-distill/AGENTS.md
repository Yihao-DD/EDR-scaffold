# EDG-EXP3-distill Instructions

This is the active EDR experiment workspace.

## Authoritative Spec

Read `../PROJECT_MASTER_PLAN.md` before any project change. The local copy at `guidance/EDR_PROJECT_MASTER_PLAN.md` mirrors the same v1.1 master plan.

If any file in this directory conflicts with `../PROJECT_MASTER_PLAN.md`, the root master plan wins.

`guidance/archive/EXP3_PHASE0_GUIDANCE_legacy.txt` is legacy Phase 0 guidance that has been absorbed into the master plan.

## Current Phase

Phase 0 is active.

Completed:

- Step 0.0 inventory: `results/s00_inventory.json`, `logs/s00.md`
- Step 0.1 pass@16 partition: `results/s01_pass16_partition.json`, `logs/s01.md`
- Step 0.2 dataset construction: `data/distill_main.jsonl`, `data/distill_star.jsonl`, `results/s02_datasets.json`, `logs/s02.md`

In progress:

- Step 0.3 LoRA/QLoRA grid training on the two servers.

## Non-Negotiable Rules

1. BFCL AST verifier is the only judge.
2. No GPT or LLM-as-judge in filtering, validation, evaluation, or partitioning.
3. Held-out and R-success leakage assertions must remain in code and tests.
4. A3/A4/A5 comparisons must keep data amount, grid, and seeds balanced as specified.
5. `PROBE signal (n=X)` labeling is required for seed<3 or n<30.
6. Gate interpretation, new metrics, distillation-filter changes, and qualitative conclusions require the user to be in the loop.

## Dependencies

Reuse instead of rewriting:

- `../EDG-EXP1/results/a2_failures.json`
- `../EDG-EXP2-struct/scripts/`
- `../EDG-EXP2-struct/results/evolution_main_merged.json`
- `../ops/SERVERS.md` for private server access details.

Do not depend on external download/source paths. Any external instruction file must be copied into the workspace before use.
