# EDR Workspace

This workspace is now organized around **EDR: Evolve -> Distill -> Retire**.

## Authoritative Spec

- `PROJECT_MASTER_PLAN.md` is the highest project guidance.
- `TASK_GUIDANCE.md` is only a compatibility pointer to the master plan.
- Legacy Gate-1 guidance is archived under `archive/legacy_gate1/`.

## Active Work

- `EDG-EXP3-distill/` is the active experiment workspace.
- Current line: Phase 0, Evolve -> Distill -> Retire existence validation.
- Current completed artifacts include Step 0.0, 0.1, and 0.2 outputs.
- Step 0.3 LoRA/QLoRA grid training is running on the two servers.
- Private server access details are in `ops/SERVERS.md`.

## Dependency Workspaces

- `EDG-EXP1/`: baseline BFCL failure data, especially `results/a2_failures.json`.
- `EDG-EXP2-struct/`: NL-evo scaffold result and reusable evaluation/scaffold code.
- `EDG-BFCL/`: BFCL-related historical setup and local verifier material.

Do not delete these dependency workspaces while EXP3 is active.

## Current Discipline

- BFCL AST is the only verifier.
- No GPT or LLM judge is allowed in data filtering, online validation, evaluation, or partitioning.
- Held-out leakage is blocked by code assertions and tests.
- All non-publication-ready numbers must be marked `PROBE signal (n=X)`.
- Any conflict between documents is resolved in favor of `PROJECT_MASTER_PLAN.md`.
- The workspace must not depend on documents stored outside `E:\EDGscaffold`; imported instructions should be copied into this workspace.
