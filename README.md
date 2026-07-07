# EDR Workspace — Evolve → Distill → Retire

Self-contained self-improvement for small function-calling models: an external
scaffold repairs a frozen 7B's failures (Evolve), verifier-certified repair
trajectories are distilled into LoRA weights (Distill), and the scaffold is
removed for all headline evaluation (Retire).

## To execute Phase 1 / Phase 2 (company / GPU machine)

**Read `RUNBOOK.md`. The whole job is:**

```bash
python3 run.py preflight
python3 run.py phase1
python3 run.py phase2
python3 run.py report
```

Supporting docs: `docs/BASELINES.md` (what every arm/baseline is),
`docs/FILEMAP.md` (what every file is), `docs/GPU_SCHEDULING.md` (lane design).

## Current state (see CHANGELOG.md latest entry for ground truth)

- **Phase 0 is complete and frozen.** Clean line = rep2
  (`repro_rep2/`, 5 seeds): retention ≈ 1.0 (C1 transfer strong), but the
  no-regression hard gate failed (sibling forget 0.045 > 0.02); C2 holds on
  heldout scaffold-only (A3−A4 = +0.153, CI [+0.064, +0.251]). rep1 is
  quarantined (provenance) and must not be cited as a clean claim.
- **Phase 1 (A5/A6/A7/A8/A11 ablations) and Phase 2 (round-2 iteration)**
  are fully automated behind `run.py` — preregistered recipes locked in
  `configs/launch.json` (CHANGELOG v1.26).
- M1 designated: main rep2 seed 20260704 (`repro_rep2/MANIFEST.json`).

## Authoritative spec

- `PROJECT_MASTER_PLAN.md` — the preregistered program; wins every conflict.
- `CHANGELOG.md` — append-only execution ledger; the last entry is the
  current truth.
- `METHODOLOGY.md` — consolidated methodology.

## Non-negotiable discipline

- BFCL AST is the only verifier; no LLM judges anywhere.
- Leakage is blocked by code asserts + pytest (`repro_rep2/tests/`), never by convention.
- `PROBE signal (n=X)` labels on any number with n<30 or seeds<3 (phase-1 arms: seeds=3 ⇒ PROBE).
- Gate reports are mechanical tallies; interpretation requires human sign-off.
- Negative branches of every gate are deliverables, not embarrassments.
