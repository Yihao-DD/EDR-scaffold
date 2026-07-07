# Project Instructions

`PROJECT_MASTER_PLAN.md` is the only authoritative specification for this
workspace; `CHANGELOG.md` (latest entry) is the current execution truth.

## If your job is to RUN Phase 1 / Phase 2

Read `RUNBOOK.md` and use `run.py` — do not hand-run individual scripts:

```bash
python3 run.py preflight && python3 run.py phase1 && python3 run.py phase2 && python3 run.py report
```

Discipline while running (violations invalidate results):

1. All gates, metrics, splits, recipes, seeds, and leakage rules are frozen.
   `configs/launch.json` recipes are preregistered (CHANGELOG v1.26) — never tune them.
2. BFCL AST is the only judge; no LLM/GPT judge anywhere.
3. Never bypass a leakage assert; report it.
4. Gate reports are mechanical; interpretation requires the project owner.
5. Negative outcomes are deliverables; ship them unsoftened.

## If your job is to MODIFY the project

1. Read `PROJECT_MASTER_PLAN.md` in full first, then the tail of `CHANGELOG.md`.
2. Any preregistration-relevant change needs a CHANGELOG entry signed by the owner.
3. Frozen Phase-0 code and data live in `repro_rep2/` — treat as read-only;
   new work goes in `phase1/`, `round2/`, or `launcher/` with contract tests
   in `repro_rep2/tests/`.
4. Reuse `EDG-EXP1/` / `EDG-EXP2-struct/` as data/code dependencies; do not rewrite them.
5. `EDG-EXP3-distill/` is the archived Phase-0 working area (audit trail), not the active workspace.

Legacy documents are archived under `archive/`; the master plan wins conflicts.
`ops/SERVERS.md` (gitignored) holds private server details.
