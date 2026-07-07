# FILEMAP — what every directory and file is for

Conflict rule: `PROJECT_MASTER_PLAN.md` beats everything; `CHANGELOG.md` is
the execution-time ledger (latest entry = current truth).

```
EDR-scaffold/
├── RUNBOOK.md                  ★ START HERE to execute Phase 1/2 (four commands)
├── run.py                      ★ unified launcher (preflight/phase1/phase2/all/status/report)
├── configs/launch.json         single source of config: model id, seeds, locked recipes, GPU knobs
├── requirements.txt            pinned core deps · requirements-phase1.txt: A6 dense extras
├── REPORT.md                   generated combined report (run.py report)
│
├── PROJECT_MASTER_PLAN.md      preregistered research program (v1.2) — claims, arms, gates, metrics
├── CHANGELOG.md                append-only decision/verdict ledger (v1.2 … v1.26). Current state = last entry
├── METHODOLOGY.md              consolidated methodology description
├── AGENTS.md                   short agent-facing rules; TASK_GUIDANCE.md is a legacy pointer
├── docs/
│   ├── BASELINES.md            ★ every arm + every reference number + provenance
│   ├── GPU_SCHEDULING.md       launcher lane design + memory table
│   ├── FILEMAP.md              this file
│   ├── PHASE2_HANDOFF.md       original round-2 handoff instruction (v1.22, Chinese)
│   └── REVIEW.md               code-review checklist results for the handoff package
│
├── launcher/                   scheduler internals
│   ├── steps.py                the full step DAG for phase1 + phase2 (commands, deps, outputs)
│   ├── runner.py               GPU-lane scheduler, resume, logging
│   ├── state.py                run_state/state.json semantics
│   ├── gpus.py                 nvidia-smi lane detection
│   └── gate2_glue.py           measured-forget + retention_2 + classify_gate2 wrapper
│
├── phase1/                     Phase-1 arm implementations (all support --dry-run)
│   ├── common.py               paths, config, frozen-split loading, 4-way leakage asserts
│   ├── a5_resample_teacher.py  GPU: teacher forward keeping ALL samples (V=0 and V=1)
│   ├── a5_build_unverified.py  A5 dataset: AST filter OFF, row-matched 444
│   ├── a6_retrieval.py         BM25 + pinned dense retrieval (frozen index) + injection forwards
│   ├── a7_a8_build.py          A7 no-replay + A8 25/50% datasets from frozen rep2 rows
│   ├── a11_placebo.py          token-scrambled patches (archived seed) + forwards
│   ├── step15_sample_nn.py     qualitative-case packaging (human writes conclusions)
│   └── gate1_report.py         Gate-1 mechanical tally (paired bootstrap, Wilson CI, PROBE tags)
│
├── round2/                     Phase-2 iteration code (see round2/README.md for the manual chain)
│   ├── collect_failures.py     F2 = M1's remaining train-share failures
│   ├── loop/                   M1-aware vendored evolve loop, teacher-2 forward, equivalence smoke
│   ├── build_t2.py             T2 (AST fail-closed) + t1_t2.jsonl (2x2 "seen" signature)
│   ├── build_m1_success.py     replay source pool (M1 successes, eval surfaces excluded)
│   ├── build_replay2.py        2:1 replay block
│   ├── train_round2.py         locked A2 recipe launcher (KL anchor = M1, not M0)
│   ├── eval_round2.py          M2 heldout/sibling/(val,old400)/teacher2 + retention_2
│   ├── classify_gate2.py       mechanical compound/converge/collapse classifier
│   └── reconcile.py            company acceptance: prints ACCEPTED iff M1 reproduces on this machine
│
├── repro_rep2/                 ★ frozen Phase-0 rep2 package (the clean line)
│   ├── MANIFEST.{md,json}      SHA256 inventory + M1_DESIGNATED (seed 20260704)
│   ├── data/                   frozen datasets, episode-id lists, pass@16 partition,
│   │                           sibling arena, teacher heldout reference (A2 = 50/158)
│   ├── artifacts/              5 rep2 adapters + their heldout/sibling evals (A3 reference)
│   ├── scripts/                frozen Phase-0 code: lora_phase0 (train/eval), s07 evaluators, probes
│   └── tests/                  pytest: leakage asserts, round2 contracts, phase1/launcher contracts
│
├── EDG-EXP1/                   BFCL v4 data + base-model failure records (a2_failures.json)
├── EDG-EXP2-struct/            EXP2 scaffold loop + H1 accepted patches (evolution_main_merged.json)
├── EDG-BFCL/                   BFCL verifier package material
├── EDG-EXP3-distill/           Phase-0 working logs/results kept for audit (not needed to execute)
│
├── phase1_outputs/             (created at run time) Phase-1 datasets, adapters, evals, gate1_report
├── round2_outputs/             (created at run time) Phase-2 artifacts, evals, gate2_report
└── run_state/                  (created at run time) state.json + per-step logs + preflight report
```

Provenance chain for any number: `REPORT.md` → per-arm summary JSON →
per-record eval JSON → adapter `train_metadata.json` → dataset jsonl →
frozen inputs in `repro_rep2/data/` — every hop is a file in this map.
