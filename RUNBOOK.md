# RUNBOOK — how to run Phase 1 + Phase 2 end to end

This is the single entry document for executing the remaining EDR experiments.
It is written for an autonomous agent (or a human) on a GPU machine. You do
not need to read the research history to execute; you DO need to follow the
discipline rules in §7 exactly.

**The entire job is four commands** (§4). Everything is resumable: if a step
fails or the machine dies, re-run the same command and it continues from
where it stopped.

---

## 1. What this repository is (30-second version)

EDR (Evolve → Distill → Retire) studies whether repairs taught to a small
model by an external scaffold can be distilled into LoRA weights and survive
after the scaffold is removed. Phase 0 (already done, results frozen in
`repro_rep2/` and `CHANGELOG.md`) established the headline results. What
remains, and what you will run:

- **Phase 1** — five preregistered ablation/control arms (A5, A6, A7, A8,
  A11) that isolate the mechanisms behind the Phase 0 result. See
  `docs/BASELINES.md` for what every arm means and where every reference
  number comes from.
- **Phase 2** — the second Evolve→Distill→Retire iteration on top of the
  designated round-1 model M1 (collect new failures F2 → evolve harness H2 →
  build dataset T2 → train A2 ×5 seeds → evaluate M2 → mechanical Gate-2
  classification).

Deliverables you will produce:

| file | what it is |
|---|---|
| `phase1_outputs/gate1_report.md` | Gate-1 mechanical tally (all five arms) |
| `round2_outputs/gate2_report.md` | Gate-2 dynamics classification |
| `REPORT.md` | combined top-level report |
| `run_state/state.json` | full step ledger (send this back too) |

## 2. Environment setup

```bash
# python 3.10+
pip install -r requirements.txt          # pinned: torch/transformers/peft/...
pip install -r requirements-phase1.txt   # sentence-transformers (A6 dense retriever)

# large artifacts (M1 adapter etc.) are in Git LFS:
git lfs install && git lfs pull

# base model: Qwen/Qwen2.5-7B-Instruct from HuggingFace.
# Either let transformers download it (set HF_HOME if needed), or download it
# yourself and set "model_id" in configs/launch.json to the local path.
```

GPU requirements:

- **Phase 1**: any CUDA card with ≥ 24 GB free (QLoRA training + forwards).
- **Phase 2**: A2 training and M2 evaluation merge M1 into bf16 weights —
  use a card with **≥ 48 GB** (A100/H100/RTX 6000 Ada class). This is a
  frozen property of the round-2 code, not a launcher choice.
- Multiple cards are used automatically, one step per card. Details and
  knobs: `docs/GPU_SCHEDULING.md`.

## 3. Configuration

Everything lives in `configs/launch.json`. You may adapt **paths and GPU
settings** (`model_id` to a local path, `model_cache_dir`, `gpu.max_lanes`,
`gpu.min_free_mem_gb`). You may **not** change seeds, recipes, fractions,
retrieval spec, or scramble seed — those are preregistered (CHANGELOG v1.26)
and changing them invalidates the run.

## 4. The four commands

Run from the repository root, in this order:

```bash
# 1. verify environment, data, LFS, package integrity, dry-run smokes (no GPU needed)
python3 run.py preflight
# must end with:  preflight PASS

# 2. all of Phase 1 (builds datasets, 12 trainings, all forwards, Gate-1 tally)
python3 run.py phase1
# add --dry-run first if you want to see the full step plan without executing

# 3. all of Phase 2 (acceptance check, F2 collection, loop, T2, 5 trainings, 5 evals, Gate-2)
python3 run.py phase2

# 4. regenerate the combined report at any time
python3 run.py report
```

Monitoring while a phase runs:

```bash
python3 run.py status          # step table
tail -f run_state/logs/<step>.log
```

Approximate wall-clock on 2× 48GB cards: Phase 1 ≈ 1–2 days (dominated by
12 small LoRA trainings + ~4k forwards), Phase 2 ≈ 1–2 days (dominated by
the evolve loop and 5 trainings). One card roughly doubles it.

## 5. Success criteria (mechanically checkable)

Phase 1 is DONE when `python3 run.py status` shows every `p1.*` step `done`
and both `phase1_outputs/gate1_report.md` and `.json` exist. Spot-check:

- `phase1_outputs/a5/a5_dataset_summary.json` — `total_rows` = 444 and
  `core_ast_fail_rows` > 0 (the arm is meaningless if no unverified rows made it in);
- `phase1_outputs/a6/a6_summary.json` — 4 cells (`bm25:k1/k3`, `dense:k1/k3`),
  each with `repair` and `ctx_tokens_mean`, plus a frozen retrieval artifact
  with the dense model revision pinned;
- `phase1_outputs/a11/a11_summary.json` — three surfaces with `placebo_repair`.

Phase 2 is DONE when every `p2.*` step is `done`, `p2.reconcile_m1` printed
`ACCEPTED` in its log (this is the company acceptance gate: M1 reproduces
heldout 51/158 ±1 and sibling 292/300 ±2 on YOUR machine), and
`round2_outputs/gate2_report.md` exists with a classification line
(`compound` / `converge` / `collapse` — all three are valid outcomes).

## 6. Failure playbook

A failed step prints `[FAIL] <step> rc=...` with a log tail and blocks only
its dependents; independent branches keep running. Then:

1. `tail -100 run_state/logs/<step>.log`
2. Match against this table:

| symptom | cause | fix |
|---|---|---|
| `CUDA out of memory` | card too small / lane contention | lower `gpu.max_lanes` to 1, or use the bigger card only (`CUDA_VISIBLE_DEVICES` before run.py); Phase 2 needs ≥48GB |
| `No module named 'scripts'` | PYTHONPATH lost (running a script by hand) | run through `run.py`, or `export PYTHONPATH=$PWD:$PWD/repro_rep2` |
| HF download errors | no network / no cache | pre-download the model, set `model_id` to the local path |
| adapter file is ~130 bytes | Git LFS not pulled | `git lfs pull`, then `python3 run.py preflight` |
| `AssertionError {..._disjoint_...}` | leakage assert fired | STOP. Do not bypass. Report the full assert payload upstream. |
| `sentence_transformers` import error | phase-1 extras missing | `pip install -r requirements-phase1.txt` |
| reconcile exits `heldout reference outside +/-1` | environment does not reproduce M1 | STOP and report; do not proceed to training on a non-reproducing environment |
| `MATERIAL_EXHAUSTION` marker from build_t2 | \|T2\| < 30 | NOT an error — preregistered finding; the pipeline continues and the result is labeled PROBE |

3. After fixing the environment, re-run the same `python3 run.py phaseN`
   command — completed steps are skipped, the failed one retries.

## 7. Discipline rules (non-negotiable)

These come from the project's preregistration protocol (`PROJECT_MASTER_PLAN.md`
Part VII). Violating them invalidates the results:

1. **Never edit recipes, seeds, or gate thresholds** to make a step pass.
2. **Never bypass a leakage assert.** An assert firing is a result, not a bug to route around.
3. **No LLM judges anywhere.** The only verifier is the BFCL AST matcher already wired in.
4. **Gate reports are mechanical tallies.** Do not add interpretation, do not
   soften a FAIL, do not re-run an arm with different settings because its
   number "looks wrong". If a number looks wrong, report it.
5. **Failures are deliverables.** A5 not separating from A3, placebo > 0,
   collapse in Gate 2 — all preregistered outcomes. Ship them as-is.
6. Anything not covered here: do NOT improvise; ask the project owner.

## 8. What to send back

- `phase1_outputs/gate1_report.{md,json}` and the per-arm summaries
  (`a5_dataset_summary.json`, `a6_summary.json`, `a11_summary.json`,
  `a7a8_dataset_summary.json`, `step15_nn_package.json`)
- `round2_outputs/gate2_report.md`, `round2_outputs/eval/gate2.json`,
  `round2_outputs/t2_summary.json`, all `m2_seed*.{heldout,sibling}.json`,
  the reconcile outputs
- `run_state/state.json` + any failed-step logs
- Trained adapters: commit via Git LFS on a branch, or archive out-of-band —
  either way record SHA256 (convention: `repro_rep2/MANIFEST.md`).

Do NOT delete anything under `phase1_outputs/`, `round2_outputs/`, or
`run_state/` — the ledger is part of the deliverable.
