# v1.22 Code Review

Branch: `handoff/phase2-round2`

Scope: rep2-only handoff package plus Phase 2 round-2 cold-start code. Rep1
quarantine/repair artifacts are intentionally excluded.

## Checklist

- [x] Handoff source text saved at `docs/PHASE2_HANDOFF.md`.
- [x] Git LFS installed and `*.safetensors`, `*.bin`, `*.pt` tracked for large adapters.
- [x] rep2 data package created under `repro_rep2/data`.
- [x] main rep2 core rows frozen as `repro_rep2/data/distill_main_core.jsonl` (148 rows).
- [x] capped replay2 training rows frozen as `repro_rep2/data/distill_main_replay2_capped.jsonl` (444 rows).
- [x] episode-id lists generated: D_val=156, D_heldout=158, old400=400, capped106=106, sibling300=300.
- [x] judge code copied with parallel-family extension and manual audit log.
- [x] Round2 code added under `round2/`.
- [x] KL anchor semantics manually checked: `repro_rep2/scripts/lora_phase0.py` merges `--base-adapter-dir` before attaching A2; therefore `disable_adapter()` in KL evaluates M1, not raw M0.
- [x] No SSH credentials or machine passwords are stored in the handoff files.
- [x] Local compile check passed.
- [x] Data/assert smoke passed without pytest dependency.
- [x] Round2 dry-run smoke passed for `collect_failures -> M1-aware loop -> train_round2`.
- [x] Loop M1 invariant manually checked: `round2/loop/evolution_loop_m1.py` loads base -> `PeftModel(A1)` -> `merge_and_unload()` and exits if no M1 adapter is supplied.
- [x] Loop has no raw-M0 residual path: all patch-search generate calls flow through `ModelRunner.generate()` in `round2/loop/evolution_loop_m1.py`.
- [x] Round2 eval dry-run command renders M2 as `M0 + A1 + A2` by passing `--base-adapter-dir` to all eval surfaces.
- [x] Focused `pytest` run for rep2 data/leakage and round2 contract asserts passed.
- [x] Artifact SHA manifest generated: `repro_rep2/MANIFEST.md`, `repro_rep2/MANIFEST.json`, `repro_rep2/SHA256SUMS`.
- [x] M1 designated seed and SHA recorded: seed `20260704`, adapter SHA256 `c3afb185a6a32480a55e678599b3bebc8b8899ed935d620605b00ab3c3e62683`.
- [x] Reconcile package-only smoke passed without printing `ACCEPTED`; full `ACCEPTED` now requires model re-evaluation.

## Smoke Output

Compile:

```text
python3 -m compileall -q round2 repro_rep2/scripts/make_2x2_labels.py repro_rep2/scripts/lora_phase0.py repro_rep2/scripts/s07_heldout_eval.py repro_rep2/scripts/s07_sibling_arena_eval.py
compile_ok
```

Data/assert smoke:

```text
assert_smoke_ok
```

Focused pytest:

```text
python3 -m pytest -q repro_rep2/tests/test_repro_rep2_asserts.py repro_rep2/tests/test_round2_contracts.py
...........                                                              [100%]
11 passed in 0.25s
```

Reconcile smoke:

```text
python3 round2/reconcile.py --skip-pytest --skip-model-check
PACKAGE_ONLY: static files and pytest passed; model re-evaluation was skipped.
```

Round2 dry-run:

```text
wrote /tmp/f2_smoke.json f2_count=20
DRY_RUN python3 round2/loop/evolution_loop_m1.py --input /tmp/f2_smoke.json --output /tmp/loop_smoke/evolution_loop.json --log /tmp/loop_smoke/evolution_loop.md --model-id Qwen/Qwen2.5-7B-Instruct --manifest repro_rep2/MANIFEST.json --seeds 20260630 --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter
DRY_RUN python3 repro_rep2/scripts/lora_phase0.py --model-id Qwen/Qwen2.5-7B-Instruct train --dataset /tmp/a2_smoke/round2_train_seed20260708.jsonl --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --output-dir /tmp/a2_smoke --rank 16 --lr 5e-5 --seed 20260708 --epochs 3 --kl-anchor-lambda 2 --max-grad-norm 1.0
```

Eval dry-run:

```text
DRY_RUN python3 repro_rep2/scripts/s07_heldout_eval.py --model-id Qwen/Qwen2.5-7B-Instruct --adapter-dir /tmp/a2_smoke --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --arm main --config round2 --seed 0 --run-id round2_eval --train-dataset T1_T2.jsonl --output /tmp/eval_smoke/m2.heldout.json
DRY_RUN python3 repro_rep2/scripts/s07_sibling_arena_eval.py --model-id Qwen/Qwen2.5-7B-Instruct --adapter-dir /tmp/a2_smoke --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --arm main --config round2 --seed 0 --run-id round2_eval --output /tmp/eval_smoke/m2.sibling.json
DRY_RUN python3 repro_rep2/scripts/lora_phase0.py --model-id Qwen/Qwen2.5-7B-Instruct --failures failures.json --exp2-root exp2 --s00-input s00.json eval --adapter-dir /tmp/a2_smoke --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --output /tmp/eval_smoke/m2.val_old400.json
DRY_RUN python3 round2/loop/teacher2_forward.py --loop-output /tmp/loop_smoke/evolution_loop.json --loop-method NL-evo --manifest repro_rep2/MANIFEST.json --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --output /tmp/eval_smoke/m2.teacher2_heldout.json
```

## Manual Review Notes

- Adapter binaries exceed GitHub's normal file-size policy and are tracked with
  Git LFS.
- Model weights are not included; fetch `Qwen/Qwen2.5-7B-Instruct` from Hugging
  Face in the execution environment.
- Round2 `train_round2.py` writes a plan JSON before invoking the trainer; this
  is the auditable record of the M1-anchor invariant.
- Round2 `train_round2.py` combines T2 and replay2 into a per-seed JSONL before
  training. `eval_round2.py` always renders heldout/sibling commands and renders
  val@156/old400 when the Phase0 EXP1/EXP2 inventory paths are supplied.
- `build_t2.py` is AST fail-closed; rows missing `ast_pass` now assert instead
  of entering T2.
- `build_replay2.py` writes `source=replay_base_success` and `partition=replay`
  on every replay row so the KL-anchor path fires.
- The EXP2 patch loop is vendored as `round2/loop/evolution_loop_m1.py` and is
  adapter-aware. It resolves M1 from `repro_rep2/MANIFEST.json` unless an
  explicit `--base-adapter-dir` is supplied, merges A1 into memory, and refuses
  raw-M0 execution.
- GPU equivalence smoke command is implemented as
  `round2/loop/no_patch_equivalence_smoke.py`; it must pass on the execution
  host before accepting a company rerun:
  `python3 round2/loop/no_patch_equivalence_smoke.py --collect-json round2_outputs/f2_failures.json --limit 10`.
- Manual review item: `rg -n "AutoModelForCausalLM.from_pretrained|PeftModel.from_pretrained|model.generate\\(" round2/loop`
  shows the only model-load path is `ModelRunner` in
  `round2/loop/evolution_loop_m1.py`; wrapper/smoke scripts import that entry
  and do not load a second model path.

## v1.23 additions (missing-producer + dynamics scriptization)

Second-review follow-up: two pipeline files were consumed but never produced
(`m1_train_success.jsonl`, `t1_t2.jsonl`), and Gate 2 / retention_2 were prose,
not code. Fixed and covered by tests.

- [x] `round2/build_m1_success.py` produces `m1_train_success.jsonl`: pool =
      M1 T=0 @ {capped-120 ∪ eliminated}, keeps V=1, reports V=0 exclusions,
      asserts disjoint from heldout/D_val/old400/sibling300. capped-120 frozen as
      `repro_rep2/data/episode_ids/capped120_replay.json` (120 unique round-1
      replay episodes, disjoint from all eval surfaces). Not just `eliminated`.
- [x] `collect_failures.py` now emits `eliminated_episode_ids` for the pool.
- [x] `build_t2.py` writes `t1_t2.jsonl` = distill_main core (148) ∪ T2, deduped
      by episode_id; `assert_main_arm` confirms no STaR fork (role fields only).
- [x] `round2/classify_gate2.py` classifies compound/converge/collapse by 5-seed
      paired-bootstrap CI on heldout; `eval_round2.py` writes `retention2.json`.
      HANDOFF §2.6 now points at the scripts, not prose rules.
- [x] KL-ref regression test: `test_kl_reference_is_merged_m1_not_raw_m0`
      (merge-before-apply-lora ordering + `disable_adapter` + merged-reference
      metadata) and `test_train_round2_dry_run_anchors_kl_to_m1`.
- [x] Input-file existence tests: `test_build_t2_produces_t2_and_t1_t2`,
      `test_build_m1_success_dry_run_produces_replay_source`.
- [x] Full round2 pipeline dry-run walked; all consumed intermediates now have a
      producing step in `round2/README.md`.

train_round2 anchor lines (manual re-confirm):
- stacking: `round2/train_round2.py` passes `--base-adapter-dir args.m1_adapter`
  to the trainer; `repro_rep2/scripts/lora_phase0.py::train` calls
  `merge_base_adapter(model, args.base_adapter_dir)` before `apply_lora(...)`.
- KL reference: `lora_phase0.py::kl_anchor_loss` uses `with model.disable_adapter():`
  after the merge, so the reference is merged M1; `train_metadata.kl_anchor_reference`
  records `merged_base_adapter`.
