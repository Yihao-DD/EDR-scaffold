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
- [x] Round2 dry-run smoke passed for `collect_failures -> loop -> train_round2`.
- [x] Round2 eval dry-run command rendered.
- [x] Focused `pytest` run for rep2 data/leakage asserts passed.
- [x] Artifact SHA manifest generated: `repro_rep2/MANIFEST.md`, `repro_rep2/MANIFEST.json`, `repro_rep2/SHA256SUMS`.
- [x] M1 designated seed and SHA recorded: seed `20260704`, adapter SHA256 `c3afb185a6a32480a55e678599b3bebc8b8899ed935d620605b00ab3c3e62683`.
- [x] Reconcile smoke passed: `ACCEPTED`.

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
python3 -m pytest -q repro_rep2/tests/test_repro_rep2_asserts.py
..                                                                       [100%]
2 passed in 0.03s
```

Reconcile smoke:

```text
python3 round2/reconcile.py --skip-pytest
ACCEPTED
```

Round2 dry-run:

```text
wrote /tmp/f2_smoke.json f2_count=20
DRY_RUN python3 EDG-EXP2-struct/scripts/evolution_main.py --adapter repro_rep2/artifacts/main_rep2_seed20260704/adapter --failures /tmp/f2_smoke.json --validation-ids repro_rep2/data/episode_ids/D_val_failures.json
DRY_RUN python3 repro_rep2/scripts/lora_phase0.py --model-id Qwen/Qwen2.5-7B-Instruct train --dataset /tmp/a2_smoke/round2_train_seed20260708.jsonl --base-adapter-dir repro_rep2/artifacts/main_rep2_seed20260704/adapter --output-dir /tmp/a2_smoke --rank 16 --lr 5e-5 --seed 20260708 --epochs 3 --kl-anchor-lambda 2 --max-grad-norm 1.0
```

Eval dry-run:

```text
DRY_RUN python3 repro_rep2/scripts/s07_heldout_eval.py --model-id Qwen/Qwen2.5-7B-Instruct --adapter-dir /tmp/a2_smoke --arm main --config round2 --seed 0 --run-id round2_eval --train-dataset round2_outputs/t2.jsonl --output /tmp/eval_smoke/m2.heldout.json
DRY_RUN python3 repro_rep2/scripts/s07_sibling_arena_eval.py --model-id Qwen/Qwen2.5-7B-Instruct --adapter-dir /tmp/a2_smoke --arm main --config round2 --seed 0 --run-id round2_eval --output /tmp/eval_smoke/m2.sibling.json
SKIP val_old400: pass --phase0-failures, --phase0-exp2-root, and --phase0-s00-input to enable it.
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
