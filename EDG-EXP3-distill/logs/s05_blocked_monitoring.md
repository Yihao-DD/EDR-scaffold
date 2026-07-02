# s05 BLOCKED KL Monitoring

## Local preparation 2026-07-02

- `CHANGELOG.md` v1.11 records BLOCKED opening, KL grid, ep3 candidates, NaN caveat guards, heldout denominator, log-prob follow-up debt, and fixed exit criteria.
- Training stability updates in `scripts/lora_phase0.py`:
  - non-finite loss guard: skip batch/step, clear gradients, log warning;
  - non-finite grad-norm guard after clipping: skip optimizer update, clear gradients, log warning;
  - explicit grad-clip/KL parameter logging;
  - finite checks in the KL log-softmax path.
- `scripts/logprob_followups.py` produced:
  - `results/s04_logprob_followups.json`;
  - `logs/s04_logprob_followups.md`.
- Replay 2:1 datasets:
  - main: 148 core rows + 296 replay rows = 444 rows; 226 unique replay rows + 70 deterministic repeat rows.
  - STaR: 148 core rows + 296 replay rows = 444 rows; 226 unique replay rows + 70 deterministic repeat rows.
  - leakage assertions remain 0 for heldout, R_success_eval, and D_val.

## Remote launch 2026-07-02 13:04 CST

- new westc: launched main replay1 queue:
  - `main_kl_r16_lr5e-5_ep2_replay1_lam0p5_seed20260703`
  - `main_kl_r16_lr5e-5_ep2_replay1_lam1_seed20260703`
  - `main_kl_r16_lr5e-5_ep2_replay1_lam2_seed20260703`
  - `main_ep3_r16_lr5e-5_replay1_seed20260703`
- old main: launched main replay2 queue:
  - `main_kl_r16_lr5e-5_ep2_replay2_lam0p5_seed20260703`
  - `main_kl_r16_lr5e-5_ep2_replay2_lam1_seed20260703`
  - `main_kl_r16_lr5e-5_ep2_replay2_lam2_seed20260703`
- old star: launched STaR replay1 queue:
  - `star_kl_r16_lr1e-4_ep2_replay1_lam0p5_seed20260703`
  - `star_kl_r16_lr1e-4_ep2_replay1_lam1_seed20260703`
  - `star_kl_r16_lr1e-4_ep2_replay1_lam2_seed20260703`
  - `star_ep3_r16_lr1e-4_replay1_seed20260703`
- new westb: launched STaR replay2 queue:
  - `star_kl_r16_lr1e-4_ep2_replay2_lam0p5_seed20260703`
  - `star_kl_r16_lr1e-4_ep2_replay2_lam1_seed20260703`
  - `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`

## Launch health 2026-07-02 13:05 CST

- all four servers have exactly one active `lora_phase0.py` process.
- first active run on each queue is the expected lambda `0.5` KL point.
- GPU memory/utilization:
  - new westc: ~16.0 GB / 24.6 GB, ~100% utilization.
  - old main: ~16.0 GB / 49.1 GB, ~98% utilization.
  - old star: ~14.6 GB / 24.6 GB, eval/training transient ~52% at check.
  - new westb: ~15.4 GB / 24.6 GB, ~100% utilization.
- guard behavior visible: non-finite loss skips are being logged instead of contaminating optimizer steps. This is expected to be monitored closely; current runs continue.
- no duplicate conflicting training run found.

Automation: heartbeat monitor `edr-s05-blocked-kl-monitor` created at a 5-minute cadence.

## Remote heartbeat 2026-07-02 13:08 CST

- all four queues remain active with exactly one expected `lora_phase0.py` process.
- active point on every queue is still the first KL point (`lambda=0.5`).
- GPU utilization is healthy:
  - new westc main replay1: ~16.0 GB / 24.6 GB, 100%.
  - old main main replay2: ~21.0 GB / 49.1 GB, 100%.
  - old star STaR replay1: ~15.8 GB / 24.6 GB, 100%.
  - new westb STaR replay2: ~17.4 GB / 24.6 GB, 100%.
- no OOM/import/CUDA/runtime failure surfaced in the checked tails.
- non-finite-loss guard is firing but runs continue:
  - new westc: 5 skips observed by batch 142/296.
  - old main: 4 skips observed by batch 112/444.
  - old star: 4 skips observed by batch 142/296.
  - new westb: 4 skips observed by batch 81/444.
- interpretation: guard is working as intended. Skip counts are nonzero and must remain part of the caveat/metadata, but they are not yet high enough to stop the queue before the first JSON calibrates behavior.

## Remote heartbeat 2026-07-02 13:12 CST

- no s05 JSON has completed yet; all four queues remain on their first KL point (`lambda=0.5`).
- all four servers have exactly one expected `lora_phase0.py` process; no duplicate conflicting run found.
- GPU utilization is healthy:
  - new westc main replay1: ~16.0 GB / 24.6 GB, 100%.
  - old main main replay2: ~21.0 GB / 49.1 GB, 100%.
  - old star STaR replay1: ~16.8 GB / 24.6 GB, 100%.
  - new westb STaR replay2: ~17.4 GB / 24.6 GB, 99%.
- non-finite-loss skip counts:
  - new westc main replay1: 9 skips; run has entered epoch 2.
  - old main main replay2: 8 skips; run is in epoch 1 after logged step 25/112.
  - old star STaR replay1: 7 skips; run has entered epoch 2.
  - new westb STaR replay2: 11 skips; run is in epoch 1 after logged step 25/112.
- error keyword scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in the checked logs.
- ETA update: replay1 first JSON likely before replay2; first JSON still expected in roughly 25-40 minutes, depending on evaluation speed. Full 14-run wave remains closer to 3-4 hours than the earlier 4-6 hour conservative upper bound.

## Remote heartbeat 2026-07-02 13:17 CST

- no s05 JSON has completed yet.
- new westc main replay1 `lambda=0.5`: training completed, adapter saved, eval has started (`eval:val 1/156`); GPU ~16.0 GB, utilization ~58% during eval transition; skip count 16.
- old star STaR replay1 `lambda=0.5`: training completed, adapter saved, eval has started (`eval:val 1/156`); GPU ~16.8 GB, utilization ~13% during eval transition; skip count 12.
- old main main replay2 `lambda=0.5`: still training, logged epoch 1 step 50/112; GPU ~22.2 GB, 100%; skip count 11.
- new westb STaR replay2 `lambda=0.5`: still training, logged epoch 2 after step 50/112; GPU ~18.6 GB, 99%; skip count 18.
- all four servers still have exactly one expected `lora_phase0.py` process; no duplicate conflicting run found.
- no OOM/import/CUDA/runtime failure surfaced in the checked tails. Skip counts are nonzero but still in the expected guarded range for this KL pass.
- ETA update: first replay1 JSON likely within the next 10-20 minutes if eval speed matches s04; replay2 first JSON trails that.

## New server triage 2026-07-02 13:20 CST

- user added a fifth server (`bjb1`, RTX 4090, 24.6 GB).
- triage result: the server is blank for this experiment.
  - no `EDG-EXP3-distill` project tree was present;
  - no Qwen2.5-7B model cache was present;
  - Python environment had `torch` but not `transformers`, `peft`, or `bitsandbytes`;
  - `/root/autodl-tmp` has 50 GB free, enough for setup in principle.
- attempted opportunistic setup was stopped because transfer to the server was too slow: the 4.9 MB minimal project tar stalled at about 1 MB, making a 15 GB model transfer impractical for the current s05 wave.
- decision: do not include this fifth server in current s05 scheduling. Existing four-server queues continue unchanged. The fifth server would only be useful if pre-staged with model cache and Python dependencies, or if a same-region/prebaked image is available.

## Remote heartbeat 2026-07-02 13:22 CST

- no s05 JSON has completed yet.
- new westc main replay1 `lambda=0.5`: training completed, adapter saved, eval in progress (`eval:val 100/156`); GPU ~16.0 GB, utilization ~26%; skip count 16.
- old star STaR replay1 `lambda=0.5`: training completed, adapter saved, eval in progress (`eval:val 50/156`); GPU ~16.8 GB, utilization ~94%; skip count 12.
- old main main replay2 `lambda=0.5`: still training in epoch 2 after logged step 75/112; GPU ~22.2 GB, 99%; skip count 17.
- new westb STaR replay2 `lambda=0.5`: still training in epoch 2 after logged step 75/112; GPU ~18.6 GB, 100%; skip count 27.
- all four servers have one expected active `lora_phase0.py` process; no duplicate conflicting run found.
- no OOM/import/CUDA/runtime failure surfaced in the checked tails. Skip counts remain a caveat but are not yet excessive relative to total batch count.
- ETA update: first replay1 JSON likely within the next heartbeat or two; replay2 first JSON trails replay1.

## New server triage 2026-07-02 13:33 CST

- user added another westc server (`44901`, RTX 4080 SUPER, 32.8 GB).
- triage result: usable hardware but blank for the experiment.
  - no `EDG-EXP3-distill` project tree was present;
  - no Qwen2.5-7B model cache was present;
  - Python has `torch`, but not `transformers`, `peft`, `bitsandbytes`, `accelerate`, or `datasets`;
  - `/root/autodl-tmp` has 50 GB free.
- same-region scp speed test from the existing westc source was about 11.5 MB/s on a 128 MB file.
- staging decision: do not attach it to active queues immediately. Start background staging only: install dependencies and copy the 15 GB model plus project tree. It may be usable for later pending points if staging finishes before the current queues drain.

## Remote update 2026-07-02 13:37 CST

- first s05 JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay1_lam0p5_seed20260703`
  - val: 73/156 = 0.4679
  - slice63: 44/63 = 0.6984
  - r_success: 357/400 = 0.8925
  - forget: 43/400 = 0.1075
  - low-prior slice: 62/139 = 0.4460
  - high-prior slice: 11/17 = 0.6471
- verdict for this point: fails the forget gate; KL `lambda=0.5` with replay 1:1 does not solve retention for main and is worse on forget than the pre-KL main replay1 upgrade.
- the new westc main replay1 queue automatically advanced to `lambda=1.0`; exactly one expected `lora_phase0.py` process is active there.
- the other three queues had not yet produced visible JSON at this check, but all active processes were alive and had completed val eval or were in eval tail.
- `44901` staging caveat: first copy script failed because `/usr/bin/time` was unavailable; copy was relaunched without it while pip install continued.

## Remote heartbeat 2026-07-02 13:41 CST

- no additional JSON completed beyond the already pulled main replay1 `lambda=0.5` point.
- new westc main replay1: queue advanced to `main_kl_r16_lr5e-5_ep2_replay1_lam1_seed20260703`; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB, 100%; non-finite skips at 6 during epoch 1 tail snapshot.
- old main replay2: still evaluating first `lambda=0.5` point; `eval:r-success` reached 250/400; exactly one active expected `lora_phase0.py`; GPU ~22.2/49.1 GB, 25%; skip count 22.
- old star replay1: still evaluating first `lambda=0.5` point; `eval:r-success` reached 250/400; exactly one active expected `lora_phase0.py`; GPU ~16.8/24.6 GB, 30%; skip count 12.
- new westb star replay2: still evaluating first `lambda=0.5` point; `eval:r-success` reached 100/400; exactly one active expected `lora_phase0.py`; GPU ~18.6/24.6 GB, 15%; skip count 30.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in the checked tails.
- ETA: the three remaining first-point JSONs should land shortly; replay1/replay2 queues remain healthy.

## Remote heartbeat 2026-07-02 13:46 CST

- new JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay2_lam0p5_seed20260703`
  - val: 67/156 = 0.4295
  - slice63: 41/63 = 0.6508
  - r_success: 370/400 = 0.9250
  - forget: 30/400 = 0.0750
  - low-prior slice: 55/139 = 0.3957
  - high-prior slice: 12/17 = 0.7059
- verdict for this point: still fails the forget gate by 3.75x, but replay 2:1 improves forget vs main replay1 `lambda=0.5` (0.0750 vs 0.1075) at a cost to val/slice63.
- new westc main replay1: currently on `lambda=1.0`; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB, 100%; skip count 10 by the snapshot.
- old main replay2: queue advanced to `lambda=1.0`; exactly one active expected `lora_phase0.py`; GPU ~17.0/49.1 GB, 100%.
- old star replay1: still evaluating first `lambda=0.5` point; `eval:r-success` reached 350/400; exactly one active expected `lora_phase0.py`; GPU ~16.8/24.6 GB.
- new westb star replay2: still evaluating first `lambda=0.5` point; `eval:r-success` reached 150/400; exactly one active expected `lora_phase0.py`; GPU ~18.6/24.6 GB.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: STaR replay1 first JSON should land next; STaR replay2 trails it. Both main queues are already on their second KL point.

## Remote heartbeat 2026-07-02 13:51 CST

- new JSON completed and was pulled locally:
  - `star_kl_r16_lr1e-4_ep2_replay1_lam0p5_seed20260703`
  - val: 63/156 = 0.4038
  - slice63: 34/63 = 0.5397
  - r_success: 363/400 = 0.9075
  - forget: 37/400 = 0.0925
  - low-prior slice: 52/139 = 0.3741
  - high-prior slice: 11/17 = 0.6471
- paired replay1 `lambda=0.5` comparison:
  - main: val 0.4679, slice63 0.6984, forget 0.1075.
  - STaR: val 0.4038, slice63 0.5397, forget 0.0925.
  - interpretation: main wins transfer strongly, especially slice63, but both fail the forget gate; STaR has lower forget at this point.
- new westc main replay1: still on `lambda=1.0`, now in eval (`eval:val 50/156`); exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB.
- old main replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~21.0/49.1 GB, 100%.
- old star replay1: queue advanced to `lambda=1.0`; exactly one expected `lora_phase0.py`; GPU ~15.8/24.6 GB, 100%.
- new westb star replay2: first `lambda=0.5` point still in r_success eval; reached 250/400 at check; exactly one expected `lora_phase0.py`; GPU ~18.6/24.6 GB.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: new westb STaR replay2 first JSON should land next; main replay1 `lambda=1.0` may also finish soon.

## Remote heartbeat 2026-07-02 13:57 CST

- no additional JSON completed at this check.
- new westc main replay1: `lambda=1.0` is in eval; `eval:r-success` has started after val completed; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB; skip count 16 for this point.
- old main replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB, 100%; skip count 11 in epoch 1 snapshot.
- old star replay1: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB, 97%; skip count 7 in early epoch 2 snapshot.
- new westb star replay2: first `lambda=0.5` point still in r_success eval; reached 350/400 at check; exactly one expected `lora_phase0.py`; GPU ~18.6/24.6 GB.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: STaR replay2 `lambda=0.5` should be the next JSON; main replay1 `lambda=1.0` follows close behind.

## Remote heartbeat 2026-07-02 14:02 CST

- new JSON completed and was pulled locally:
  - `star_kl_r16_lr1e-4_ep2_replay2_lam0p5_seed20260703`
  - val: 63/156 = 0.4038
  - slice63: 34/63 = 0.5397
  - r_success: 373/400 = 0.9325
  - forget: 27/400 = 0.0675
  - low-prior slice: 48/139 = 0.3453
  - high-prior slice: 15/17 = 0.8824
- paired STaR replay comparison at `lambda=0.5`:
  - replay1: val 0.4038, slice63 0.5397, forget 0.0925.
  - replay2: val 0.4038, slice63 0.5397, forget 0.0675.
  - interpretation: replay 2:1 gives STaR a clear forget improvement without moving aggregate val/slice63, but still fails the 0.02 forget gate by 3.4x.
- new westc main replay1: `lambda=1.0` still in r_success eval; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB.
- old main replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB, 100%; skip count 17 in epoch 2 snapshot.
- old star replay1: `lambda=1.0` now in eval (`eval:val 1/156`); exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB.
- new westb star replay2: queue advanced to `lambda=1.0`; exactly one expected `lora_phase0.py`; GPU ~17.4/24.6 GB, 87%.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: main replay1 `lambda=1.0` should be the next JSON; STaR replay1 `lambda=1.0` is already in eval.

## Remote heartbeat 2026-07-02 14:07 CST

- no additional JSON completed at this check.
- new westc main replay1: `lambda=1.0` in r_success eval; reached 300/400 at check; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB; skip count 16 for this point.
- old main replay2: `lambda=1.0` in eval; val eval has started after training; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB; skip count 22 for this point.
- old star replay1: `lambda=1.0` in val eval; reached 100/156 at check; exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB; skip count 12 for this point.
- new westb star replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~17.4/24.6 GB, 100%; skip count 11 in epoch 1 snapshot.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: main replay1 `lambda=1.0` remains the next expected JSON; old main replay2 and old star replay1 `lambda=1.0` are close behind.

## Remote heartbeat 2026-07-02 14:12 CST

- new JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay1_lam1_seed20260703`
  - val: 68/156 = 0.4359
  - slice63: 43/63 = 0.6825
  - r_success: 364/400 = 0.9100
  - forget: 36/400 = 0.0900
  - low-prior slice: 55/139 = 0.3957
  - high-prior slice: 13/17 = 0.7647
- main replay1 lambda trend so far:
  - `lambda=0.5`: val 0.4679, slice63 0.6984, forget 0.1075.
  - `lambda=1.0`: val 0.4359, slice63 0.6825, forget 0.0900.
  - interpretation: stronger KL modestly improves forget but loses transfer; still fails the forget gate by 4.5x.
- new westc main replay1: queue advanced to `lambda=2.0`; exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB, 98%; early skip count 4 at check.
- old main replay2: `lambda=1.0` in eval; val eval reached 50/156 at check; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB.
- old star replay1: `lambda=1.0` in eval; val eval reached 150/156 and r_success started at check; exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB.
- new westb star replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~18.6/24.6 GB, 100%; skip count 19 by early epoch 2 snapshot.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: old star replay1 `lambda=1.0` and old main replay2 `lambda=1.0` are the next expected JSONs.

## Remote heartbeat 2026-07-02 14:17 CST

- no additional JSON completed at this check.
- new westc main replay1: `lambda=2.0` training active; exactly one expected `lora_phase0.py`; GPU ~16.0/24.6 GB, 100%; skip count 9 in early epoch 2 snapshot.
- old main replay2: `lambda=1.0` in r_success eval; reached 150/400 at check; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB.
- old star replay1: `lambda=1.0` in r_success eval; reached 100/400 at check; exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB.
- new westb star replay2: `lambda=1.0` training active; exactly one expected `lora_phase0.py`; GPU ~18.6/24.6 GB, 90%; skip count 27 in epoch 2 snapshot.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: old main replay2 `lambda=1.0` and old star replay1 `lambda=1.0` remain the next expected JSONs.

## Remote heartbeat 2026-07-02 14:22 CST

- no additional JSON completed at this check.
- new westc main replay1: `lambda=2.0` in val eval (`eval:val 1/156` at check); exactly one active expected `lora_phase0.py`; GPU ~16.0/24.6 GB; skip count 16 for this point.
- old main replay2: `lambda=1.0` in r_success eval; reached 300/400 at check; exactly one expected `lora_phase0.py`; GPU ~22.2/49.1 GB.
- old star replay1: `lambda=1.0` in r_success eval; reached 200/400 at check; exactly one expected `lora_phase0.py`; GPU ~16.8/24.6 GB.
- new westb star replay2: `lambda=1.0` in val eval (`eval:val 1/156` at check); exactly one expected `lora_phase0.py`; GPU ~18.6/24.6 GB; skip count 30 for this point.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- ETA: old main replay2 `lambda=1.0` should land next, followed by old star replay1 `lambda=1.0`.

## Candidate server 44901 check 2026-07-02 14:23 CST

- `44901` is not part of the original four-server heartbeat queue, so it is monitored as a staging candidate rather than an active run host.
- hardware/env:
  - RTX 4080 SUPER, 32.8 GB, idle.
  - dependencies now present: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`.
- staging status:
  - model copy completed at 14:22 CST (`model.done` present).
  - project copy is still running; `repo.done` is not present yet.
  - current partial project size: 636 MB.
- decision: do not schedule a run on `44901` until the project copy finishes and integrity checks pass. Starting before `repo.done` risks a partial-data or partial-script run.

## Remote heartbeat and reschedule 2026-07-02 14:28 CST

- new JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay2_lam1_seed20260703`
  - val: 64/156 = 0.4103
  - slice63: 42/63 = 0.6667
  - r_success: 372/400 = 0.9300
  - forget: 28/400 = 0.0700
  - low-prior slice: 50/139 = 0.3597
  - high-prior slice: 14/17 = 0.8235
- main replay2 lambda trend so far:
  - `lambda=0.5`: val 0.4295, slice63 0.6508, forget 0.0750.
  - `lambda=1.0`: val 0.4103, slice63 0.6667, forget 0.0700.
  - interpretation: stronger KL slightly improves forget but still fails the forget gate by 3.5x.
- `44901` staging completed:
  - `model.done`, `pip.done`, `repo.done`, and `all.done` are present.
  - GPU idle before scheduling: RTX 4080 SUPER 32.8 GB.
- reschedule action:
  - STOPped only the new westc replay1 parent bash (`run_s05_main_replay1_new4.sh`) so it will not launch `main_ep3_r16_lr5e-5_replay1_seed20260703` after current `lambda=2.0`.
  - left the current new westc `lambda=2.0` Python child running; it continues normally.
  - launched `main_ep3_r16_lr5e-5_replay1_seed20260703` on `44901`.
  - confirmation: `44901` has one active expected `lora_phase0.py`, GPU ~12.1/32.8 GB, 100%; log shows model loaded and training started at epoch 1/3.
- active queue state after reschedule:
  - new westc: `main replay1 lambda=2.0` eval/running, parent bash STOPped to prevent ep3 duplication.
  - `44901`: `main replay1 ep3` running.
  - old main: advanced to `main replay2 lambda=2.0`.
  - old star: `star replay1 lambda=1.0` in r_success eval.
  - new westb: `star replay2 lambda=1.0` in eval.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.

## Remote heartbeat 2026-07-02 14:32 CST

- new JSON completed and was pulled locally:
  - `star_kl_r16_lr1e-4_ep2_replay1_lam1_seed20260703`
  - val: 56/156 = 0.3590
  - slice63: 29/63 = 0.4603
  - r_success: 369/400 = 0.9225
  - forget: 31/400 = 0.0775
  - low-prior slice: 45/139 = 0.3237
  - high-prior slice: 11/17 = 0.6471
- STaR replay1 lambda trend so far:
  - `lambda=0.5`: val 0.4038, slice63 0.5397, forget 0.0925.
  - `lambda=1.0`: val 0.3590, slice63 0.4603, forget 0.0775.
  - interpretation: stronger KL improves forget modestly but loses transfer; still fails the 0.02 forget gate by 3.9x.
- new westc main replay1: `lambda=2.0` in eval; parent bash remains STOPped to prevent duplicate main ep3; exactly one active expected `lora_phase0.py`.
- old main replay2: `lambda=2.0` training active; exactly one expected `lora_phase0.py`.
- old star replay1: queue advanced to `lambda=2.0`; exactly one expected `lora_phase0.py`.
- new westb star replay2: `lambda=1.0` in eval; exactly one expected `lora_phase0.py`.
- `44901`: `main_ep3_r16_lr5e-5_replay1_seed20260703` training active, GPU utilized.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.

## Remote heartbeat and reschedule 2026-07-02 14:39 CST

- no additional JSON completed at this check; local s05 JSON count remains 7.
- active queue state:
  - new westc: `main replay1 lambda=2.0` in r_success eval, reached 250/400; GPU ~16.0/24.6 GB.
  - old main: `main replay2 lambda=2.0` training active, epoch 1 step 50/112; GPU ~22.2/49.1 GB, ~98%.
  - old star: `star replay1 lambda=2.0` training active; GPU ~16.8/24.6 GB, ~98%.
  - new westb: `star replay2 lambda=1.0` in r_success eval, reached 100/400; GPU ~18.6/24.6 GB.
  - `44901`: `main_ep3_r16_lr5e-5_replay1_seed20260703` training active, epoch 2 step 50/111; GPU ~15.4/32.8 GB, ~98%.
- scheduling action:
  - STOPped the actual old-star replay1 queue bash as well as its wrapper, leaving the current `lambda=2.0` Python child running. This prevents duplicate `star_ep3_r16_lr1e-4_replay1_seed20260703`.
  - launched a wait-then-run `star_ep3_r16_lr1e-4_replay1_seed20260703` script on new westc. It waits for the current new-westc `main lambda=2.0` process to exit, then starts STaR ep3 if its JSON is still absent.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs. Non-finite skip guards are active and counts are in the same order as prior KL points.
- ETA after reschedule: next JSONs should be new-westc `main lambda=2.0` and new-westb `star replay2 lambda=1.0`; full s05 wave is estimated at roughly 45-65 minutes if no run stalls.

## Remote heartbeat 2026-07-02 14:42 CST

- no additional JSON completed at this check; local s05 JSON count remains 7.
- active queue state:
  - new westc: `main replay1 lambda=2.0` in r_success eval, reached 350/400; GPU ~16.0/24.6 GB. The `star_ep3` wait script is present and waiting for this process to exit.
  - old main: `main replay2 lambda=2.0` training active, epoch 2 step 75/112; GPU ~22.2/49.1 GB, 100%.
  - old star: `star replay1 lambda=2.0` training active, epoch 2 step 50/74; wrapper and queue bash are STOPped to prevent duplicate `star_ep3`, while the current Python child continues.
  - new westb: `star replay2 lambda=1.0` in r_success eval, reached 150/400.
  - `44901`: `main_ep3_r16_lr5e-5_replay1_seed20260703` training active, epoch 3 step 75/111; GPU ~15.4/32.8 GB, ~96%.
- error scan found no OOM, out-of-memory, traceback, CUDA error, RuntimeError, or ImportError lines in checked logs.
- non-finite skip guards continue to fire at expected low levels; no run has exited without JSON.
- ETA unchanged: next JSONs should be new-westc `main lambda=2.0` and new-westb `star replay2 lambda=1.0`; full s05 wave remains roughly 40-60 minutes if no run stalls.

## Remote heartbeat 2026-07-02 14:50 CST

- new JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay1_lam2_seed20260703`
  - val: 62/156 = 0.3974
  - slice63: 39/63 = 0.6190
  - r_success: 367/400 = 0.9175
  - forget: 33/400 = 0.0825
  - low-prior slice: 48/139 = 0.3453
  - high-prior slice: 14/17 = 0.8235
- main replay1 lambda trend:
  - `lambda=0.5`: val 0.4679, slice63 0.6984, forget 0.1075.
  - `lambda=1.0`: val 0.4359, slice63 0.6825, forget 0.0900.
  - `lambda=2.0`: val 0.3974, slice63 0.6190, forget 0.0825.
  - interpretation: stronger KL monotonically improves forget but monotonically erodes transfer; `lambda=2.0` still fails the 0.02 forget gate by 4.1x.
- scheduling state:
  - new westc: `main replay1 lambda=2.0` finished and the wait script correctly started `star_ep3_r16_lr1e-4_replay1_seed20260703`; exactly one active `lora_phase0.py`, GPU ~16.3/24.6 GB, 100%.
  - old main: `main replay2 lambda=2.0` saved adapter and entered val eval; exactly one active `lora_phase0.py`.
  - old star: `star replay1 lambda=2.0` saved adapter and entered val eval; wrapper and queue bash remain STOPped, preventing duplicate `star_ep3`.
  - new westb: `star replay2 lambda=1.0` in r_success eval, reached 250/400; `lambda=2.0` still pending in the queue.
  - `44901`: `main_ep3_r16_lr5e-5_replay1_seed20260703` trained and saved its adapter, but failed before writing JSON because the staged server lacked `../EDG-EXP1/results/a2_failures.json` and `../EDG-EXP2-struct`.
- recovery action for `44901`:
  - copied the missing EXP1/EXP2 eval dependencies to `/root/autodl-tmp`.
  - relaunched eval-only against the already saved `main_ep3` adapter with explicit `--failures` and `--exp2-root` paths plus `PYTHONPATH`.
  - confirmed the eval-only process is alive, model weights loaded, and val eval has started; no retraining was needed.
- error status:
  - the only new error was the `44901` missing-path traceback, now diagnosed and recovered.
  - no OOM, out-of-memory, CUDA error, RuntimeError from training, or ImportError remains active.
- ETA: because `new_westb` still has `star replay2 lambda=2.0` pending and `44901 main_ep3` is now eval-only, full s05 completion is roughly 45-65 minutes from this checkpoint if no further stalls occur.

## Remote heartbeat 2026-07-02 14:52 CST

- no additional JSON completed at this check; local s05 JSON count remains 8.
- active queue state:
  - new westc: `star_ep3_r16_lr1e-4_replay1_seed20260703` training active, epoch 3 step 100/111; GPU ~16.3/24.6 GB, 100%.
  - old main: `main replay2 lambda=2.0` saved adapter and entered eval; val eval completed and r_success started; exactly one active `lora_phase0.py`.
  - old star: `star replay1 lambda=2.0` saved adapter and entered eval; val eval reached 100/156; wrapper and queue bash remain STOPped to prevent duplicate `star_ep3`.
  - new westb: `star replay2 lambda=1.0` in r_success eval, reached 350/400; `lambda=2.0` still pending in queue.
  - `44901`: eval-only recovery for `main_ep3_r16_lr5e-5_replay1_seed20260703` is alive and reached val 50/156; GPU ~6.9/32.8 GB.
- error status:
  - grep still sees the old `44901` missing-path traceback in the original training log, but the recovery eval log is active and clean so far.
- no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- non-finite skip counts remain in the expected low range for the active training jobs.
- ETA: unchanged within noise; full s05 wave remains roughly 40-60 minutes, dominated by pending `star replay2 lambda=2.0` plus ep3 evals.

## User ETA check 2026-07-02 14:54 CST

- no additional JSON completed at this check.
- near-term completions:
  - new westb `star replay2 lambda=1.0`: r_success reached 350/400, should finish in minutes, then the same queue starts `lambda=2.0`.
  - old main `main replay2 lambda=2.0`: r_success reached 50/400 after val eval.
  - old star `star replay1 lambda=2.0`: val eval completed, r_success started.
  - new westc `star_ep3`: adapter saved, val eval started.
  - `44901` `main_ep3` eval-only recovery reached val 50/156.
- bottleneck: `star replay2 lambda=2.0` has not started yet and is expected to be roughly one full replay2 run after `lambda=1.0` completes.
- ETA: non-bottleneck jobs should mostly land within ~10-20 minutes; full s05 wave is estimated at ~55-70 minutes unless `star replay2 lambda=2.0` runs faster than prior replay2 points.

## User ETA check 2026-07-02 14:56 CST

- new JSON completed and was pulled locally:
  - `star_kl_r16_lr1e-4_ep2_replay2_lam1_seed20260703`
  - val: 54/156 = 0.3462
  - slice63: 29/63 = 0.4603
  - r_success: 373/400 = 0.9325
  - forget: 27/400 = 0.0675
  - low-prior slice: 41/139 = 0.2950
  - high-prior slice: 13/17 = 0.7647
- active queue state:
  - new westb: `star replay2 lambda=2.0` started at 14:54:54 CST and is in early training.
  - new westc: `star_ep3` saved adapter and entered eval; val eval reached 50/156.
  - old main: `main replay2 lambda=2.0` in r_success eval, reached 100/400.
  - old star: `star replay1 lambda=2.0` in r_success eval, reached 50/400.
  - `44901`: `main_ep3` eval-only recovery reached val 100/156.
- ETA: the only full remaining train+eval is `star replay2 lambda=2.0`; based on prior replay2 runtimes, full completion is now estimated at ~45-60 minutes, with the other eval-only/in-eval jobs expected earlier.

## Remote heartbeat 2026-07-02 14:58 CST

- no additional JSON completed at this check; local s05 JSON count remains 9.
- active queue state:
  - new westc: `star_ep3_r16_lr1e-4_replay1_seed20260703` in eval, val reached 100/156; exactly one active `lora_phase0.py`.
  - old main: `main replay2 lambda=2.0` in r_success eval, reached 150/400; exactly one active `lora_phase0.py`.
  - old star: `star replay1 lambda=2.0` in r_success eval, reached 100/400; wrapper and queue bash remain STOPped to prevent duplicate ep3.
  - new westb: `star replay2 lambda=2.0` active in early training, epoch 1 step 1/112 with 4 non-finite-loss skips; GPU ~17.4/24.6 GB, 100%.
  - `44901`: `main_ep3` eval-only recovery completed val eval and reached r_success 50/400.
- error status:
  - old `44901` missing-path traceback remains present in the original failed training log; recovery eval log has no new traceback and is active.
- no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- ETA: unchanged within noise; full s05 completion remains roughly 40-55 minutes, dominated by `star replay2 lambda=2.0`.

## Remote heartbeat 2026-07-02 15:04 CST

- no additional JSON completed at this check; local s05 JSON count remains 9.
- active queue state:
  - new westc: `star_ep3_r16_lr1e-4_replay1_seed20260703` in r_success eval, reached 100/400; exactly one active `lora_phase0.py`.
  - old main: `main replay2 lambda=2.0` in r_success eval, reached 350/400; exactly one active `lora_phase0.py`.
  - old star: `star replay1 lambda=2.0` in r_success eval, reached 150/400; wrapper and queue bash remain STOPped to prevent duplicate ep3.
  - new westb: `star replay2 lambda=2.0` active in training, epoch 1 step 50/112; GPU ~17.4/24.6 GB, 100%; non-finite-loss skip count 15 so far, consistent with prior replay2 KL points.
  - `44901`: `main_ep3` eval-only recovery in r_success eval, reached 200/400.
- error status:
  - old `44901` missing-path traceback remains present in the original failed training log only; recovery eval log remains active and clean.
- no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- ETA: non-bottleneck eval jobs should land soon; full s05 completion remains roughly 35-50 minutes, dominated by `star replay2 lambda=2.0`.

## Remote heartbeat 2026-07-02 15:09 CST

- new JSON completed and was pulled locally:
  - `main_kl_r16_lr5e-5_ep2_replay2_lam2_seed20260703`
  - val: 48/156 = 0.3077
  - slice63: 33/63 = 0.5238
  - r_success: 379/400 = 0.9475
  - forget: 21/400 = 0.0525
  - low-prior slice: 35/139 = 0.2518
  - high-prior slice: 13/17 = 0.7647
- main replay2 lambda trend:
  - `lambda=0.5`: val 0.4295, slice63 0.6508, forget 0.0750.
  - `lambda=1.0`: val 0.4103, slice63 0.6667, forget 0.0700.
  - `lambda=2.0`: val 0.3077, slice63 0.5238, forget 0.0525.
  - interpretation: strong KL finally buys meaningful forget reduction, but at a large transfer cost; still fails the 0.02 forget gate by 2.6x.
- active queue state:
  - old main: queue complete; GPU idle.
  - new westc: `star_ep3_r16_lr1e-4_replay1_seed20260703` in r_success eval, reached 250/400.
  - old star: `star replay1 lambda=2.0` in r_success eval, reached 250/400; wrapper and queue bash remain STOPped to prevent duplicate ep3.
  - new westb: `star replay2 lambda=2.0` active in training, epoch 2 step 75/112; GPU ~18.6/24.6 GB, ~99%; non-finite-loss skip count 22 so far, consistent with prior replay2 KL points.
  - `44901`: `main_ep3` eval-only recovery in r_success eval, reached 300/400.
- error status:
  - old `44901` missing-path traceback remains present in the original failed training log only; recovery eval log remains active.
  - no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- scheduling decision: old main is idle, but all remaining pre-registered work is already running; do not launch duplicates.
- ETA: full s05 completion is now roughly 25-40 minutes, dominated by the remaining train+eval tail of `star replay2 lambda=2.0`.

## Remote heartbeat 2026-07-02 15:14 CST

- three new JSONs completed and were pulled locally:
  - `main_ep3_r16_lr5e-5_replay1_seed20260703`
    - val: 79/156 = 0.5064
    - slice63: 45/63 = 0.7143
    - r_success: 366/400 = 0.9150
    - forget: 34/400 = 0.0850
    - low-prior slice: 67/139 = 0.4820
    - high-prior slice: 12/17 = 0.7059
  - `star_ep3_r16_lr1e-4_replay1_seed20260703`
    - val: 70/156 = 0.4487
    - slice63: 36/63 = 0.5714
    - r_success: 367/400 = 0.9175
    - forget: 33/400 = 0.0825
    - low-prior slice: 57/139 = 0.4101
    - high-prior slice: 13/17 = 0.7647
  - `star_kl_r16_lr1e-4_ep2_replay1_lam2_seed20260703`
    - val: 49/156 = 0.3141
    - slice63: 24/63 = 0.3810
    - r_success: 370/400 = 0.9250
    - forget: 30/400 = 0.0750
    - low-prior slice: 36/139 = 0.2590
    - high-prior slice: 13/17 = 0.7647
- ep3 observation:
  - main ep3 restores strong transfer (slice63 0.7143) but forget remains 0.0850.
  - STaR ep3 also fails the forget gate at 0.0825; no ep3 point is viable under the hard 0.02 rule.
- active queue state:
  - new westc: complete; GPU idle. The STOPped original main replay1 parent remains stopped and did not launch a duplicate.
  - old main: complete; GPU idle.
  - old star: complete; GPU idle. STOPped parent remains stopped and did not launch a duplicate ep3.
  - `44901`: main ep3 eval-only recovery complete; GPU idle.
  - new westb: `star replay2 lambda=2.0` trained and saved adapter; process still active and should enter eval next. It is the only remaining `lora_phase0.py`.
- error status:
  - old `44901` missing-path traceback remains present in the original failed training log only; recovery eval completed successfully.
  - no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- ETA: only `star replay2 lambda=2.0` remains, now past training and at the adapter-saved boundary; expected completion is roughly 10-20 minutes.

## Remote heartbeat 2026-07-02 15:19 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- active queue state:
  - new westc: complete; GPU idle. STOPped original replay1 parent remains stopped and no duplicate launched.
  - old main: complete; GPU idle.
  - old star: complete; GPU idle. STOPped parent remains stopped and no duplicate ep3 launched.
  - `44901`: main ep3 eval-only recovery complete; GPU idle.
  - new westb: only remaining active run is `star replay2 lambda=2.0`; training completed, adapter saved, eval started and reached val 50/156.
- error status:
  - old `44901` missing-path traceback remains present only in the original failed training log; recovery eval completed successfully.
  - no OOM, out-of-memory, CUDA error, RuntimeError from active training/eval, or ImportError is active.
- ETA: final JSON should land soon; estimated remaining time roughly 5-12 minutes, dominated by `star replay2 lambda=2.0` eval.

## Remote heartbeat 2026-07-02 15:24 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: training completed, adapter saved, eval active; val reached 100/156.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON expected in roughly 3-8 minutes.

## Remote heartbeat 2026-07-02 15:29 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: val eval completed, r_success eval reached 50/400.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON expected in roughly 3-6 minutes.

## Remote heartbeat 2026-07-02 15:34 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: val eval completed, r_success eval reached 150/400.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON expected soon, roughly one more short eval segment.

## Remote heartbeat 2026-07-02 15:39 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: val eval completed, r_success eval reached 200/400.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON is still pending; eval is slow but progressing normally.

## Remote heartbeat 2026-07-02 15:44 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: val eval completed, r_success eval reached 300/400.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON should land after the remaining 100 r_success examples.

## Remote heartbeat 2026-07-02 15:49 CST

- no additional JSON completed at this check; local s05 JSON count remains 13.
- only remaining active run: `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`.
- progress: val eval completed, r_success eval reached 350/400.
- error status: no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or new traceback in the active log.
- ETA: final JSON should land after the remaining 50 r_success examples.

## Remote heartbeat 2026-07-02 15:54 CST

- final JSON completed and was pulled locally:
  - `star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703`
  - val: 40/156 = 0.2564
  - slice63: 23/63 = 0.3651
  - r_success: 377/400 = 0.9425
  - forget: 23/400 = 0.0575
  - low-prior slice: 26/139 = 0.1871
  - high-prior slice: 14/17 = 0.8235
- s05 BLOCKED run complete: 14/14 JSONs are present locally.
- mechanical exit check:
  - no point satisfies `forget <= 0.02`.
  - therefore no point can satisfy the success criterion `forget <= 0.02 and slice63 >= 0.40`.
  - best forget observed: `main_kl_r16_lr5e-5_ep2_replay2_lam2_seed20260703`, forget 21/400 = 0.0525, slice63 33/63 = 0.5238.
- error status: active queues complete; no OOM, out-of-memory, CUDA error, RuntimeError, ImportError, or active traceback in the final run.
