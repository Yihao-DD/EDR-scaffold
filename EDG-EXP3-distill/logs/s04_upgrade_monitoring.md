# Step 0.4 Upgrade and Probe Monitoring

Status: post-clean-grid upgrade/probe monitoring. These jobs follow the preregistered full-grid failure branch and do not change the heldout discipline.

## Local setup 2026-07-02

- Clean grid decision: `24/24` JSONs completed; no point passes `r_success >= 0.98`.
- Fallback record point: `main_r8_lr2e-5_ep1`.
- Replay-1:1 datasets rebuilt without overwriting the clean-grid datasets:
  - `data/distill_main_replay1.jsonl`: 296 rows, 148 replay rows.
  - `data/distill_star_replay1.jsonl`: 296 rows, 148 replay rows.
- `CHANGELOG.md` v1.10 records the mechanical decision, upgrade configs, C2 full-grid shape, epoch/lr reversal, Pareto gap, and BLOCKED continuation.
- KL-anchor support was added to `scripts/lora_phase0.py` behind default-off flags: `--kl-anchor-lambda 0.0` keeps the current SFT recipe unchanged.
- Added pure-forward helper scripts:
  - `scripts/heldout_pass16.py`
  - `scripts/logprob_probe.py`
- Local syntax check passed for modified/new scripts via `py_compile`.
- Local pytest could not be run because the local system Python lacks `pytest`.

## Remote launch 2026-07-02 12:08 CST

- new westc: launched `main_upgrade_r16_lr2p5e-5_ep2_replay1_seed20260703`.
  - config: `rank=16`, `lr=2.5e-5`, `epochs=2`, `replay=1:1`.
  - dataset: `data/distill_main_replay1.jsonl`.
  - output: `results/s04_upgrade/main/main_upgrade_r16_lr2p5e-5_ep2_replay1_seed20260703.json`.
- old star: launched `star_upgrade_r16_lr5e-5_ep2_replay1_seed20260703`.
  - config: `rank=16`, `lr=5e-5`, `epochs=2`, `replay=1:1`.
  - dataset: `data/distill_star_replay1.jsonl`.
  - output: `results/s04_upgrade/star/star_upgrade_r16_lr5e-5_ep2_replay1_seed20260703.json`.
- old main: launched heldout pass@16 partition labeling.
  - output: `results/s01_heldout_pass16_partition.json`.
- new westb: launched base log-probability probe.
  - output: `results/s04_logprob_probe.json`.

## Remote status 2026-07-02 12:13 CST

- new westc main upgrade: active `lora_phase0.py`; GPU 15601/24564 MB, 100% util; training has reached at least step 25/74; no JSON yet.
- old star upgrade: active `lora_phase0.py`; GPU 16378/24564 MB, 100% util; training has reached at least step 25/74; no JSON yet.
- old main heldout pass@16: active `heldout_pass16.py`; GPU 22263/49140 MB, 96% util; progress 50/158; no JSON yet.
- new westb log-probability probe: completed and pulled locally.

## Log-Probability Probe Result

- target_count: 569.
- caveat: 14 `R_success_eval` records had no effective scored output token after prompt/target truncation at `max_length=2048`; `R_success` summaries therefore use `n=386`.
- train teacher scaffold-only mean token log-prob: `-1.1114` (`n=64`).
- train teacher sampling-rescuable mean token log-prob: `-0.9808` (`n=10`).
- train base pass@16 success mean token log-prob: `-1.0298` (`n=32`).
- `R_success_eval` first-point shared fragile mean token log-prob: `-1.1728` (`n=24`).
- `R_success_eval` non-first-shared mean token log-prob: `-1.1376` (`n=362`).
- Interpretation: C2-side base-prior prediction is directionally supported; H-margin is mixed and remains `PROBE`.

## Automation

- Created heartbeat monitor `edr-s04-upgrade-monitor` at 5-minute cadence for the two upgrade jobs, heldout pass@16, and completed/pulled probe artifacts.

## Remote heartbeat 2026-07-02 12:24 CST

- new westc main upgrade: active `lora_phase0.py`; GPU 15603/24564 MB, eval-phase utilization around 26%; validation eval completed and `r_success` eval reached 50/400; no JSON yet.
- main upgrade caveat: training log contains one `loss=nan` at epoch 2/2 step 50/74, followed by a finite logged loss at step 74/74 and successful adapter save. Keep this checkpoint caveated until the JSON and any sanity checks are read.
- old star upgrade: active `lora_phase0.py`; GPU 16378/24564 MB, eval-phase utilization around 17%; validation eval reached 100/156; no JSON yet. No runtime errors or NaN found in the checked log.
- old main heldout pass@16: completed and pulled locally.
  - total heldout failures: 158.
  - teacher_success: 50/158 = 0.3165.
  - scaffold_only: 47/158 = 0.2975.
  - sampling_rescuable: 10/158 = 0.0633.
  - neither: 101/158 = 0.6392.
- new westb log-prob probe: already completed and pulled; GPU idle.
- no duplicate conflicting runs found in this heartbeat.

## Remote heartbeat 2026-07-02 12:29 CST

- new westc main upgrade: active `lora_phase0.py`; GPU 15603/24564 MB, eval-phase utilization around 25%; `r_success` eval reached 150/400; no JSON yet.
- main upgrade caveat remains: one logged `loss=nan` at epoch 2/2 step 50/74, followed by finite loss at step 74/74 and adapter save. No additional runtime error surfaced in this heartbeat.
- old star upgrade: active `lora_phase0.py`; GPU 16378/24564 MB, eval-phase utilization around 15%; validation eval completed and `r_success` eval reached 50/400; no JSON yet.
- old main heldout pass@16: completed and already pulled locally.
- new westb log-prob probe: completed and already pulled locally.
- no duplicate conflicting runs found in this heartbeat.

## Remote heartbeat 2026-07-02 12:32 CST

- new westc main upgrade: active `lora_phase0.py`; GPU 15603/24564 MB, eval-phase utilization around 26%; `r_success` eval reached 250/400; no JSON yet.
- main upgrade caveat unchanged: one logged `loss=nan` at epoch 2/2 step 50/74, followed by finite loss at step 74/74 and adapter save. No new runtime error surfaced.
- old star upgrade: active `lora_phase0.py`; GPU 16378/24564 MB, eval-phase utilization around 16%; `r_success` eval reached 100/400; no JSON yet.
- old main heldout pass@16: completed and already pulled locally; GPU idle.
- new westb log-prob probe: completed and already pulled locally; GPU idle.
- no duplicate conflicting runs found in this heartbeat.

## Remote heartbeat 2026-07-02 12:38 CST

- new westc main upgrade: completed; JSON pulled locally.
  - `val_repair`: 73/156 = 0.4679.
  - `r_success`: 366/400 = 0.9150; `forget`: 34/400 = 0.0850.
  - slice63 teacher-repaired: 43/63 = 0.6825.
  - slice93 teacher-unrepaired: 30/93 = 0.3226.
  - slice63 scaffold-only: 38/55 = 0.6909; slice63 sampling-rescuable: 5/8 = 0.6250.
  - slice93 neither: 25/84 = 0.2976; slice93 sampling-rescuable: 5/9 = 0.5556.
  - low-prior combined: 63/139 = 0.4532; high-prior combined: 10/17 = 0.5882.
  - caveat: training log contains one `loss=nan` at epoch 2/2 step 50/74, followed by finite loss at step 74/74, successful adapter save, and completed eval. Treat this checkpoint as caveated.
- old star upgrade: still active in `r_success` eval tail; last checked progress reached at least 200/400, GPU memory ~16.4 GB, utilization eval-phase ~16%; no JSON yet and no OOM/import/CUDA/NAN error surfaced.
- old main heldout pass@16: completed and already pulled; GPU idle.
- new westb log-prob probe: completed and already pulled; GPU idle.
- no duplicate conflicting training runs found.

## Remote heartbeat 2026-07-02 12:42 CST

- old star upgrade: still active in `r_success` eval tail; progress reached 250/400, GPU memory ~16.4 GB, utilization eval-phase ~15%; no JSON yet and no OOM/import/CUDA/NAN error surfaced.
- new westc main upgrade: completed and already pulled locally; GPU idle.
- old main heldout pass@16: completed and already pulled locally; GPU idle.
- new westb log-prob probe: completed and already pulled locally; GPU idle.
- no duplicate conflicting training runs found.

## Remote heartbeat 2026-07-02 12:47-12:49 CST

- old star upgrade: completed at 12:48 CST; JSON pulled locally.
  - `val_repair`: 70/156 = 0.4487.
  - `r_success`: 360/400 = 0.9000; `forget`: 40/400 = 0.1000.
  - slice63 teacher-repaired: 39/63 = 0.6190.
  - slice93 teacher-unrepaired: 31/93 = 0.3333.
  - slice63 scaffold-only: 33/55 = 0.6000; slice63 sampling-rescuable: 6/8 = 0.7500.
  - slice93 neither: 24/84 = 0.2857; slice93 sampling-rescuable: 7/9 = 0.7778.
  - low-prior combined: 57/139 = 0.4101; high-prior combined: 13/17 = 0.7647.
  - no OOM/import/CUDA/NAN/runtime error surfaced in checked logs.
- new westc main upgrade: completed and already pulled locally; GPU idle.
- old main heldout pass@16: completed and already pulled locally; GPU idle.
- new westb log-prob probe: completed and already pulled locally; GPU idle.
- all four s04 jobs are now complete; all GPUs are idle; no duplicate conflicting training runs found.
