# Step 0.3 Clean Grid Monitoring

Status: monitoring-only. These single-checkpoint reads are not model-selection decisions and do not change the preregistered clean-grid selection rule.

## main_r8_lr2e-5_ep1_seed20260703

- arm: main
- rank: 8
- learning_rate: 2e-5
- epochs: 1
- val_repair@156: 20/156 = 0.1282
- val_teacher_repaired_63: 13/63 = 0.2063
- val_teacher_unrepaired_93: 7/93 = 0.0753
- r_success: 374/400 = 0.9350
- forget: 0.0650
- note: this is the first conservative clean point and is reported only as an early retention/repair monitor.

## star_r8_lr2e-5_ep1_seed20260703

- arm: star
- rank: 8
- learning_rate: 2e-5
- epochs: 1
- val_repair@156: 19/156 = 0.1218
- val_teacher_repaired_63: 11/63 = 0.1746
- val_teacher_unrepaired_93: 8/93 = 0.0860
- r_success: 373/400 = 0.9325
- forget: 0.0675
- note: this is the matched first conservative clean point and is reported only as an early retention/repair monitor.

## main_r8_lr2e-5_ep2_seed20260703

- arm: main
- rank: 8
- learning_rate: 2e-5
- epochs: 2
- val_repair@156: 55/156 = 0.3526
- slice63_scaffold_only: 28/55 = 0.5091
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 16/84 = 0.1905
- slice93_sampling_rescuable: 6/9 = 0.6667
- low_prior_combined: 44/139 = 0.3165
- high_prior_combined: 11/17 = 0.6471
- r_success: 343/400 = 0.8575
- forget: 0.1425
- note: monitoring-only dose increment; not a model-selection decision.

## star_r8_lr2e-5_ep2_seed20260703

- arm: star
- rank: 8
- learning_rate: 2e-5
- epochs: 2
- val_repair@156: 47/156 = 0.3013
- slice63_scaffold_only: 23/55 = 0.4182
- slice63_sampling_rescuable: 3/8 = 0.3750
- slice93_neither: 14/84 = 0.1667
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 37/139 = 0.2662
- high_prior_combined: 10/17 = 0.5882
- r_success: 346/400 = 0.8650
- forget: 0.1350
- note: monitoring-only dose increment; not a model-selection decision.

## r8_lr2e-5_ep2 paired monitoring follow-up

Status: zero-GPU analysis, monitoring-only.

- main `ep1` repaired set is not nested in `ep2`: 20 -> 55 repaired, 9 first-point repairs lost, 44 second-point repairs gained.
- STaR `ep1` repaired set is not nested in `ep2`: 19 -> 47 repaired, 10 first-point repairs lost, 38 second-point repairs gained.
- first-point shared fragile 24 carryover into `ep2` newly-wrong: main 14/24, STaR 13/24.
- `ep2` newly-wrong overlap: main 57, STaR 54, intersection 45, union 66, Jaccard 0.6818.
- `ep2` paired type+subtype agreement on shared newly-wrong episodes: 43/45.
- low-prior validation 2x2 (`n=139`): both 33, main-only 11, STaR-only 4, neither 91.
- scaffold-only validation 2x2 (`n=55`): both 21, main-only 7, STaR-only 2, neither 25.
- all validation 2x2 (`n=156`): both 42, main-only 13, STaR-only 5, neither 96.

Interpretation: the second point supports a dose-order waterline view at aggregate level but not literal set nesting. C2-shaped separation appears in the predicted low-prior/scaffold-only region, while substantial shared repair and shared damage show that content-agnostic movement remains large.

## main_r8_lr5e-5_ep1_seed20260703

- arm: main
- rank: 8
- learning_rate: 5e-5
- epochs: 1
- val_repair@156: 63/156 = 0.4038
- val_teacher_repaired_63: 37/63 = 0.5873
- val_teacher_unrepaired_93: 26/93 = 0.2796
- slice63_scaffold_only: 31/55 = 0.5636
- slice63_sampling_rescuable: 6/8 = 0.7500
- slice93_neither: 18/84 = 0.2143
- slice93_sampling_rescuable: 8/9 = 0.8889
- low_prior_combined: 49/139 = 0.3525
- high_prior_combined: 14/17 = 0.8235
- r_success: 349/400 = 0.8725
- forget: 0.1275
- note: monitoring-only higher-lr point; it reaches the Gate 0 usable retention band on the validation preview while still breaking the forgetting gate by more than 6x.

## main_r8_lr5e-5_ep2_seed20260703

- arm: main
- rank: 8
- learning_rate: 5e-5
- epochs: 2
- val_repair@156: 77/156 = 0.4936
- val_teacher_repaired_63: 42/63 = 0.6667
- val_teacher_unrepaired_93: 35/93 = 0.3763
- slice63_scaffold_only: 37/55 = 0.6727
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 29/84 = 0.3452
- slice93_sampling_rescuable: 6/9 = 0.6667
- low_prior_combined: 66/139 = 0.4748
- high_prior_combined: 11/17 = 0.6471
- r_success: 361/400 = 0.9025
- forget: 0.0975
- note: monitoring-only; stronger retention preview than `r8_lr5e-5_ep1`, but still far above the forgetting gate.

## main_r8_lr1e-4_ep1_seed20260703

- arm: main
- rank: 8
- learning_rate: 1e-4
- epochs: 1
- val_repair@156: 77/156 = 0.4936
- val_teacher_repaired_63: 41/63 = 0.6508
- val_teacher_unrepaired_93: 36/93 = 0.3871
- slice63_scaffold_only: 36/55 = 0.6545
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 29/84 = 0.3452
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 65/139 = 0.4676
- high_prior_combined: 12/17 = 0.7059
- r_success: 343/400 = 0.8575
- forget: 0.1425
- note: monitoring-only; higher lr reaches similar aggregate repair to `r8_lr5e-5_ep2` with worse retention.

## star_r8_lr5e-5_ep1_seed20260703

- arm: STaR
- rank: 8
- learning_rate: 5e-5
- epochs: 1
- val_repair@156: 52/156 = 0.3333
- val_teacher_repaired_63: 29/63 = 0.4603
- val_teacher_unrepaired_93: 23/93 = 0.2473
- slice63_scaffold_only: 24/55 = 0.4364
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 17/84 = 0.2024
- slice93_sampling_rescuable: 6/9 = 0.6667
- low_prior_combined: 41/139 = 0.2950
- high_prior_combined: 11/17 = 0.6471
- r_success: 343/400 = 0.8575
- forget: 0.1425
- note: monitoring-only.

## star_r8_lr5e-5_ep2_seed20260703

- arm: STaR
- rank: 8
- learning_rate: 5e-5
- epochs: 2
- val_repair@156: 66/156 = 0.4231
- val_teacher_repaired_63: 35/63 = 0.5556
- val_teacher_unrepaired_93: 31/93 = 0.3333
- slice63_scaffold_only: 30/55 = 0.5455
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 23/84 = 0.2738
- slice93_sampling_rescuable: 8/9 = 0.8889
- low_prior_combined: 53/139 = 0.3813
- high_prior_combined: 13/17 = 0.7647
- r_success: 343/400 = 0.8575
- forget: 0.1425
- note: monitoring-only; old star parent loop is paused so the remaining `lr1e-4` STaR rank8 points are handled by the old-main extra queue.

## main_r16_lr2e-5_ep1_seed20260703

- arm: main
- rank: 16
- learning_rate: 2e-5
- epochs: 1
- val_repair@156: 47/156 = 0.3013
- val_teacher_repaired_63: 25/63 = 0.3968
- val_teacher_unrepaired_93: 22/93 = 0.2366
- slice63_scaffold_only: 21/55 = 0.3818
- slice63_sampling_rescuable: 4/8 = 0.5000
- slice93_neither: 15/84 = 0.1786
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 36/139 = 0.2590
- high_prior_combined: 11/17 = 0.6471
- r_success: 336/400 = 0.8400
- forget: 0.1600
- note: monitoring-only; not a model-selection decision.

## star_r16_lr2e-5_ep1_seed20260703

- arm: STaR
- rank: 16
- learning_rate: 2e-5
- epochs: 1
- val_repair@156: 42/156 = 0.2692
- val_teacher_repaired_63: 23/63 = 0.3651
- val_teacher_unrepaired_93: 19/93 = 0.2043
- slice63_scaffold_only: 21/55 = 0.3818
- slice63_sampling_rescuable: 2/8 = 0.2500
- slice93_neither: 11/84 = 0.1310
- slice93_sampling_rescuable: 8/9 = 0.8889
- low_prior_combined: 32/139 = 0.2302
- high_prior_combined: 10/17 = 0.5882
- r_success: 348/400 = 0.8700
- forget: 0.1300
- note: monitoring-only; not a model-selection decision.

## star_r16_lr2e-5_ep2_seed20260703

- arm: STaR
- rank: 16
- learning_rate: 2e-5
- epochs: 2
- val_repair@156: 55/156 = 0.3526
- val_teacher_repaired_63: 30/63 = 0.4762
- val_teacher_unrepaired_93: 25/93 = 0.2688
- slice63_scaffold_only: 25/55 = 0.4545
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 18/84 = 0.2143
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 43/139 = 0.3094
- high_prior_combined: 12/17 = 0.7059
- r_success: 348/400 = 0.8700
- forget: 0.1300
- note: monitoring-only; this is the final legitimate new3 output before the STaR rank16 tail was repartitioned.

## main_r16_lr2e-5_ep2_seed20260703

- arm: main
- rank: 16
- learning_rate: 2e-5
- epochs: 2
- val_repair@156: 67/156 = 0.4295
- val_teacher_repaired_63: 36/63 = 0.5714
- val_teacher_unrepaired_93: 31/93 = 0.3333
- slice63_scaffold_only: 31/55 = 0.5636
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 24/84 = 0.2857
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 55/139 = 0.3957
- high_prior_combined: 12/17 = 0.7059
- r_success: 353/400 = 0.8825
- forget: 0.1175
- note: monitoring-only; not a model-selection decision.

## star_r8_lr1e-4_ep1_seed20260703

- arm: STaR
- rank: 8
- learning_rate: 1e-4
- epochs: 1
- val_repair@156: 65/156 = 0.4167
- val_teacher_repaired_63: 32/63 = 0.5079
- val_teacher_unrepaired_93: 33/93 = 0.3548
- slice63_scaffold_only: 27/55 = 0.4909
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 26/84 = 0.3095
- slice93_sampling_rescuable: 7/9 = 0.7778
- low_prior_combined: 53/139 = 0.3813
- high_prior_combined: 12/17 = 0.7059
- r_success: 340/400 = 0.8500
- forget: 0.1500
- note: monitoring-only; not a model-selection decision.

## main_r8_lr1e-4_ep2_seed20260703

- arm: main
- rank: 8
- learning_rate: 1e-4
- epochs: 2
- val_repair@156: 83/156 = 0.5321
- val_teacher_repaired_63: 47/63 = 0.7460
- val_teacher_unrepaired_93: 36/93 = 0.3871
- slice63_scaffold_only: 42/55 = 0.7636
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 31/84 = 0.3690
- slice93_sampling_rescuable: 5/9 = 0.5556
- low_prior_combined: 73/139 = 0.5252
- high_prior_combined: 10/17 = 0.5882
- r_success: 358/400 = 0.8950
- forget: 0.1050
- note: monitoring-only; not a model-selection decision.

## star_r8_lr1e-4_ep2_seed20260703

- arm: STaR
- rank: 8
- learning_rate: 1e-4
- epochs: 2
- val_repair@156: 70/156 = 0.4487
- val_teacher_repaired_63: 38/63 = 0.6032
- val_teacher_unrepaired_93: 32/93 = 0.3441
- slice63_scaffold_only: 33/55 = 0.6000
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 26/84 = 0.3095
- slice93_sampling_rescuable: 6/9 = 0.6667
- low_prior_combined: 59/139 = 0.4245
- high_prior_combined: 11/17 = 0.6471
- r_success: 348/400 = 0.8700
- forget: 0.1300
- note: monitoring-only; not a model-selection decision.

## main_r16_lr5e-5_ep1_seed20260703

- arm: main
- rank: 16
- learning_rate: 5e-5
- epochs: 1
- val_repair@156: 72/156 = 0.4615
- val_teacher_repaired_63: 39/63 = 0.6190
- val_teacher_unrepaired_93: 33/93 = 0.3548
- slice63_scaffold_only: 34/55 = 0.6182
- slice63_sampling_rescuable: 5/8 = 0.6250
- slice93_neither: 27/84 = 0.3214
- slice93_sampling_rescuable: 6/9 = 0.6667
- low_prior_combined: 61/139 = 0.4388
- high_prior_combined: 11/17 = 0.6471
- r_success: 338/400 = 0.8450
- forget: 0.1550
- note: monitoring-only; not a model-selection decision.

## Remote monitor snapshot 2026-07-02 09:22 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicating the STaR `lr1e-4` tail now running on old main.
- old main server: two active jobs (`main_r8_lr1e-4_ep2`, `star_r8_lr1e-4_ep2`), GPU 28.2/49.1 GB, 100% util; no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, GPU 17.9/24.6 GB, eval-phase low util; no recent OOM/import/cuda errors found.
- new westc server: `main_r16_lr2e-5_ep2` completed, active `main_r16_lr5e-5_ep1`, GPU 16.6/24.6 GB, 99% util; no recent OOM/import/cuda errors found.

## Remote monitor snapshot 2026-07-02 09:25 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicate STaR tail work.
- old main server: `main_r8_lr1e-4_ep2` completed and was pulled locally; active `star_r8_lr1e-4_ep2`, GPU 18.2/49.1 GB, 100% util; no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, GPU 17.9/24.6 GB, eval-phase low util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep1`, GPU 19.0/24.6 GB, eval/train-transition util; no recent OOM/import/cuda errors found.

## Remote monitor snapshot 2026-07-02 09:32 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicate STaR tail work.
- old main server: active `star_r8_lr1e-4_ep2`, GPU 18.2/49.1 GB, 26% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, GPU 17.9/24.6 GB, 14% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep1`, GPU 19.0/24.6 GB, 27% util; eval-phase low util, no recent OOM/import/cuda errors found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 09:37 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicate STaR tail work.
- old main server: active `star_r8_lr1e-4_ep2`, GPU 18.2/49.1 GB, 27% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, GPU 17.9/24.6 GB, 13% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep1`, GPU 19.0/24.6 GB, 85% util; no recent OOM/import/cuda errors found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 09:42 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicate STaR tail work.
- old main server: active `star_r8_lr1e-4_ep2`, GPU 18.2/49.1 GB, 25% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, GPU 17.9/24.6 GB, 13% util; eval-phase low util, no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep1`, GPU 19.0/24.6 GB, 25% util; eval-phase low util, no recent OOM/import/cuda errors found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 09:44 CST

- old main server: active `star_r8_lr1e-4_ep2`, currently in `r_success` eval at 250/400; GPU 18.2/49.1 GB, 25% util; no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, currently in `r_success` eval at 250/400; GPU 17.9/24.6 GB, 14% util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep1`, currently in `r_success` eval at 300/400; GPU 19.0/24.6 GB, 73% util; no recent OOM/import/cuda errors found.
- ETA note: current queued bottleneck is STaR rank16 tail on new westb.

## Remote monitor snapshot 2026-07-02 09:47 CST

- old star server: no active training process; GPU idle by design because its launcher is paused to avoid duplicate STaR tail work.
- old main server: `star_r8_lr1e-4_ep2` completed and was pulled locally; GPU idle, no recent OOM/import/cuda errors found.
- new westb server: active `star_r16_lr2e-5_ep2`, currently in `r_success` eval at 300/400; GPU 17.9/24.6 GB, 13% util; no recent OOM/import/cuda errors found.
- new westc server: `main_r16_lr5e-5_ep1` completed and was pulled locally; active `main_r16_lr5e-5_ep2`, GPU 19.0/24.6 GB, 100% util; no recent OOM/import/cuda errors found.
- note: old main 49 GB GPU is now free; no new work was launched during this monitor tick.

## Remote scheduling update 2026-07-02 09:55 CST

- `star_r16_lr2e-5_ep2` completed on new westb and was pulled locally.
- new westb STaR rank16 parent runner was stopped, then killed, after a short duplicate `star_r16_lr5e-5_ep1` child was detected and terminated. No duplicate JSON was produced there.
- old star server now owns `star_r16_lr5e-5_ep1` and `star_r16_lr5e-5_ep2`; active process confirmed at 15.4/24.6 GB and training.
- old main server now owns `star_r16_lr1e-4_ep1` and `star_r16_lr1e-4_ep2`; active process confirmed at 15.5/49.1 GB and training.
- new westc continues the original main rank16 queue, currently `main_r16_lr5e-5_ep2`.
- effective active GPUs after reschedule: old star, old main, new westc. New westb is intentionally idle after producing its final assigned JSON.

## Remote monitor snapshot 2026-07-02 09:57 CST

- old star server: active `star_r16_lr5e-5_ep1`; training complete and eval started, GPU 17.8/24.6 GB, 17% util; no recent OOM/import/cuda errors found.
- old main server: active `star_r16_lr1e-4_ep1`; training complete and eval at `val` 50/156, GPU 17.9/49.1 GB, 25% util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep2`; training complete and eval at `val` 100/156, GPU 19.0/24.6 GB, 26% util; no recent OOM/import/cuda errors found.
- new westb server: intentionally idle after `star_r16_lr2e-5_ep2`; no residual runner or duplicate process found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 10:02 CST

- old star server: active `star_r16_lr5e-5_ep1`; eval at `val` 100/156, GPU 17.8/24.6 GB, 16% util; no recent OOM/import/cuda errors found.
- old main server: active `star_r16_lr1e-4_ep1`; eval at `r_success` 1/400, GPU 17.9/49.1 GB, 25% util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep2`; eval at `r_success` 100/400, GPU 19.0/24.6 GB, 25% util; no recent OOM/import/cuda errors found.
- new westb server: intentionally idle after `star_r16_lr2e-5_ep2`; no residual runner or duplicate process found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 10:07 CST

- old star server: active `star_r16_lr5e-5_ep1`; eval at `r_success` 1/400, GPU 17.8/24.6 GB, 17% util; no recent OOM/import/cuda errors found.
- old main server: active `star_r16_lr1e-4_ep1`; eval at `r_success` 150/400, GPU 17.9/49.1 GB, 26% util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep2`; eval at `r_success` 200/400, GPU 19.0/24.6 GB, 26% util; no recent OOM/import/cuda errors found.
- new westb server: intentionally idle after `star_r16_lr2e-5_ep2`; no residual runner or duplicate process found.
- no new JSONs since the previous snapshot.

## Remote monitor snapshot 2026-07-02 10:12 CST

- old star server: active `star_r16_lr5e-5_ep1`; eval at `r_success` 100/400, GPU 17.8/24.6 GB, 15% util; no recent OOM/import/cuda errors found.
- old main server: active `star_r16_lr1e-4_ep1`; eval at `r_success` 250/400, GPU 17.9/49.1 GB, 26% util; no recent OOM/import/cuda errors found.
- new westc server: active `main_r16_lr5e-5_ep2`; eval at `r_success` 350/400, GPU 19.0/24.6 GB, 25% util; no recent OOM/import/cuda errors found.
- new westb server: intentionally idle after `star_r16_lr2e-5_ep2`; no residual runner or duplicate process found.
- no new JSONs since the previous snapshot.

## Remote completion snapshot 2026-07-02 11:50 CST

- all four remote servers have no active `lora_phase0.py` or rank16 launcher processes.
- GPUs are idle: old star 1/24564 MB, old main 0/49140 MB, new westc 1/24564 MB, new westb 1/24564 MB.
- rank16 tail JSONs were completed and pulled locally; clean grid now has 24/24 JSONs.
- no duplicate conflicting JSONs were found. The broad grep surfaced only JSON `parse_error: null` fields, not runtime OOM/import/CUDA failures.
- pre-registered `r_success >= 0.98` gate: no clean-grid point passes.

Full clean-grid watch table:

| config | val | slice63 | scaffold-only | low-prior | r_success | forget |
|---|---:|---:|---:|---:|---:|---:|
| `main_r8_lr2e-5_ep1` | 20/156 = 0.1282 | 13/63 = 0.2063 | 7/55 = 0.1273 | 10/139 = 0.0719 | 374/400 = 0.9350 | 0.0650 |
| `main_r8_lr2e-5_ep2` | 55/156 = 0.3526 | 33/63 = 0.5238 | 28/55 = 0.5091 | 44/139 = 0.3165 | 343/400 = 0.8575 | 0.1425 |
| `main_r8_lr5e-5_ep1` | 63/156 = 0.4038 | 37/63 = 0.5873 | 31/55 = 0.5636 | 49/139 = 0.3525 | 349/400 = 0.8725 | 0.1275 |
| `main_r8_lr5e-5_ep2` | 77/156 = 0.4936 | 42/63 = 0.6667 | 37/55 = 0.6727 | 66/139 = 0.4748 | 361/400 = 0.9025 | 0.0975 |
| `main_r8_lr1e-4_ep1` | 77/156 = 0.4936 | 41/63 = 0.6508 | 36/55 = 0.6545 | 65/139 = 0.4676 | 343/400 = 0.8575 | 0.1425 |
| `main_r8_lr1e-4_ep2` | 83/156 = 0.5321 | 47/63 = 0.7460 | 42/55 = 0.7636 | 73/139 = 0.5252 | 358/400 = 0.8950 | 0.1050 |
| `main_r16_lr2e-5_ep1` | 47/156 = 0.3013 | 25/63 = 0.3968 | 21/55 = 0.3818 | 36/139 = 0.2590 | 336/400 = 0.8400 | 0.1600 |
| `main_r16_lr2e-5_ep2` | 67/156 = 0.4295 | 36/63 = 0.5714 | 31/55 = 0.5636 | 55/139 = 0.3957 | 353/400 = 0.8825 | 0.1175 |
| `main_r16_lr5e-5_ep1` | 72/156 = 0.4615 | 39/63 = 0.6190 | 34/55 = 0.6182 | 61/139 = 0.4388 | 338/400 = 0.8450 | 0.1550 |
| `main_r16_lr5e-5_ep2` | 85/156 = 0.5449 | 46/63 = 0.7302 | 41/55 = 0.7455 | 74/139 = 0.5324 | 365/400 = 0.9125 | 0.0875 |
| `main_r16_lr1e-4_ep1` | 82/156 = 0.5256 | 43/63 = 0.6825 | 38/55 = 0.6909 | 73/139 = 0.5252 | 347/400 = 0.8675 | 0.1325 |
| `main_r16_lr1e-4_ep2` | 82/156 = 0.5256 | 46/63 = 0.7302 | 41/55 = 0.7455 | 70/139 = 0.5036 | 366/400 = 0.9150 | 0.0850 |
| `star_r8_lr2e-5_ep1` | 19/156 = 0.1218 | 11/63 = 0.1746 | 6/55 = 0.1091 | 10/139 = 0.0719 | 373/400 = 0.9325 | 0.0675 |
| `star_r8_lr2e-5_ep2` | 47/156 = 0.3013 | 26/63 = 0.4127 | 23/55 = 0.4182 | 37/139 = 0.2662 | 346/400 = 0.8650 | 0.1350 |
| `star_r8_lr5e-5_ep1` | 52/156 = 0.3333 | 29/63 = 0.4603 | 24/55 = 0.4364 | 41/139 = 0.2950 | 343/400 = 0.8575 | 0.1425 |
| `star_r8_lr5e-5_ep2` | 66/156 = 0.4231 | 35/63 = 0.5556 | 30/55 = 0.5455 | 53/139 = 0.3813 | 343/400 = 0.8575 | 0.1425 |
| `star_r8_lr1e-4_ep1` | 65/156 = 0.4167 | 32/63 = 0.5079 | 27/55 = 0.4909 | 53/139 = 0.3813 | 340/400 = 0.8500 | 0.1500 |
| `star_r8_lr1e-4_ep2` | 70/156 = 0.4487 | 38/63 = 0.6032 | 33/55 = 0.6000 | 59/139 = 0.4245 | 348/400 = 0.8700 | 0.1300 |
| `star_r16_lr2e-5_ep1` | 42/156 = 0.2692 | 23/63 = 0.3651 | 21/55 = 0.3818 | 32/139 = 0.2302 | 348/400 = 0.8700 | 0.1300 |
| `star_r16_lr2e-5_ep2` | 55/156 = 0.3526 | 30/63 = 0.4762 | 25/55 = 0.4545 | 43/139 = 0.3094 | 348/400 = 0.8700 | 0.1300 |
| `star_r16_lr5e-5_ep1` | 59/156 = 0.3782 | 31/63 = 0.4921 | 27/55 = 0.4909 | 48/139 = 0.3453 | 346/400 = 0.8650 | 0.1350 |
| `star_r16_lr5e-5_ep2` | 71/156 = 0.4551 | 38/63 = 0.6032 | 34/55 = 0.6182 | 60/139 = 0.4317 | 349/400 = 0.8725 | 0.1275 |
| `star_r16_lr1e-4_ep1` | 66/156 = 0.4231 | 34/63 = 0.5397 | 30/55 = 0.5455 | 57/139 = 0.4101 | 333/400 = 0.8325 | 0.1675 |
| `star_r16_lr1e-4_ep2` | 72/156 = 0.4615 | 42/63 = 0.6667 | 37/55 = 0.6727 | 63/139 = 0.4532 | 357/400 = 0.8925 | 0.1075 |

Best points:

- best val repair: `main_r16_lr5e-5_ep2`, val 85/156 = 0.5449, slice63 46/63 = 0.7302, scaffold-only 41/55 = 0.7455, r_success 365/400 = 0.9125, forget 0.0875.
- best retention: `main_r8_lr2e-5_ep1`, r_success 374/400 = 0.9350, forget 0.0650, but val 20/156 = 0.1282.
- best main-vs-star matched headline shape at high repair: main top val 0.5449 vs star top val 0.4615; both still break the retention gate by a wide margin.
