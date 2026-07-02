# Deferred Metrics

## Answer-Set Width Probe

Status: registered as exploratory `PROBE` analysis under v1.4. This metric is not a Gate 0 decision input, not a model-selection input, and not part of the headline C2 claim.

Hypothesis: scaffold-only episodes may have narrower acceptable answer sets than sampling-rescuable episodes, helping explain why base sampling reaches one group but not the other.

Caveat: the comparison is generator-confounded. Sampling-rescuable outputs come from base pass@16 records, while scaffold-only collapse is measured from teacher+patch outputs. Interpret as directional probe only.

Metric: for each episode, count distinct normalized passing outputs in the pass@16 retained records, then compare with the teacher-side scaffold-only collapse count.

Result:

- sampling-rescuable train episodes, all: n=32, distinct passing output distribution `{1: 32}`; pass-count distribution `{1: 2, 2: 2, 3: 8, 4: 4, 5: 4, 6: 3, 7: 2, 8: 5, 9: 2}`.
- sampling-rescuable train episodes that are also scaffold-repaired: n=10, distinct passing output distribution `{1: 10}`; pass-count distribution `{1: 1, 3: 2, 4: 2, 6: 2, 8: 1, 9: 2}`.
- sampling-rescuable train episodes not scaffold-repaired: n=22, distinct passing output distribution `{1: 22}`; pass-count distribution `{1: 1, 2: 2, 3: 6, 4: 2, 5: 4, 6: 1, 7: 2, 8: 4}`.
- scaffold-only repaired train episodes on the teacher side: n=64, distinct output distribution `{1: 64}`.

Interpretation: the output-cardinality version of the width hypothesis is not supported by the normalized records. Sampling-rescuable episodes also collapse to one distinct successful output per episode; the apparent difference is in hit probability/pass count, not in the number of distinct successful outputs observed.

## Base/Teacher Overlap Output Identity Check

Status: completed zero-GPU audit under v1.4. This check compares the 10 train episodes where scaffold repair and base pass@16 both produced at least one successful trajectory.

Metric: normalize each successful base output and each teacher output with the same `canonical_call` / `output_text` path used for Step 0.2, then compare the unique normalized call per episode.

Result:

- overlap episodes: n=10.
- same normalized target: 10/10.
- different normalized target: 0/10.
- multi-unique ambiguity: 0/10.
- base pass-count distribution on the 10 overlap episodes: `{1: 1, 3: 2, 4: 2, 6: 2, 8: 1, 9: 2}`.
- teacher source structure on each overlap episode: one `teacher_t0` row and one `teacher_t08` row, collapsing to one normalized output.

Interpretation: in the overlap region, base and teacher teach the literal same normalized target. This supports the stronger point-mass framing: the "point" appears to be task-level rather than model-relative, at least on the 10 constructible overlap episodes.

## Base Log-Probability Probe

Status: completed on 2026-07-02 as exploratory `PROBE` analysis only. This is not a Gate 0 decision input, not a model-selection input, and not part of the headline C2 claim.

Metric: compute teacher-forced base-model log-probability for the unique normalized correct output on each episode. Compare the distribution for scaffold-only repaired train episodes (`n=64`) against sampling-rescuable train episodes (`n=32`).

Prediction, written before running the probe: scaffold-only episodes will have significantly lower base log-probability on the unique correct output than sampling-rescuable episodes.

Second-layer prediction, written before running the probe: within scaffold-only validation or heldout episodes, transfer-success cases will have higher base log-probability on the unique correct output than transfer-failure cases. This tests whether base prior mass predicts internalizability even inside the low-prior stratum.

H-margin prediction, written before running the probe: the 24 shared fragile `R_success_eval` episodes will have lower base log-probability on the original correct call than non-fragile `R_success_eval` controls. Margin prediction: `log P(correct call) - log P(observed flipped-to call)` will be smaller for fragile episodes than for controls or non-fragile matched examples.

Dose-order prediction, written after the matched second clean point and before running the probe: validation episodes should order by base log-probability as `ep1 transferred` > `ep2 newly transferred` > `still not transferred`, within the same stratum where possible. On the regression side, `ep2` newly fragile episodes should occupy the next-lowest margin segment after the first-point shared fragile set.

Interpretation target: if supported, the C2 mechanism becomes "scaffold teaches points that the base model assigns low probability mass to; self-sampling reinforces points the base model already visits." If not supported, keep the negative result and do not use it as headline evidence.

Scheduling rule: first idle GPU runs heldout pass@16 before this probe; the log-probability probe runs only after that queued pure-forward labeling task.

Execution note: the log-probability probe was run in parallel after the clean grid completed, while heldout pass@16 ran on a separate GPU. This does not touch training data or model selection.

Result summary:

- scored targets: 569 requested; 14 `R_success_eval` records had no effective scored output token after prompt/target truncation at `max_length=2048`, so the `R_success` summary has `n=386`.
- train teacher-repaired scaffold-only: `n=64`, mean token log-prob `-1.1114`.
- train teacher-repaired sampling-rescuable: `n=10`, mean token log-prob `-0.9808`.
- all train base pass@16 successes: `n=32`, mean token log-prob `-1.0298`.
- validation teacher-repaired: `n=63`, mean token log-prob `-1.1035`.
- `R_success_eval` first-point shared fragile: `n=24`, mean token log-prob `-1.1728`.
- `R_success_eval` non-first-shared: `n=362`, mean token log-prob `-1.1376`.
- mean margin, main first point newly-wrong outputs: `+0.0533`.
- mean margin, STaR first point newly-wrong outputs: `+0.0404`.
- mean margin, main second point newly-wrong outputs: `-0.0315`.
- mean margin, STaR second point newly-wrong outputs: `+0.0376`.

Interpretation (`PROBE`): the first C2-side prediction is directionally supported: scaffold-only teacher targets are lower base-probability than sampling-rescuable targets. The H-margin probe is mixed rather than cleanly confirmed: shared fragile successes are only slightly lower-confidence than non-fragile controls, and the observed flipped-to margins are not uniformly smaller across the second-point grids. Keep the probability-terrain framing as a live mechanism hypothesis, not as a settled result.

## Val Repair Slice Reporting

Status: registered reporting-only. These slices are not selection criteria and do not alter the preregistered clean-grid selection rule, which remains aggregate `val_repair@156` plus `r_success >= 0.98`.

Metric: for each clean grid checkpoint, report aggregate `val_repair@156` and two disjoint slices:

- `val_teacher_repaired_63`: the 63 validation failures that teacher repair fixed. These are legal clean transfer targets because no validation episode enters training.
- `val_teacher_unrepaired_93`: the 93 validation failures teacher repair did not fix. Any non-zero repair here is beyond-teacher signal, with interpretation conditioned on function overlap with the train pool.

Aggregation identity: `val_repair@156 = 0.404 * slice_63 + 0.596 * slice_93`, up to rounding. Selection continues to use only the aggregate value and the preregistered retention gate.

Result after the first conservative clean point for each arm:

- slice composition before model evaluation:
  - `val_teacher_repaired_63`: scaffold-only 55, sampling-rescuable 8.
  - `val_teacher_unrepaired_93`: neither 84, sampling-rescuable 9.
- main `r8_lr2e-5_ep1`:
  - `slice63` successes: 13 total; scaffold-only 7, sampling-rescuable 6. All 13 functions are seen in the main training core.
  - `slice93` successes: 7 total; neither 3, sampling-rescuable 4. Function distribution: `Travel_1_FindAttractions` 2, `Trains_1_FindTrains` 1, `adriel_experiences_and_education` 1, `connect_to_server` 1, `music_shop.find_nearby` 1, `set_integer` 1. Seen in split-train function set: 3/7; seen in main training core: 2/7.
- STaR `r8_lr2e-5_ep1`:
  - `slice63` successes: 11 total; scaffold-only 6, sampling-rescuable 5. Seen in split-train function set: 11/11; seen in STaR training core: 7/11.
  - `slice93` successes: 8 total; neither 4, sampling-rescuable 4. Function distribution: `Travel_1_FindAttractions` 2, `Events_3_FindEvents` 1, `Trains_1_FindTrains` 1, `adriel_experiences_and_education` 1, `connect_to_server` 1, `music_shop.find_nearby` 1, `set_integer` 1. Seen in split-train function set: 4/8; seen in STaR training core: 3/8.

Interpretation: this first single-seed monitoring point does not yet show a clear C2-shaped split where main is scaffold-only-heavy and STaR is sampling-rescuable-heavy. The stronger immediate signal is that the 63-teacher-repaired validation slice itself is mostly scaffold-only (55/63), while the slice93 beyond-teacher repairs mix sampling-rescuable and neither partitions and include several train-function-unseen cases.

## Internalization-Forgetting Dose-Response Watch

Status: registered reporting-only. This watch is not a selection rule and does not change the preregistered clean-grid selection rule.

Metric: for every clean grid point, report four validation rates plus `forget`:

- `slice63_scaffold_only`: successes / 55.
- `slice63_sampling_rescuable`: successes / 8.
- `slice93_neither`: successes / 84.
- `slice93_sampling_rescuable`: successes / 9.

First conservative clean point (`r8_lr2e-5_ep1`) rates:

| Stratum | n | main | STaR |
| --- | ---: | ---: | ---: |
| slice63 scaffold-only | 55 | 7/55 = 0.1273 | 6/55 = 0.1091 |
| slice63 sampling-rescuable | 8 | 6/8 = 0.7500 | 5/8 = 0.6250 |
| slice93 neither | 84 | 3/84 = 0.0357 | 4/84 = 0.0476 |
| slice93 sampling-rescuable | 9 | 4/9 = 0.4444 | 4/9 = 0.4444 |
| low-prior combined (scaffold-only + neither) | 139 | 10/139 = 0.0719 | 10/139 = 0.0719 |
| high-prior combined (sampling-rescuable) | 17 | 10/17 = 0.5882 | 9/17 = 0.5294 |

Interpretation: at the first point, transfer landing is dominated by base-prior stratum. High-prior sampling-rescuable validation episodes repair at roughly eight times the low-prior scaffold-only/neither rate, while main vs STaR differences are within single-seed noise. The watch question for the full grid is whether stronger recipes move the scaffold-only column upward, and whether main separates from STaR there.

Main second point, monitoring-only (`main_r8_lr2e-5_ep2`):

- `val_repair@156`: 55/156 = 0.3526.
- `slice63_scaffold_only`: 28/55 = 0.5091.
- `slice63_sampling_rescuable`: 5/8 = 0.6250.
- `slice93_neither`: 16/84 = 0.1905.
- `slice93_sampling_rescuable`: 6/9 = 0.6667.
- low-prior combined: 44/139 = 0.3165.
- high-prior combined: 11/17 = 0.6471.
- `r_success`: 343/400 = 0.8575.
- `forget`: 0.1425.

First dose-response slope, main only: moving from `ep1` to `ep2` substantially increases low-prior/scaffold-only transfer (`slice63_scaffold_only` 7/55 to 28/55) but also sharply worsens forgetting (`0.0650` to `0.1425`). This is monitoring-only and not a selection decision.

STaR second point, monitoring-only (`star_r8_lr2e-5_ep2`):

- `val_repair@156`: 47/156 = 0.3013.
- `slice63_scaffold_only`: 23/55 = 0.4182.
- `slice63_sampling_rescuable`: 3/8 = 0.3750.
- `slice93_neither`: 14/84 = 0.1667.
- `slice93_sampling_rescuable`: 7/9 = 0.7778.
- low-prior combined: 37/139 = 0.2662.
- high-prior combined: 10/17 = 0.5882.
- `r_success`: 346/400 = 0.8650.
- `forget`: 0.1350.

First dose-response slope, STaR only: moving from `ep1` to `ep2` also increases low-prior/scaffold-only transfer (`slice63_scaffold_only` 6/55 to 23/55) while worsening forgetting (`0.0675` to `0.1350`). This matches the main-arm frontier shape directionally.

Main higher-lr single-epoch point, monitoring-only (`main_r8_lr5e-5_ep1`):

- `val_repair@156`: 63/156 = 0.4038.
- `val_teacher_repaired_63`: 37/63 = 0.5873.
- `val_teacher_unrepaired_93`: 26/93 = 0.2796.
- `slice63_scaffold_only`: 31/55 = 0.5636.
- `slice63_sampling_rescuable`: 6/8 = 0.7500.
- `slice93_neither`: 18/84 = 0.2143.
- `slice93_sampling_rescuable`: 8/9 = 0.8889.
- low-prior combined: 49/139 = 0.3525.
- high-prior combined: 14/17 = 0.8235.
- `r_success`: 349/400 = 0.8725.
- `forget`: 0.1275.

Interpretation: higher learning rate at one epoch reaches the validation-preview usable band and nearly reaches the 0.60 retention-preview shoulder on `slice63`, but forgetting remains far above the 0.02 gate. This continues the frontier picture rather than providing a clean selectable point.

Additional rank8 clean-grid points, monitoring-only:

| point | val@156 | slice63 | slice93 | scaffold-only | low-prior | high-prior | r_success | forget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `main_r8_lr5e-5_ep2` | 77/156 = 0.4936 | 42/63 = 0.6667 | 35/93 = 0.3763 | 37/55 = 0.6727 | 66/139 = 0.4748 | 11/17 = 0.6471 | 361/400 = 0.9025 | 0.0975 |
| `main_r8_lr1e-4_ep1` | 77/156 = 0.4936 | 41/63 = 0.6508 | 36/93 = 0.3871 | 36/55 = 0.6545 | 65/139 = 0.4676 | 12/17 = 0.7059 | 343/400 = 0.8575 | 0.1425 |
| `star_r8_lr5e-5_ep1` | 52/156 = 0.3333 | 29/63 = 0.4603 | 23/93 = 0.2473 | 24/55 = 0.4364 | 41/139 = 0.2950 | 11/17 = 0.6471 | 343/400 = 0.8575 | 0.1425 |
| `star_r8_lr5e-5_ep2` | 66/156 = 0.4231 | 35/63 = 0.5556 | 31/93 = 0.3333 | 30/55 = 0.5455 | 53/139 = 0.3813 | 13/17 = 0.7647 | 343/400 = 0.8575 | 0.1425 |

Interpretation: the waterline/frontier picture continues. Increasing dose raises low-prior/scaffold-only repair, but all observed rank8 higher-dose points remain far outside the forgetting gate.

## Clean Second-Point Turnover and Pairing Probe

Status: completed zero-GPU set analysis for the matched `r8_lr2e-5_ep2` points. This is monitoring-only `PROBE` analysis, not a selection rule and not a heldout claim.

Validation repair turnover from `ep1` to `ep2`:

- main: `ep1` repaired 20 episodes, `ep2` repaired 55 episodes. The `ep1` repaired set is not a subset of `ep2`: 9 first-point repairs are lost, and 44 second-point repairs are new.
- STaR: `ep1` repaired 19 episodes, `ep2` repaired 47 episodes. The `ep1` repaired set is not a subset of `ep2`: 10 first-point repairs are lost, and 38 second-point repairs are new.

Interpretation: the dose-order waterline is not literal monotonic set inclusion at the episode level. It is a probabilistic/greedy-boundary model: higher dose raises aggregate low-prior transfer, while individual repairs can turn over.

First-point fragile-set carryover into `ep2` newly-wrong episodes:

- main: the 24 shared first-point fragile episodes are not a subset of the `ep2` newly-wrong set; 14/24 remain newly wrong and 10/24 are no longer newly wrong.
- STaR: the 24 shared first-point fragile episodes are not a subset of the `ep2` newly-wrong set; 13/24 remain newly wrong and 11/24 are no longer newly wrong.

Interpretation: damage also rotates with dose. The first-point fragile set remains enriched for later damage, but higher dose does not simply preserve a fixed fragile core and add new failures.

Second-point `R_success_eval` newly-wrong overlap:

- main newly wrong: 57 episodes; type distribution `parameter_drift=51`, `function_flip=6`.
- STaR newly wrong: 54 episodes; type distribution `parameter_drift=45`, `function_flip=9`.
- set overlap: intersection 45, union 66, Jaccard `0.6818`.
- main overlap fraction: 45/57 = `0.7895`.
- STaR overlap fraction: 45/54 = `0.8333`.
- paired type+subtype agreement on the 45 shared second-point newly-wrong episodes: 43/45.

Second-point parameter-drift subtype distributions:

- main: add 19, add+value 1, drop 3, drop+add 12, drop+add+value 1, value 15; function-flip/NA 6.
- STaR: add 14, add+value 1, drop 1, drop+add 12, drop+add+value 1, drop+value 2, value 14; function-flip/NA 9.

Paired subtype table on the 45 shared second-point newly-wrong episodes:

| main subtype | STaR subtype | count |
| --- | --- | ---: |
| add | add | 12 |
| add | NA/function_flip | 1 |
| add+value | add+value | 1 |
| drop | drop | 1 |
| drop+add | drop+add | 10 |
| drop+add+value | drop+add+value | 1 |
| NA/function_flip | NA/function_flip | 6 |
| value | value | 12 |
| value | drop+value | 1 |

Interpretation: the first-point exact paired agreement `24/24` relaxes at higher dose but remains strong (`43/45`). This keeps the structural-damage hypothesis alive: different teaching sets still tend to push shared fragile episodes toward the same subtype of wrong neighbor.

Second-point C2 paired cross tables:

Low-prior validation episodes (`scaffold_only + neither`, n=139):

| | STaR repaired | STaR not repaired |
| --- | ---: | ---: |
| main repaired | 33 | 11 |
| main not repaired | 4 | 91 |

Scaffold-only validation episodes (`n=55`):

| | STaR repaired | STaR not repaired |
| --- | ---: | ---: |
| main repaired | 21 | 7 |
| main not repaired | 2 | 25 |

All validation failures (`n=156`):

| | STaR repaired | STaR not repaired |
| --- | ---: | ---: |
| main repaired | 42 | 13 |
| main not repaired | 5 | 96 |

Interpretation: the second point has a C2-shaped marginal direction in the low-prior/scaffold-only strata, but the paired tables show substantial shared content-agnostic movement. This remains single-seed monitoring, not a claim.

## Clean First-Point Newly-Wrong Probe

Status: completed for the first conservative clean point in each arm. This is a diagnostic `PROBE`, not a selection rule and not a heldout claim.

Metric: on `R_success_eval`, compare each newly wrong checkpoint output against the base model's original successful call. Categorize as format drift, function flip, or parameter drift. Also compare the main/STaR newly-wrong episode sets.

Result:

- main `r8_lr2e-5_ep1`: 26 newly wrong episodes; parameter_drift 26/26, function_flip 0, format_drift 0.
- STaR `r8_lr2e-5_ep1`: 27 newly wrong episodes; parameter_drift 27/27, function_flip 0, format_drift 0.
- set overlap: intersection 24, union 29, Jaccard 0.8276.
- main overlap fraction: 24/26 = 0.9231.
- STaR overlap fraction: 24/27 = 0.8889.
- main-only new-wrong episode IDs: `live_multiple_162-63-1`, `live_multiple_74-34-0`.
- STaR-only new-wrong episode IDs: `live_multiple_361-134-6`, `live_multiple_546-152-8`, `multiple_134`.
- overlap function distribution: `Movies_3_FindMovies` 3, `Events_3_FindEvents` 2, `RentalCars_3_GetCarsAvailable` 2, and 17 singletons.

Interpretation: the clean first-point forgetting mechanism matches the contaminated-diagnostic pattern even more sharply: newly wrong outputs are entirely parameter drift. The high cross-arm overlap supports a structural interference story, where the same regression episodes are fragile under failure-repair fine-tuning regardless of whether the repair data came from scaffold distillation or STaR pass@16.

## Fragile-Set Characterization

Status: completed zero-GPU characterization for the 24 newly-wrong episodes shared by main and STaR at the first conservative clean point. This is diagnostic only and must not be used to tune against `R_success_eval` without creating a fresh regression arena.

Fragile function distribution:

- `Movies_3_FindMovies`: 3.
- `Events_3_FindEvents`: 2.
- `RentalCars_3_GetCarsAvailable`: 2.
- singletons: `Alarm_1_GetAlarms`, `ChaDri.change_drink`, `HNA_NEWS.search`, `Music_3_LookupMusic`, `RideSharing_2_GetRide`, `Travel_1_FindAttractions`, `adriel_contact`, `calculate_probability`, `contact`, `cookbook.search_recipe`, `flight.status.check`, `get_crime_rate`, `get_detail_adriel_projects`, `get_event_date`, `get_identity_provider_patch`, `multilingual_llm`, `search_web_tool`.

Training-side coverage of the 24 shared fragile episodes:

- exact function in main repair-core function set: 9/24.
- exact function in STaR repair-core function set: 9/24.
- exact function in union repair-core function set: 11/24.
- exact function in union training data including replay: 14/24.
- exact function in clean train-failure function set: 15/24.
- coarse family in union repair-core family set: 11/24.
- coarse family in union training data including replay: 15/24.
- coarse family in clean train-failure family set: 16/24.

Parameter-drift subtype distribution:

- shared 24, main view: add 3, add+value 1, drop 11, drop+value 2, value 7.
- shared 24, STaR view: add 3, add+value 1, drop 11, drop+value 2, value 7.
- paired subtype agreement: 24/24 exact subtype agreement between arms.
- all main newly wrong 26: add 4, add+value 1, drop 12, drop+value 2, value 7.
- all STaR newly wrong 27: add 3, add+value 1, drop 13, drop+value 2, value 8.

Interpretation: the shared fragile set is not concentrated enough for a narrow exact repair-core rule to cover a majority. A broader clean train-failure function/family rule covers more of it, but because the motivation came from inspected `R_success_eval` failures, targeted replay remains conditional on a fresh regression arena.

## H-Content Zero-GPU Check

Status: completed zero-GPU check. This tests a data-side hypothesis for drop drift and is diagnostic only.

Hypothesis, written before calculation: if H-content is true, training outputs should be statistically thinner than fragile correct calls, and the parameters dropped in fragile episodes should be rare in training outputs.

Result:

- main core training outputs: mean top-level argument count 3.6351; mean optional-present ratio 0.7205.
- STaR core training outputs: mean top-level argument count 3.4730; mean optional-present ratio 0.5824.
- union core training outputs: mean top-level argument count 3.5541; mean optional-present ratio 0.6515.
- all training rows including replay: mean top-level argument count 3.2635; mean optional-present ratio 0.5511.
- fragile 24 correct calls: mean top-level argument count 2.9167; mean optional-present ratio 0.4806.
- non-fragile `R_success_eval` controls: mean top-level argument count 2.7655; mean optional-present ratio 0.3424.

Dropped-parameter frequency:

- shared-fragile dropped-parameter occurrences: 16, covering 14 unique `(function, parameter)` pairs.
- five of sixteen dropped-parameter occurrences have zero same-function core rows.
- for the eleven occurrences with same-function core rows, the dropped top-level parameter is present in those same-function core rows.

Interpretation: this does not support the simple H-content story. Training outputs are not thinner than fragile correct calls; they are fatter and more optional-parameter-heavy. Dropped parameters are mixed: some have no same-function core support, but when same-function core support exists, the dropped parameter is not rare there. This shifts weight toward H-margin as the primary mechanism hypothesis.

## s04 Log-Prob Follow-Up Analyses

Status: completed from `results/s04_logprob_probe.json`; no additional GPU scoring was needed because flipped-output scores and validation grid-success flags were already present.

H-margin follow-up:

| group | n | mean | median |
| --- | ---: | ---: | ---: |
| first_shared_fragile correct logprob | 24 | -1.1728 | -1.0609 |
| non_first_shared correct logprob | 362 | -1.1376 | -1.0821 |
| ep2_union_fragile correct logprob | 63 | -1.2096 | -1.0893 |
| fragile24 margin main_r8_lr2e-5_ep1 | 24 | 0.0356 | 0.0648 |
| fragile24 margin star_r8_lr2e-5_ep1 | 24 | 0.0340 | 0.0648 |
| fragile24 margin main_r8_lr2e-5_ep2 | 14 | -0.1896 | -0.0553 |
| fragile24 margin star_r8_lr2e-5_ep2 | 13 | -0.1758 | -0.0772 |

Interpretation: correct-call confidence is directionally lower for fragile sets, weakly for first-shared and more strongly for the ep2 union. Actual flipped-to outputs are very close to, and sometimes higher-probability than, the correct output. Non-fragile margin controls are unavailable because non-fragile episodes have no observed flipped-to output in the logged grid JSONs; this is therefore a direct fragile-24 margin check rather than a controlled margin comparison.

Dose-order prediction:

| arm | partition | ep1 transferred | ep2 new | not by ep2 |
| --- | --- | ---: | ---: | ---: |
| main | all validation teacher-repaired | 13 / -0.9919 | 24 / -1.1117 | 26 / -1.1517 |
| main | scaffold-only | 7 / -1.0011 | 23 / -1.1184 | 25 / -1.1697 |
| STaR | all validation teacher-repaired | 11 / -0.9960 | 21 / -1.1213 | 31 / -1.1295 |
| STaR | scaffold-only | 6 / -0.9968 | 19 / -1.1409 | 30 / -1.1438 |

Interpretation: the registered dose-order prediction is broadly supported, especially for main and for the scaffold-only slice: higher base log-probability episodes transfer earlier, while lower-probability episodes either need ep2 or remain untransferred.

Scaffold-only internal success/failure contrast:

| arm | grid | transferred | not transferred |
| --- | --- | ---: | ---: |
| main | main_r8_lr2e-5_ep1 | 7 / -1.0011 | 48 / -1.1451 |
| main | main_r8_lr2e-5_ep2 | 28 / -1.0923 | 27 / -1.1625 |
| STaR | star_r8_lr2e-5_ep1 | 6 / -0.9968 | 49 / -1.1427 |
| STaR | star_r8_lr2e-5_ep2 | 23 / -1.1093 | 32 / -1.1394 |

Interpretation: within scaffold-only validation teacher repairs, transferred episodes have higher base log-probability than non-transferred episodes in all four monitored points. This supports the waterline/prior-mass model at the slice where C2 has most of its transfer-layer discriminative power.

## s04 Replay 2:1 Weighted Dataset

Status: built for the BLOCKED KL grid.

- main: 148 core rows, 296 replay rows, 444 total rows; replay uses 226 unique replay rows plus 70 deterministic repeat rows.
- STaR: 148 core rows, 296 replay rows, 444 total rows; replay uses 226 unique replay rows plus 70 deterministic repeat rows.
- leakage assertions: heldout/R_success/D_val intersections are all 0 for both arms.

Interpretation: replay 2:1 uses all available unique clean replay rows before repeat weighting. The repeat structure is transparent and symmetric across arms.
