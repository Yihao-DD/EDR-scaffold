# CHANGELOG

## v1.2 - 2026-07-01

### Phase 0 validation-contamination correction

The initial Phase 0 grid trained on D_val episodes because Step 0.0 allowed repaired trajectories from `D_train ∪ D_val`, while the governance table said D_val must not enter training and the implemented leakage assertions only covered `D_heldout` and `R_success_eval`.

All initial grid numbers are now labeled `PROBE (contaminated)` and cannot be used for model selection, Gate decisions, or publication-ready claims. The completed contaminated checkpoints and logs are archived under `contaminated_gridv1/`; their only allowed use is forgetting diagnosis on the clean `R_success_eval` split.

Option A is selected because train-only repaired episodes are >=50 and clean validation failures are >=30. D_val exits training. The new invariant is:

`episodes(training) ∩ episodes(D_val) = ∅`

This invariant is now enforced alongside heldout and R-success leakage checks.

Clean Phase 0 selection will use all D_val failures. Heldout remains untouched until a clean configuration is selected.

## v1.3 - 2026-07-01

### Optional A4b diagnostic arm preregistration

Add optional diagnostic arm **A4b episode-matched scaffold-taught**.

Status: diagnostic only, not headline, not a Gate 0 decision input, and not part of the main four-arm table unless explicitly run after Gate 0. It receives no independent hyperparameter tuning; if run, it uses the already selected A3 configuration and 5 seeds.

Purpose: answer the narrow rebuttal-style question of whether scaffold-taught trajectories are stronger than self-sampled trajectories on the subset of episodes both sources can reach. This is separate from the headline C2 test, which remains A3 vs A4 on the heldout scaffold-only partition with paired CI.

Trigger: run only if needed for analysis/rebuttal or if clean results make the episode-diversity confound central. Otherwise leave it unrun and report it as preregistered optional diagnostics.

Addendum: after the v1.4 point-mass and STaR dedup audits, the actually constructible A4b overlap is only 10 episodes, because main has teacher data for only 10 of STaR's 31 selected episodes. Evidence ceiling is therefore `PROBE`; after Gate 0, A4b is likely downgraded to footnote analysis or skipped with this structural non-constructibility recorded.

### Clean pool note

The clean full main core pool has 148 rows, not >=1.5x the 148-row paired headline core, so the proposed A8 full-pool rule is not triggered. Full clean core pools are archived under `EDG-EXP3-distill/data/full_clean_pools/` for future diagnostics and scaling work.

## v1.4 - 2026-07-02

### Augmentation ceiling and Gate 0 retry-playbook correction

Timing: recorded before any clean-grid selection metric was returned; heldout remains untouched.

Trigger: augmentation funnel audit on the clean main teacher pool. Main T=0.8 x8 produced 592 sampled trajectories, 553 AST-valid trajectories (93.4%), and after per-episode deduplication exactly one unique output for each of the 74 train repaired episodes (`teacher_t08=74`).

Conclusion: the teacher's correct output for these repaired episodes behaves as a point-mass distribution. Output-side augmentation diversity is structurally capped by the task/output space; additional sampling cannot create diversity that is not present.

Revisions:

1. In the Gate 0 primary criterion retry clause for `<0.40`, the augmentation lever is retired. Retry levers are now hyperparameters (`lr`, `epochs`, `rank`) and replay ratio. The Gate thresholds and bands (`0.60` / `0.40`) are unchanged.
2. Step 0.2's "sample more up to x8 if insufficient" clause has already been executed to its ceiling and is no longer a usable lever for Phase 0 clean retry.
3. A8 data-volume scaling changes to episode-level downsampling: `{25%, 50%, 100%}` are sampled by episode, not by row, because row-level sampling would break the repeated-weight structure induced by point-mass outputs.

Signed-off: Ian.

## v1.5 - 2026-07-02

### First clean-point forgetting observation and BLOCKED menu

Timing: recorded after the first conservative clean grid point for each arm returned, before any clean-grid model-selection decision and with heldout still untouched.

Observation: the most conservative clean point already breaks the forgetting gate in both arms:

- main `r8_lr2e-5_ep1`: `r_success=374/400=0.9350`, `forget=0.0650`.
- STaR `r8_lr2e-5_ep1`: `r_success=373/400=0.9325`, `forget=0.0675`.

This observation is `PROBE` and monitoring-only, not a selection decision. However, the cross-arm shape is now recorded: main and STaR have nearly equal forgetting despite very different unique teaching sets (`main=74` unique examples at x2 repeat weight; `STaR=31` unique examples at about x4.8 repeat weight). This supports the working hypothesis that forgetting is a generic interference effect of fine-tuning on failure-repair trajectories, not something that scales simply with unique teaching diversity or depends specifically on teacher-generated content.

If the preregistered upgrade point (`replay=1:1` plus halved learning rate from the best repair candidate) also breaks the forgetting gate, the human decision menu is:

1. Increase replay further (`2:1` or `3:1`). This is the cheapest in-family retry. Evidence required: the upgrade point must show a favorable forget-improvement slope from replay.
2. Add a KL-anchor / distillation-anchor loss on replay inputs to penalize divergence from the base model. This is a recipe extension and requires a CHANGELOG entry before use. Evidence required: standard literature precedent plus a scoped implementation plan.
3. Try targeted replay weighted toward drift-prone function families. This is parked, not a default action, because using drift patterns read from `R_success_eval` to redesign training would adaptively contaminate the regression gate. Preconditions: carve a fresh regression arena before using targeted replay as a training-design input.
4. Accept the frontier negative result as publishable evidence if 1-3 are exhausted: for 7B plus this recipe family, the internalization-forgetting frontier does not reach the preregistered acceptable region. C1 is then downgraded to boundary characterization, while C2 and the mechanism analyses remain active.

The forgetting gate remains unchanged at `forget <= 0.02`.

## v1.12 - 2026-07-02

### s05 KL-anchor result and targeted-replay decision setup

Timing: recorded after all `14/14` s05 BLOCKED first-wave JSONs returned, before any targeted replay run or fresh-arena revision.

Mechanical decision: no s05 point satisfies the regression gate. The best observed forgetting point is `main replay2 lambda=2`, with `forget=21/400=0.0525` and `slice63=33/63=0.5238`; this is still 2.6x above the fixed `forget <= 0.02` gate. Therefore s05 has no success or partial-success point under the preregistered v1.11 exit criteria.

s05 result table:

| arm | config | val_repair | slice63 | r_success | forget |
| --- | --- | ---: | ---: | ---: | ---: |
| main | replay1 lambda0.5 | 73/156=0.4679 | 44/63=0.6984 | 357/400=0.8925 | 43/400=0.1075 |
| main | replay1 lambda1 | 68/156=0.4359 | 43/63=0.6825 | 364/400=0.9100 | 36/400=0.0900 |
| main | replay1 lambda2 | 62/156=0.3974 | 39/63=0.6190 | 367/400=0.9175 | 33/400=0.0825 |
| main | replay2 lambda0.5 | 67/156=0.4295 | 41/63=0.6508 | 370/400=0.9250 | 30/400=0.0750 |
| main | replay2 lambda1 | 64/156=0.4103 | 42/63=0.6667 | 372/400=0.9300 | 28/400=0.0700 |
| main | replay2 lambda2 | 48/156=0.3077 | 33/63=0.5238 | 379/400=0.9475 | 21/400=0.0525 |
| main | ep3 replay1 | 79/156=0.5064 | 45/63=0.7143 | 366/400=0.9150 | 34/400=0.0850 |
| STaR | replay1 lambda0.5 | 63/156=0.4038 | 34/63=0.5397 | 363/400=0.9075 | 37/400=0.0925 |
| STaR | replay1 lambda1 | 56/156=0.3590 | 29/63=0.4603 | 369/400=0.9225 | 31/400=0.0775 |
| STaR | replay1 lambda2 | 49/156=0.3141 | 24/63=0.3810 | 370/400=0.9250 | 30/400=0.0750 |
| STaR | replay2 lambda0.5 | 63/156=0.4038 | 34/63=0.5397 | 373/400=0.9325 | 27/400=0.0675 |
| STaR | replay2 lambda1 | 54/156=0.3462 | 29/63=0.4603 | 373/400=0.9325 | 27/400=0.0675 |
| STaR | replay2 lambda2 | 40/156=0.2564 | 23/63=0.3651 | 377/400=0.9425 | 23/400=0.0575 |
| STaR | ep3 replay1 | 70/156=0.4487 | 36/63=0.5714 | 367/400=0.9175 | 33/400=0.0825 |

Interpretation (`PROBE`): KL-anchor and replay2 reduce forgetting but trade away transfer. Ep3 increases transfer but does not clear the regression gate. Main remains stronger than STaR on high-transfer points, but neither arm has a deployment-grade regression point.

Decision setup for the final targeted-replay attempt: D1 may only proceed after a fresh regression arena is constructed or the arena protocol is explicitly revised. The targeting rule remains train-side only: choose the non-targeted configuration with the lowest fresh forgetting, then replace uniform replay with a targeted replay pool of the same size and otherwise identical hyperparameters.

## v1.13 - 2026-07-02

### Fresh-arena blocker,可满足性协议, and targeted-pool construction

Timing: recorded after the s05 table and before any targeted replay run. This entry supersedes the impossible v1.12 fresh-arena precondition.

Fresh arena blocker: the original strict fresh-arena rule is not constructible in the current `multiple` base-success pool. Pool arithmetic: total multiple base successes `626`; old `R_success_eval` uses `400`; historical replay variants cover the remaining `226`; therefore the number of base-success episodes that have never entered eval, replay, or training is `0`.

Attribution: this is a specification bug in the preregistration document, not an agent execution error. The rule was written without a feasibility/pool-arithmetic check.

Anti-self-deception protocol, new rule 9: every newly registered condition that constrains data availability must include a satisfiability check at registration time. This means a pool arithmetic check or constructibility assert must be recorded before the condition is treated as executable. Unexecutable conditions may not be silently relaxed after results are known.

Revised arena source hierarchy:

1. Primary: use untouched BFCL sibling categories if they are compatible with the current single-call evaluator and yield at least `250` base T=0 successes. The arena measures regression on same-family unseen data and becomes the exploration regression gate.
2. Secondary: if primary is insufficient, use a capped-pool design that reserves a held-out subset from the replay-success pool and fills replay by repeat weighting. This is `PROBE`-level because the arena is smaller.
3. Fallback: if neither primary nor secondary is feasible, exploration-stage regression remains old `R_success_eval` plus Simple with full adaptivity caveat; final regression authority moves to D2 heldout base successes.

D2 metric addition: `forget_heldout` on heldout base-success episodes is now a formal D2 metric. This guarantees the final regression claim includes a genuinely fresh arena even if exploration-stage targeting must use a caveated proxy.

Category touch audit result (`results/v13_category_touch_audit.json`): all categories already present in EXP1 baseline records have zero untouched successes after excluding historical eval/training/replay. Recorded categories and untouched successes: `live_multiple=0`, `multiple=0`, `live_simple=0`, `simple_python=0`, `simple_java=0`, `simple_javascript=0`.

Primary sibling-category arena status: failed under the current single-call evaluator. Unrecorded raw BFCL categories `memory` (`155` rows) and `web_search` (`100` rows) are not function-calling tasks under the current evaluator (`0` parseable ground-truth rows), and `parallel` / `parallel_multiple` / `live_parallel` / `live_parallel_multiple` plus multi-turn variants require multi-call or multi-turn metrics. The only unrecorded raw rows that are parseable by the current evaluator are `simple_python` extras (`241` rows), but that is an already touched Simple family and is below the `250` threshold even before base-success filtering. Therefore the primary fresh-arena source is not available without a new evaluator or a new preregistered arena source.

A2 targeted ceiling update (`results/v13_residual_targeted_coverage.json`): on s05 `replay2 lambda=2` residual new wrongs, train-failure exact/coarse coverage is:

- main: `21` new wrong; exact function covers `14/21`; coarse family covers `17/21`; perfect-protection residual forget would be `0.0175` exact or `0.0100` coarse.
- STaR: `23` new wrong; exact function covers `18/23`; coarse family covers `19/23`; perfect-protection residual forget would be `0.0125` exact or `0.0100` coarse.

Targeted replay pool built (`results/s06_targeted_replay2.json`): each arm has `148` core rows plus `296` targeted replay rows, total `444`. The targeted replay rows are selected by train-failure exact function first; this exact stage alone supplies the whole budget via repeat weighting: `123` unique replay episodes repeated to `296` rows. Leakage assertions remain zero against heldout, old `R_success_eval`, and `D_val` for both arms.

Secondary capped-pool arena activated (`results/v13_capped_pool_arena.json`, `results/s06_capped_pool_datasets.json`) because the primary sibling-category arena is unavailable. The original `226` unique replay-success rows are split deterministically into `120` allowed replay rows and `106` capped-arena rows. The capped arena is `PROBE`-level by design because the source pool was used in earlier historical replay variants, but it is held out from all capped D3/D1 runs.

Capped uniform replay2 datasets: each arm has `148` core rows + `296` replay rows, with the replay rows drawn from `120` unique allowed replay episodes and repeat-weighted to 296. Capped targeted replay2 datasets: each arm has `148` core rows + `296` targeted replay rows; exact train-failure function matching supplies the whole budget from `63` unique allowed replay episodes, repeat-weighted to 296. All capped datasets have zero intersection with the capped arena, heldout, old `R_success_eval`, and `D_val`.

The fixed forgetting gate remains unchanged at `forget <= 0.02`.

## v1.14 - 2026-07-02

### Capped-arena measurement audit and D2 representative revision

Timing: recorded after s06 D3 and targeted C returned, before D2 heldout 5-seed runs. This entry separates valid capped-arena measurements from invalid retrospective s05 capped-arena backfills.

Measurement-validity bug: the capped arena was reserved only for future capped D3/D1 runs. Retrospective s05 checkpoints were trained on the old uncapped replay pools, which overlap the capped arena. Therefore all s05 capped-arena backfills are `INVALID (arena-train overlap)` and may not be used as fresh-pass evidence. Attribution: v1.13 added constructibility checks but did not also require per-checkpoint measurement-validity checks against training rows.

Anti-self-deception protocol, rule 9 expansion: every new evaluation arena must include both (a) satisfiability/constructibility arithmetic and (b) per-checkpoint measurement-validity assertions: `episodes(training_checkpoint) ∩ episodes(evaluation_arena) = ∅` before any number can carry gate authority.

Exact capped-arena overlap audit (`arena_n=106`):

| checkpoint | status | unique train episodes | train rows | arena unique overlap | arena row overlap |
|---|---:|---:|---:|---:|---:|
| s05 main ep3 replay1 | INVALID | 222 | 296 | 65 | 65 |
| s05 main replay2 lambda2 | INVALID | 300 | 444 | 106 | 138 |
| s05 STaR ep3 replay1 | INVALID | 179 | 296 | 75 | 75 |
| s05 STaR replay2 lambda2 | INVALID | 257 | 444 | 106 | 138 |
| D3 main lambda1 | VALID | 194 | 444 | 0 | 0 |
| D3 main lambda2 | VALID | 194 | 444 | 0 | 0 |
| D3 STaR lambda1 | VALID | 151 | 444 | 0 | 0 |
| D3 STaR lambda2 | VALID | 151 | 444 | 0 | 0 |
| C main targeted | VALID | 137 | 444 | 0 | 0 |
| C STaR targeted | VALID | 94 | 444 | 0 | 0 |

The apparent s05 STaR ep3 replay1 capped-arena pass (`1/106`) is a mirage: that checkpoint trained on `75/106` arena episodes. It must not appear as a fresh regression success in reports.

Valid capped-arena comparison: D3 main lambda1 `7/106`, D3 main lambda2 `5/106`, D3 STaR lambda1 `5/106`, D3 STaR lambda2 `3/106`; C main targeted `6/106`, C STaR targeted `4/106`. Targeted replay differs from clean D3 baselines by only 1-2 episodes at `n=106`; the earlier wording that targeted replay worsened fresh forgetting is withdrawn. The mechanical D1 targeted verdict still fails because both targeted points exceed the capped `2/106` gate, and the preregistered hard stop remains active.

NaN caveat strengthened: targeted C had substantial non-finite-loss skipping (`main=36`, `STaR=54`). These runs are valid for the binary "failed the capped gate" verdict, but their transfer magnitudes should be read lightly because `lambda2` plus high-repeat targeted replay appears numerically brittle.

Coverage does not imply protection (`PROBE` mechanism note): s05 replay2 lambda2 checkpoints trained over all `106` capped-arena episodes, yet still missed `4` (main) and `5` (STaR). All of those missed episode IDs were present in their respective training rows. This invalidates the fresh-pass use of those numbers, but it is useful sample-in evidence that replay exposure alone does not guarantee protection.

Capped-arena wrong-set audit: capped-arena wrongs are disjoint from the old fragile-24 set (`0` overlap). The clean D3/C runs have no episode wrong in every clean configuration, but five recurring wrong IDs appear in at least three clean configurations: `live_multiple_1003-232-2`, `live_multiple_453-145-4`, `live_multiple_469-145-20`, `live_multiple_956-203-0`, and `multiple_178`. These form the current `PROBE` irreducible-interference watch set.

Sibling-category audit closure: the tier-1 sibling-category arena is unavailable under the current single-call evaluator. No untouched recorded success pool remains, and unrecorded compatible single-call categories are insufficient; multi-call, parallel, memory, web-search, and multi-turn categories require different evaluators. Thus the capped arena remains an exploration proxy only, and final regression authority stays with D2 `forget_heldout`.

D2 representatives revised and locked:

- rep1, best-transfer representative: both arms use s05 `ep3 replay1`. The s05 capped-arena backfill is invalid, but D2 `forget_heldout` will adjudicate regression on genuinely fresh heldout base-success episodes.
- rep2, valid lowest capped forgetting: both arms use D3 `ep3 replay2 lambda2` (`main=5/106`, `STaR=3/106`). s05 checkpoints exit rep2 competition because their capped-arena measurements are invalid.
- final regression authority: `forget_heldout` in D2. Capped arena remains `PROBE`/reference, old `R_success_eval` remains continuity-only, and no additional D1 recipe exploration is authorized.

## v1.6 - 2026-07-02

### Prior-governed transfer observation and targeted-replay specification

Timing: recorded after zero-GPU stratification of the first conservative clean point, before any clean-grid model-selection decision and with heldout still untouched.

Observation (`PROBE`, single seed, weakest grid point): transfer landing is dominated by base-prior stratum, not by teaching source.

| Validation stratum | n | main repair rate | STaR repair rate |
| --- | ---: | ---: | ---: |
| slice63 scaffold-only | 55 | 7/55 = 0.1273 | 6/55 = 0.1091 |
| slice63 sampling-rescuable | 8 | 6/8 = 0.7500 | 5/8 = 0.6250 |
| slice93 neither | 84 | 3/84 = 0.0357 | 4/84 = 0.0476 |
| slice93 sampling-rescuable | 9 | 4/9 = 0.4444 | 4/9 = 0.4444 |
| low-prior combined (scaffold-only + neither) | 139 | 10/139 = 0.0719 | 10/139 = 0.0719 |
| high-prior combined (sampling-rescuable) | 17 | 10/17 = 0.5882 | 9/17 = 0.5294 |

Working interpretation: this recipe currently behaves more like generic distribution sharpening than deep scaffold transfer. It strongly recovers validation episodes where base sampling already puts appreciable mass on the unique correct point, and weakly moves low-prior scaffold-only/neither points. Teaching source has no visible explanatory power at this first point.

The BLOCKED menu item 3 is upgraded from parked to specified candidate, not default action:

- Candidate training-side rule: upweight replay by function names or coarse function families derived only from clean training-side data, especially the clean train-failure function set and/or repair-core function set.
- Current diagnostic coverage of the 24 shared newly-wrong fragile episodes:
  - exact function in union repair-core function set: 11/24.
  - coarse family in union repair-core family set: 11/24.
  - exact function in clean train-failure function set: 15/24.
  - coarse family in clean train-failure family set: 16/24.
- Because `R_success_eval` diagnostics have already been inspected, any activation of targeted replay requires a fresh regression arena before using targeted replay as a training-design input. This is now a precondition, not a suggestion.
- Theoretical first-point upside if the 24 shared fragile episodes were protected without adding new failures: main forget would drop from 26/400 to 2/400; STaR forget would drop from 27/400 to 3/400. This motivates keeping targeted replay on the menu despite the leakage caveat.

## v1.7 - 2026-07-02

### Paired-drift mechanism, H-content result, and BLOCKED menu reorder

Timing: recorded after zero-GPU subtype analysis and main second-point monitoring, before any clean-grid model-selection decision and with heldout still untouched.

Mechanism observation (`PROBE`): the 24 shared newly-wrong episodes have paired subtype agreement `24/24` between main and STaR. The two arms have largely different unique teaching sets, yet for each shared fragile episode they induce the same drift subtype: which parameters are dropped, added, or changed is matched episode by episode. Working interpretation: each fragile episode has a base-distribution "nearest wrong neighbor"; generic fine-tuning perturbation pushes it toward that same neighbor regardless of teaching source.

This unifies the current Phase 0 mechanism hypothesis: base probability terrain governs both directions. Transfer lands mainly on base-prior-high failures, while damage lands on low-margin base successes. Teaching source has not shown explanatory power at the weakest point.

H-margin is promoted to the main pending mechanism probe. Prediction, before GPU evaluation: fragile `R_success_eval` episodes will have lower base log-probability on the correct call and smaller margin between the correct call and the observed flipped-to call than non-fragile `R_success_eval` controls.

H-content zero-GPU check did not support the "training outputs are thinner" version of the data-side hypothesis:

- union core training outputs: mean top-level argument count `3.5541`, mean optional-present ratio `0.6515`.
- fragile 24 correct calls: mean top-level argument count `2.9167`, mean optional-present ratio `0.4806`.
- non-fragile `R_success_eval` controls: mean top-level argument count `2.7655`, mean optional-present ratio `0.3424`.
- dropped-parameter occurrences in shared fragile episodes: 16 occurrences, 14 unique `(function, parameter)` pairs. Five of sixteen have zero same-function core rows; when same-function core rows exist, the dropped top-level parameter is present in those rows. This does not look like a simple "training never shows this parameter" story.

Targeted replay ceiling arithmetic is updated: the broadest training-side family rule covers 16/24 shared fragile episodes. Under an unrealistically perfect-protection assumption, main would still have `26-16=10` new wrongs before considering the two main-only errors, or `8/400=0.0200` only in the most optimistic coverage of all main-only errors. Therefore training-side targeted replay alone is unlikely to clear the forgetting gate; it is a combination item, not a first-line standalone fix.

BLOCKED menu reorder:

1. First-line retry family: replay increase plus KL-anchor on replay/base-success inputs. KL-anchor targets the hypothesized cause, margin erosion around base successes.
2. Targeted replay becomes a combination enhancer with a fresh-regression-arena precondition. It may be layered onto replay/KL but should not be used alone unless new evidence changes the ceiling.
3. Further pure replay escalation remains cheap but mechanistically less direct.
4. Frontier negative result remains the final branch if recipe extensions fail.

The forgetting gate remains unchanged at `forget <= 0.02`.

## v1.8 - 2026-07-02

### Dose-order waterline model and second-point frontier observation

Timing: recorded after the matched second clean point for main and STaR returned, before any clean-grid model-selection decision and with heldout still untouched.

The static v1.6/v1.7 reading that transfer landing is fixed by base-prior stratum is partially falsified by the second point. At `ep1`, low-prior transfer was weak (`main=10/139=0.0719`, `STaR=10/139=0.0719`) and scaffold-only transfer was around 11-13%. At `ep2`, only one extra epoch lifts low-prior transfer to `main=44/139=0.3165`, `STaR=37/139=0.2662`, with scaffold-only transfer rising to `main=28/55=0.5091`, `STaR=23/55=0.4182`.

Revised working model (`PROBE`): LoRA applies a dose-dependent global distribution shift. Episodes cross decision boundaries in order of their distance or margin from the boundary. Failure-side transfer and success-side damage are two sides of the same movement: higher dose moves more low-prior failures across the correct boundary, while also moving more low-margin successes into their nearest wrong neighbor.

Frontier endpoints now observed:

- `ep1`: retention preview is below the useful band (`main slice63=13/63=0.2063`) while forgetting already breaks the gate (`main forget=0.0650`, `STaR forget=0.0675`).
- `ep2`: retention preview reaches the useful band (`main slice63=33/63=0.5238`) but forgetting worsens sharply (`main forget=0.1425`, `STaR forget=0.1350`).
- main dose exchange from `ep1` to `ep2`: `delta val_repair=0.2244`, `delta forget=0.0775`, ratio approximately `2.90`.
- STaR dose exchange from `ep1` to `ep2`: `delta val_repair=0.1795`, `delta forget=0.0675`, ratio approximately `2.66`.

C2 monitoring shape (`PROBE`): the second point shows the first C2-shaped separation in the predicted low-prior/scaffold-only region. On scaffold-only validation episodes, main repairs `28/55=0.5091` and STaR repairs `23/55=0.4182`. On low-prior validation episodes, main repairs `44/139=0.3165` and STaR repairs `37/139=0.2662`. This is single-seed monitoring only, not a claim.

C2 caveat recorded before seeing later grid points: STaR also reaches `23/55=0.4182` on scaffold-only validation despite only 31 unique self-sampled teaching episodes. Therefore the transfer-layer claim may need to be phrased as "consistently superior and concentrated in the scaffold-only stratum" rather than categorical access, if 5-seed heldout later supports it. The reach-layer result remains categorical: base sampling produced no training signal for 64/74 scaffold-repaired train episodes.

Approximate net-value note (`PROBE`, with base-rate caveat): using a rough multiple-turn base success rate around 0.65, `ep2` may create positive overall value while breaking the forgetting gate. This makes the problem an explicit decoupling problem rather than a no-signal problem.

BLOCKED first-line implementation spec is added now, before any fallback run: KL-anchor means an added distillation loss on replay/base-success inputs that penalizes divergence from the base model distribution. Candidate small grid: `lambda in {0.5, 1, 2}` crossed with replay ratio `{1:1, 2:1}`. This is a recipe extension and still requires the preregistered fallback path to fail before use.

The forgetting gate remains unchanged at `forget <= 0.02`.

## v1.10 - 2026-07-02

### Clean grid decision, upgrade trigger, and full-grid mechanism observations

Timing: recorded after the full clean grid returned (`24/24` JSONs), before any heldout score for trained checkpoints. Heldout remains untouched for model selection.

Mechanical decision: the preregistered composite criterion has no satisfying point. No clean-grid checkpoint reaches `r_success >= 0.98`; the best observed retention point is `main_r8_lr2e-5_ep1` with `r_success=374/400=0.9350` and `forget=0.0650`, still more than 3x above the `forget <= 0.02` gate.

Fallback record point: `main_r8_lr2e-5_ep1` is recorded as the best-regression fallback, not as an acceptable selected model.

Preregistered upgrade rule is triggered. Each arm takes its highest-repair clean-grid point, halves the learning rate, keeps `epochs=2`, and increases replay to `1:1`:

- main upgrade: from `main_r16_lr5e-5_ep2` (`val=85/156=0.5449`, `r_success=365/400=0.9125`) to `rank=16`, `lr=2.5e-5`, `epochs=2`, `replay=1:1`.
- STaR upgrade: from `star_r16_lr1e-4_ep2` (`val=72/156=0.4615`, `r_success=357/400=0.8925`) to `rank=16`, `lr=5e-5`, `epochs=2`, `replay=1:1`.

C2 full-grid shape observation (`PROBE`, single seed, correlated paired configurations): main beats STaR on aggregate validation repair in all 12 matched configurations. On the validation scaffold-only slice, main is 11 wins and 1 tie. Main also has lower forgetting in 9/12 matched configurations. This is stronger directional evidence than the earlier single-point shape, but it is not a heldout claim and not a substitute for the preregistered 5-seed heldout CI. The transfer-layer wording remains "consistently superior, concentrated in the scaffold-only stratum" rather than categorical access, because STaR also reaches high scaffold-only repair at high dose.

Epoch/lr observation (`PROBE`): the preregistered expectation that `epochs=2` would be more damaging than a higher-lr single epoch is falsified in all four direct comparisons of `5e-5 x ep2` versus `1e-4 x ep1` at matched arm/rank. Five of the six lr-rank cells show `ep2` Pareto-dominating `ep1` by increasing repair while reducing forgetting; the main exception is the most conservative `2e-5` cell. Working hypothesis: replay protection may consolidate over the second epoch, so the frontier direction is "longer and gentler with more replay", not "short and sharp". `ep3` is registered as a possible BLOCKED-menu candidate only; it is not activated by this entry.

Frontier observation (`PROBE`): the main-arm Pareto set has a conservative low-repair point and two high-repair `r16 ep2` points, leaving a gap between `forget=0.065` and roughly `0.085` where no meaningful transfer point exists. The replay-1:1 upgrade is aimed at this gap.

Approximate net-value note (`PROBE`, base-rate caveat): using a rough multiple-turn base success rate near 0.65, the best repair point is likely net-positive overall while still breaking the regression gate. Both facts are recorded: the recipe creates capability but silently damages too much pre-existing capability for the preregistered deployment-style gate.

Operational note: the earlier duplicate STaR rank16 child process on the new westb server was detected, stopped, and produced no duplicate JSON. The final clean grid has exactly the expected 24 JSON outputs.

BLOCKED continuation if the two replay-1:1 upgrade points still break the gate: open the registered BLOCKED path. First-line recipe extension remains replay increase plus KL-anchor on replay/base-success inputs (`lambda in {0.5, 1, 2}` crossed with replay `{1:1, 2:1}`), with targeted replay only as a combination enhancer requiring a fresh regression arena.

## v1.11 - 2026-07-02

### BLOCKED opening, KL-anchor grid, and exit criteria

Timing: recorded after the two replay-1:1 upgrade JSONs returned and before any KL-anchor/BLOCKED run.

Mechanical decision: both preregistered replay-1:1 upgrade points still break the forgetting gate, so the upgrade retry is exhausted and BLOCKED is formally open.

- main upgrade (`r16`, `lr=2.5e-5`, `ep2`, `replay=1:1`): `val=73/156=0.4679`, `slice63=43/63=0.6825`, `r_success=366/400=0.9150`, `forget=34/400=0.0850`.
- STaR upgrade (`r16`, `lr=5e-5`, `ep2`, `replay=1:1`): `val=70/156=0.4487`, `slice63=39/63=0.6190`, `r_success=360/400=0.9000`, `forget=40/400=0.1000`.

Pareto observation (`PROBE`): the main upgrade point is strictly Pareto-dominated by replay-50% clean-grid points. Main clean-grid `r16_lr5e-5_ep2` has `val=0.5449`, `slice63=0.7302`, `forget=0.0875`; main clean-grid `r16_lr1e-4_ep2` has `val=0.5256`, `slice63=0.7302`, `forget=0.0850`. Replay increase helped only marginally on forgetting relative to the `r16_lr2e-5_ep2` neighbor, while learning-rate halving cost too much transfer. Conclusion: replay scaling alone is not expected to close the roughly 4x remaining forgetting gap; KL-anchor is now the last untried first-line mechanism.

C2 monitoring update (`PROBE`): the upgrade pair preserves the main-over-STaR direction on aggregate validation repair (`+3/156`) and scaffold-only repair (`38/55` vs `33/55`). Counting the 12 clean-grid matched configurations plus this upgrade pair, the monitoring tally is now `13/13` for aggregate main-over-STaR. This remains monitoring-only; heldout 5-seed CI is still the adjudicator.

NaN caveat: the main upgrade training log contains one `loss=nan` at epoch 2 step 50/74, followed by a finite logged loss, successful adapter save, and completed evaluation. The checkpoint remains caveated and its exact metric values are not used for fine-grained interpretation. Before KL-anchor runs, the training loop must include:

1. non-finite-loss guard: skip the batch/step, clear accumulated gradients, and log the skip;
2. non-finite-gradient guard after gradient clipping: skip optimizer update and clear gradients;
3. explicit grad-clip threshold logging;
4. finite checks in the KL log-softmax path.

Heldout pass@16 partition is now measured for the current harness: `158` heldout failures, teacher repairs `50/158=0.3165`, scaffold-only `47/158=0.2975`, sampling-rescuable `10/158=0.0633`, neither `101/158=0.6392`. Among teacher-repaired heldout failures, `47/50=0.9400` are scaffold-only. This makes the C2 reach layer in the main arena more extreme than the train-side `64/74=0.8649`. The retention denominator for this harness is now the measured `0.3165`, not the EXP2 five-loop mean `0.2405`; the higher denominator makes retention harder to pass and is therefore conservative against us.

Log-prob probe status: current results directionally support the C2 base-prior prediction (`train teacher scaffold-only mean=-1.1114` vs `train teacher sampling-rescuable mean=-0.9808`) and give mixed/weak support for the first H-margin prediction (`first-shared fragile=-1.1728` vs non-first-shared `-1.1376`). The truncation caveat is recorded: `14/400` `R_success_eval` records had no effective scored target token at `max_length=2048`; the fragile-24 set is unaffected. The following registered analyses must be computed from the existing scored probe before interpretation is considered complete:

1. H-margin prediction 2: `log P(correct) - log P(observed flipped-to output)` for the fragile-24 actual flips.
2. Dose-order prediction: base log-prob should order `ep1 transferred > ep2-new transferred > not transferred`.
3. Scaffold-only internal transfer-success contrast: successful transfer should have higher base log-prob than failed transfer within the scaffold-only slice.

BLOCKED first-line KL grid is now fixed, symmetric across arms and parity-preserving:

- main base config: clean-grid repair max `rank=16`, `lr=5e-5`, `epochs=2`.
- STaR base config: clean-grid repair max `rank=16`, `lr=1e-4`, `epochs=2`.
- KL-anchor grid: `lambda in {0.5, 1, 2}` crossed with `replay in {1:1, 2:1}` for each arm, `6` runs per arm, `12` total.
- Replay `2:1` may require repeated replay rows because the unique replay pool is finite. Repeat weighting is allowed and must be reported with row and unique counts, matching the already recorded repeat-weight structure of the core datasets.

BLOCKED ep3 candidate is activated as an explicit mechanism check: each arm runs one additional no-KL point at its repair-max config with `epochs=3` and `replay=1:1`. This tests the consolidation hypothesis that longer gentle training with replay may further reduce forgetting. Total BLOCKED first wave: `14` runs.

BLOCKED exit criteria, fixed before any KL number returns:

- Success: any point satisfies `forget <= 0.02` and `slice63 >= 0.40`. That point enters the selection flow; the same config is run as paired 5-seed arms and then evaluated on heldout.
- Partial: any point satisfies `forget <= 0.02` but `slice63 < 0.40`. Allow exactly one combination round: best lambda crossed with best replay plus ep3, at no more than two runs per arm.
- Failure: after the combination round, if no qualifying point exists, hold a negative-result decision review. The paper path switches to the internalization-forgetting frontier for 7B plus this recipe family, with C2 and mechanism analyses retained. No second combination round and no new unregistered lever may be added at that point.

The forgetting gate remains unchanged at `forget <= 0.02`.
