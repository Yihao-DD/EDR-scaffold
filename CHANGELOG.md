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

## v1.18 - 2026-07-02

### Phase 0 final-evaluation preconditions and monitoring-surface caveat

The prior 5-seed report over `val@156 + old R_success@400` is explicitly classified as a monitoring-surface confirmation, not the Phase 0 final evaluation. It is useful for frontier calibration and seed-level sanity checks, but the preregistered final C1/C2 claims require heldout `158` plus the sibling regression arbitration surface.

Monitoring-surface observations recorded before final heldout evaluation:

- Aggregate validation main-vs-STaR differences are not statistically settled at 5 seeds and include a seed-level reversal. Therefore no report may claim that main wins STaR on aggregate validation repair.
- The predicted cross-stratum shape appears in monitoring: STaR can match or exceed main on sampling-rescuable/high-prior slices, while main's advantage is concentrated in scaffold-only/low-prior slices. This remains a monitoring observation until the heldout scaffold-only bootstrap.
- Forgetting on the old `R_success@400` surface is seed-stable around the 5-7% band, far above the 2% gate. This supports the irreducible-interference interpretation but is not the final regression arbitration surface.

Final-evaluation protocol is fixed as pure forward evaluation of the frozen 20 checkpoints: heldout `158`, retention ratio and conditional-recovery denominators, teacher agreement, per-arm SS/SN/NS/NN labels, five regression surfaces, and episode-level paired bootstrap on heldout scaffold-only `47` as the unique C2 adjudicator.

## v1.19 - 2026-07-02

### Judge freeze, sibling arena lock, and teacher denominator re-anchor

Parallel judge manual audit is confirmed by Ian under the judgment-dense-step rule. The audit checked both parsing fidelity and pass/fail matching logic for 15 base outputs, with special attention to fail cases and `live_parallel_multiple`. No parsing or matching defect was found. The judge is frozen for s07 final evaluation.

Freeze identifiers:

- git HEAD at freeze time: `f3dcc4f2b628c355498d29d1c1e45088ae7e8fa5`
- `EDG-EXP3-distill/scripts/v17_parallel_arena.py` sha256: `761c6ab312d97fdb840ed7c2731c5ef63312359a32683f866d88c5c375e753aa`
- manual audit file: `EDG-EXP3-distill/logs/v17_parallel_judge_manual_audit.md`, status `CONFIRMED`

Sibling arena decision tree lands mechanically in route A: never-touched parallel-family base successes are sufficient.

- parallel-family raw: `440`; base successes: `304` (`0.6909`)
- simple_python unrecorded raw: `241`; base successes: `134` (`0.5560`)
- arena source: parallel-family successes only
- arena size: `300`
- arena category composition: `parallel=146`, `parallel_multiple=134`, `live_parallel=11`, `live_parallel_multiple=9`
- five asserts are zero: arena intersect all historical training rows, old `R_success@400`, capped-106, `D_val`, and `D_heldout`

Teacher heldout denominator is re-anchored in the pinned s07 evaluation environment. The re-run returns the same denominator as the prior pass@16 partition file:

- old denominator: `50/158 = 0.3164557`
- pinned-environment re-anchor: `50/158 = 0.3164557`
- denominator for Gate 0 retention ratio remains `50/158`; both old and re-anchored values should be reported.

Two missing-adapter recoveries are completed and provenance-checked:

- `main rep1 seed20260704` v2 recovery: `val_repair=0.5192`, `r_success=0.9275`, exactly matching the original JSON.
- `STaR rep2 seed20260703` v2 recovery: `val_repair=0.3013`, `r_success=0.9425`, exactly matching the original JSON.

The non-finite-loss skip issue is reclassified from stochastic instability to deterministic row exclusion: static tokenization predicts the exact skip counts for replay1 and capped replay2 datasets. The relevant caveat is effective training row count, not random numerical corruption.

With these preconditions satisfied, the s07 final forward-only batch is authorized: 20 frozen checkpoints over heldout `158` and sibling arena `300`, followed by the preregistered C2 paired bootstrap.

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

## v1.15 - 2026-07-02

### Saturated-replay interference, v1.14 wording fixes, and D2 launch spec

Timing: recorded after the capped-arena overlap audit and before D2 heldout 5-seed training/evaluation. This entry supersedes three v1.14 wordings without changing the mechanical D1 failure or D2 representative choices.

Saturated-replay observation upgraded: s05 `replay2 lambda2` checkpoints trained on all `106/106` capped-arena episodes (`main` row overlap `138`, `STaR` row overlap `138`) while using the strongest tested KL anchor (`lambda=2`). Even under this saturated replay-plus-KL condition, main still missed `4/106` and STaR missed `5/106`, and every missed ID was present in that checkpoint's training rows. The hard observation is no longer merely "coverage does not imply protection"; it is: saturated replay exposure plus KL anchoring still leaves a residual interference core. The mechanistic explanation remains `PROBE`, but the sample-in observation itself is hard.

Mechanistic implication (`PROBE`): the BLOCKED-stage failures are consistent with a structural conflict between repair gradients and already-correct behavior. Replay scaling, KL anchoring, and targeted replay all operate by feeding or constraining protected examples; the saturated-replay miss set shows that protection dose is not sufficient for some boundary-near successes.

Drift subtype check on saturated-replay misses:

| source | missed IDs | subtype summary |
|---|---:|---|
| s05 main replay2 lambda2 | 4 | value `3`, drop `1` |
| s05 STaR replay2 lambda2 | 5 | value `2`, drop `2`, function flip `1` |

The recurring clean capped-arena watch set is also mostly parameter/value level rather than format level: `live_multiple_1003-232-2` is consistently drop; `live_multiple_453-145-4`, `live_multiple_469-145-20`, and `live_multiple_956-203-0` are consistently value drift; `multiple_178` is mostly add, with one parse-error outlier. Existing local log-prob probes do not contain these capped-arena IDs, so base-margin scoring for this exact set remains a pending forward probe rather than an already available zero-GPU result.

Distributional fragility note: capped-arena wrongs have `0` overlap with the old fragile-24 set, but capped wrongs recur across clean capped D3/C configurations. This supports the interpretation that fragility is a distributional boundary layer: different regression surfaces reveal different fragile points, while each surface can still show stable cross-arm weak spots.

v1.14 wording fixes:

1. Targeted replay verdict: the binary failure stands (`C main=6/106`, `C STaR=4/106`, both above the capped `2/106` gate), and the earlier "targeted worsened fresh forgetting" wording remains withdrawn. However, the statement "no detectable effect" carries the targeted-run numeric caveat because targeted C had substantial non-finite-loss skipping (`main=36`, `STaR=54`). The valid statement is: no reliable improvement was detected, and the targeted runs are numerically caveated.
2. rep2 selection: D3 `ep3 replay2 lambda2` is selected using capped-arena numbers even though capped arena is `PROBE`/reference. This is intentional and not a gate claim: `PROBE` numbers may be used for representative selection, while final regression authority is reserved for D2 `forget_heldout`.
3. Coverage/protection status: "coverage does not imply protection" is promoted from a weak `PROBE` observation to a hard sample-in observation under saturated replay. The mechanistic interpretation of an irreducible interference core remains `PROBE` until D2 provides a fresh heldout regression surface.

D2 representatives remain locked:

- rep1, best transfer: main `main_ep3_r16_lr5e-5_replay1_seed20260703`; STaR `star_ep3_r16_lr1e-4_replay1_seed20260703`.
- rep2, valid lowest capped forgetting: main `main_d3c_r16_lr5e-5_ep3_replay2_lam2_seed20260703`; STaR `star_d3c_r16_lr1e-4_ep3_replay2_lam2_seed20260703`.

D2 numerical hygiene spec: every D2 run must report `skipped_nonfinite_loss`, `skipped_nonfinite_grad`, `kl_anchor_batches`, train seconds, and whether eval-only recovery was used. Existing s05 rep1 logs show that main ep3 required eval-only recovery after a staged-server missing-path failure, and current logs do not preserve a final exact skip total for both rep1 runs. Therefore D2 must record skip counts directly from per-seed `train_metadata.json`; summaries without skip counts are incomplete.

D2 adjudication remains:

- C1 final wording is determined by heldout repair plus `forget_heldout`.
- C2 final wording is determined by heldout scaffold-only paired comparison over 5 seeds.
- capped arena is retained as exploration/reference only.
- old `R_success_eval` is continuity only.
- no additional D1 recipe exploration is authorized.

## v1.16 - 2026-07-02

### D2 preflight audit, forget-heldout demotion, and sibling-arena blocker

Timing: recorded before any D2 5-seed training/evaluation. This entry supersedes the v1.15 statement that `forget_heldout` could adjudicate final regression.

D2 Step 0 audit artifact: `results/v16_d2_step0_audit.json`, generated by `scripts/v16_d2_prep.py`. The script reads local artifacts only and does not use GPU.

Main repair arena remains valid: the frozen heldout failure list has `158` episodes and is unchanged. All repair-side claims continue to use this frozen list plus the existing pass@16 partition labels.

Success-side regression correction: `forget_heldout` is demoted from adjudicator to non-adjudicating decomposition/continuity metric. The heldout-all base-success slice has `163` episodes, but it decomposes exactly as:

- `65` intersect historical training/replay rows;
- `98` intersect old `R_success_eval`;
- `0` residual untouched successes after removing those two sources.

Therefore `forget_heldout` has no fresh regression information. The `65` slice is renamed `trained_success_retention` and is mechanistic only; the `98` slice is a continuity mirror of old regression and may not carry gate authority.

Specification bug #6: success-side pools were selected from all multiple base successes without excluding the heldout region because the heldout artifact tracked failures only. This is a document/specification bug, not an execution error. The failure-side heldout assertions still protected the main repair arena.

151/158 reconciliation: the frozen heldout failure list (`158`) matches the failure-only deterministic split exactly (`extra=0`, `missing=0`). The `151` count arises from a different construction: split all multiple records first, then count base failures inside that heldout-all slice. The difference of `7` is therefore a split-universe difference, not evidence that the frozen 158 list drifted or that hardware changed base behavior.

Teacher heldout output check: `results/s01_heldout_pass16_partition.json` contains teacher outputs for all `50/50` teacher-success heldout failures. No missing teacher-output backfill is required before wrapper dry-run.

Category touch table summary:

| category | status | single-call compatible | raw rows | unrecorded raw rows |
|---|---:|---:|---:|---:|
| live_multiple | train_touched | yes | 1053 | 0 |
| multiple | train_touched | yes | 200 | 0 |
| live_simple | eval_touched_only | yes | 258 | 0 |
| simple_java | eval_touched_only | yes | 100 | 0 |
| simple_javascript | eval_touched_only | yes | 50 | 0 |
| simple_python | eval_touched_only | yes | 400 | 241 |
| live_parallel | never_touched | no | 16 | 16 |
| live_parallel_multiple | never_touched | no | 24 | 24 |
| memory | never_touched | no | 155 | 155 |
| multi_turn_base | never_touched | no | 200 | 200 |
| multi_turn_long_context | never_touched | no | 200 | 200 |
| multi_turn_miss_func | never_touched | no | 200 | 200 |
| multi_turn_miss_param | never_touched | no | 200 | 200 |
| parallel | never_touched | no | 200 | 200 |
| parallel_multiple | never_touched | no | 200 | 200 |
| web_search | never_touched | no | 100 | 100 |

Sibling-arena blocker: under the current single-call evaluator, there are `0` never-touched single-call-compatible sibling rows. The only downgrade candidate is `simple_python` unrecorded raw rows (`241`), but the category is already eval-touched and the candidate count is below the requested `n >= 300` before base-success filtering. Therefore `forget_sibling` is not constructible as a clean adjudicating arena with the current evaluator and current data.

D2 launch status: D2 5-seed training/evaluation is not launched from this state. Before D2 can carry a deployment regression gate, one of the following must be preregistered: (1) a new evaluator for never-touched multi-call/multi-turn/web-search sibling categories, or (2) an explicitly downgraded non-adjudicating/probe regression surface such as unrecorded `simple_python`, with its limitations stated up front.

## v1.17 - 2026-07-02

### D2 dual-lane launch and composite regression adjudication

Timing: recorded after the v1.16 blocker and before any v1.17 D2 checkpoint metrics are read.

Decision: use a Route 1+2 composite regression adjudication surface and pipeline it with Route 3. D2 training may start immediately; the regression arena is constructed in parallel and then evaluated on the frozen D2 checkpoints by pure forward pass. This is scientifically equivalent because representative points and D2 seeds are frozen before arena metrics are observed.

Rejected single-route alternatives:

- Route 1 only was rejected because implementing a new parallel-family judge is necessary but may not by itself produce enough base-success adjudication examples.
- Route 2 only was rejected because `simple_python` unrecorded rows are episode-untouched but category eval-touched and therefore cannot alone carry a clean deployment regression claim.
- Route 3 only was rejected because delaying all D2 training until the arena is complete wastes training-card time; arena evaluation is a pure-forward measurement on already frozen checkpoints.

D2 training lane A:

- New runs: `16` train/eval points, seeds `20260704..20260707`.
- Reused checkpoints: the four seed `20260703` representatives already produced in s05/s06.
- Representative definitions:
  - `rep1`: both arms use the s05 `ep3 replay1` configuration and original replay1 pool.
  - `rep2`: both arms use the D3 `ep3 replay2 lambda=2` configuration and capped replay2 pool.
- Numeric hygiene required per D2 seed: `skipped_nonfinite_loss`, `skipped_nonfinite_grad`, `kl_anchor_batches` for rep2, `train_seconds`, and whether eval-only recovery occurred.
- s05 rep1 numerical-history check remains required: original main/STaR ep3 replay1 logs must be inspected and any non-finite or recovery history attached to rep1's file card.

Launched queue artifact: `tmp_remote_scripts/run_s07_d2_queue.sh`. It serializes four new seeds for one queue, writes results under `results/s07_d2_5seed/{main,star}/`, writes adapters under `adapters/s07_d2_5seed/{main,star}/`, and performs eval-only recovery if an adapter exists without a JSON result.

Training queue assignment:

- new a3: `main_rep2`;
- new a4: `star_rep2`;
- old westb: `main_rep1`;
- old westc: `star_rep1`.

Evaluator/arena lane B:

- New script: `scripts/v17_parallel_arena.py`.
- Judge scope: parallel-family differential judge only. Covered categories are `parallel`, `parallel_multiple`, `live_parallel`, and `live_parallel_multiple`.
- Output semantics: prediction must parse as a list of function calls; predicted call count must equal ground-truth call count; matching is order-insensitive; each individual call uses the existing single-call AST/value semantics.
- This is a differential project judge for this regression surface only. It is not claimed as official BFCL parallel scoring.
- Multi-turn, memory, and web-search categories remain out of scope for v1.17.

Judge smoke test passed before base-forward evaluation:

- order-insensitive two-call success;
- wrong call count failure;
- wrong argument failure;
- nested dictionary argument success.

Raw candidate pool before base-success filtering:

- parallel family: `440` rows (`parallel=200`, `parallel_multiple=200`, `live_parallel=16`, `live_parallel_multiple=24`);
- `simple_python` unrecorded candidate: `241` rows;
- total candidate rows: `681`.

Arena construction rules:

- Run base `T=0` on the `681` candidate rows after the judge smoke and human spot-check are complete.
- Freeze `simple_python` unrecorded episode IDs into the resulting artifact.
- Human spot-check: inspect `10-15` base outputs from the parallel-family judge before freezing the judge; record the review in logs. This is a judgment-dense step and requires a person in the loop.
- Judge freeze: once spot-check passes, record the git hash in this changelog before evaluating any D2 checkpoint on the new arena.
- Five intersection asserts for the final arena: zero intersection with all training rows, old `R_success_eval` 400, capped106, `D_val`, and `D_heldout`.

Final regression report structure is five-faced:

- `forget_sibling`: composite adjudication surface from v1.17 lane B;
- old 400: continuity only;
- capped106: PROBE/reference only, with known invalidity for historical non-capped rep1 where applicable;
- trained-success-retention 65: mechanistic only;
- heldout-98: continuity mirror only.

Heldout repair and C2 evaluation remain unchanged from v1.16: the 158 heldout failure arena, retention double reporting, teacher agreement, per-arm 2x2 tables, and scaffold-only paired bootstrap remain the final repair-side/C2 judge.

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

## v1.20 - 2026-07-02

### s07-final quarantine, rep1 adjudication, and rep2 preservation

Status: the previously generated `s07-final` table is placed under quarantine for Gate/C1/C2 interpretation until the rep1 provenance issue is resolved. The table may be used only as an audit object, not as the final Phase 0 decision table. This entry records the adjudication evidence gathered after the inconsistency was found.

Reason for quarantine: three contradictions are concentrated in rep1 while rep2 remains internally consistent.

1. Dry-run versus final mismatch: main rep1 seed `20260703` dry-run reported `80/158` heldout repairs, but the final 5-seed rep1 mean was only `51.4/158`. STaR rep1 dry-run reported `74/158`, but the final seed `20260703` JSON reports `10/158`.
2. Rep1 aggregate C2 reversal: the final rep1 aggregate bootstrap favored STaR over main, contradicting the dry-run direction and the earlier validation monitoring shape.
3. Rep1 cross-arena inconsistency: validation monitoring showed a stable main scaffold-only advantage, while heldout rep1 did not. Rep2 did not show this inconsistency.

Adjudication experiment and evidence:

- Main rep1 seed `20260703` was re-evaluated in the pinned final evaluation environment and reproduced the dry-run and final JSON exactly: `80/158` heldout repairs, retention ratio `1.600`, partition repairs `scaffold_only=37/47`, `sampling_rescuable=8/10`, `neither=35/101`. Per-episode success diffs versus the dry-run JSON were `0`.
- Main rep1 seed `20260704` was re-evaluated and reproduced the low final result: `9/158` heldout repairs, partition repairs `scaffold_only=4/47`, `sampling_rescuable=3/10`, `neither=2/101`. The adapter file exists and has a complete adapter size; this is a true bad/no-op-like recovered checkpoint under the current artifact, not a missing-adapter fallback.
- Rep1 training data hashes are consistent across seeds. Main rep1 seeds use `data/distill_main_replay1.jsonl` with SHA256 `f9af3ffb30c0d58657dadccb6300241306af51889b547aa9421dc444f61031d2`. STaR rep1 seeds use `data/distill_star_replay1.jsonl` with SHA256 `2028e65067ff3ccf44b1d0509e8345a8cde2e0e9085f835dd3541669bc7be99b`. The rep1 issue is not explained by seeds `20260704..20260707` using the wrong replay pool.
- STaR rep1 seed `20260703` is provenance-broken. The dry-run JSON reports `74/158`, but the current adapter at the same path reproduces the final result `10/158` when re-evaluated. The current adapter checksum is stable across available machines, and the evaluation script plus heldout partition inputs match. No alternate full adapter matching the dry-run behavior was recovered. Therefore the most likely explanation is that the adapter artifact at that path changed or was overwritten between dry-run and final evaluation. STaR rep1 seed `20260703` must not be used in any clean 5-seed rep1 claim unless it is retrained or the original dry-run artifact is recovered and checksum-verified.
- Main and STaR rep2 dry-run versus final heldout evaluations are per-episode identical where checked. Rep2 remains the clean C2 adjudication line.

Preserved results:

- Rep2 C2 remains valid under the current evidence. On heldout scaffold-only `47` episodes, the episode-level paired bootstrap for rep2 gives main minus STaR `+0.1532` with 95% CI `[+0.0638, +0.2511]`, not containing zero. This supports C2 in the preregistered scope: the main arm's advantage is concentrated in the sampling-inaccessible scaffold-only stratum.
- The no-regression gate remains failed under all audited branches. The best sibling-forget estimate in the quarantined table is still above the hard gate (`forget <= 0.02`), so the C1 deployment-style claim remains: high-retention internalization and net repair are observed, but no no-regression point has been established within this recipe family.

Interpretation policy until repair:

- Do not cite the full `s07-final` table as final.
- Do not use rep1 aggregate bootstrap, rep1 retention means, or rep1 arm comparisons as clean claims.
- Rep1 may only be discussed as a provenance/case-study audit until STaR seed `20260703` is retrained or the original adapter artifact is recovered.
- Rep2 may remain in the final evidence set because it passes dry-run/final reconciliation and preserves the preregistered heldout scaffold-only C2 result.

Process correction:

1. Every final evaluation table must include a dry-run reconciliation row for reused checkpoints before being interpreted.
2. Every reused checkpoint JSON must record the resolved adapter path, adapter SHA256, adapter file size, loaded trainable-parameter count if available, data file hash, and evaluation script hash.
3. Any reused checkpoint evaluated on a different machine must be checksum-verified before evaluation. If an artifact path is reused after recovery or resync, the new artifact must receive a versioned provenance label instead of silently occupying the original path.

## v1.22 - 2026-07-03

### rep2 handoff branch and Phase 2 round-2 cold-start package

Scope: branch `handoff/phase2-round2` is a rep2-only handoff package. Rep1 quarantine and repair artifacts are excluded from this branch. The handoff instruction text is preserved at `docs/PHASE2_HANDOFF.md`.

Rep2 reproduction package:

- Package root: `repro_rep2/`.
- Core data: `repro_rep2/data/distill_main_core.jsonl`, 148 rows.
- Training data: `repro_rep2/data/distill_main_replay2_capped.jsonl`, 444 rows.
- Frozen episode-id lists: D_val 156, D_heldout 158, old400 400, capped106 106, sibling300 300.
- Code: Phase 0 LoRA train/eval, heldout/sibling evaluators, frozen parallel judge, capped-arena evaluator, pass@16/judge support code, and focused pytest asserts.
- Artifacts: five main rep2 adapters plus training JSON, heldout JSON, sibling JSON, and provenance logs under `repro_rep2/artifacts/main_rep2_seed*/`.
- SHA inventory: `repro_rep2/MANIFEST.md`, `repro_rep2/MANIFEST.json`, and `repro_rep2/SHA256SUMS`.
- Large binaries: adapter safetensors are tracked via Git LFS (`*.safetensors`, `*.bin`, `*.pt`).

M1 designation rule and result:

- Rule: among the five main rep2 seeds, choose the median heldout repair count; ties choose the smaller seed.
- Heldout repair counts: seed20260703 `54/158`, seed20260704 `51/158`, seed20260705 `46/158`, seed20260706 `58/158`, seed20260707 `43/158`.
- `M1_DESIGNATED`: seed `20260704`.
- M1 adapter path: `repro_rep2/artifacts/main_rep2_seed20260704/adapter/adapter_model.safetensors`.
- M1 adapter SHA256: `c3afb185a6a32480a55e678599b3bebc8b8899ed935d620605b00ab3c3e62683`.
- Reference numbers for company reconciliation: heldout repair `51/158`, sibling success `292/300`.

Phase 2 round-2 decisions locked in code:

1. Recipe locked: rank `16`, lr `5e-5`, epochs `3`, replay `2:1`, KL lambda `2`, non-finite guards on.
2. KL anchor is M1, not raw M0. Implementation: `repro_rep2/scripts/lora_phase0.py` accepts `--base-adapter-dir`, merges A1/M1 into base weights, then attaches the new A2 LoRA; the existing KL disable-adapter path therefore evaluates frozen M1.
3. New seed block: `{20260708, 20260709, 20260710, 20260711, 20260712}`.

Round2 cold-start scripts added under `round2/`:

- `collect_failures.py`: enumerate or collect F2 on the train share under M0+A1.
- `loop/evolution_loop_round2.py`: thin wrapper for the vendored EDG-EXP2-struct NL-evo loop, with D_val AST-only acceptance.
- `build_t2.py`: build T2 with AST filtering, dedupe, material-exhaustion marker, and leakage asserts.
- `build_replay2.py`: construct M1-success replay at 2:1 with eval-surface exclusion asserts.
- `train_round2.py`: locked A2 training launcher with M1-anchor invariant.
- `eval_round2.py`: M2 evaluation launcher for heldout/sibling surfaces and extension points for val/old400/teacher2.
- `reconcile.py`: company acceptance smoke; prints `ACCEPTED` when manifest references are in tolerance and tests pass.

Review result:

- Local compile check passed.
- Focused pytest for rep2 data/leakage asserts passed: `2 passed`.
- Dry-run chain passed for `collect_failures -> loop -> train_round2`.
- Eval dry-run command rendering passed.
- `round2/reconcile.py --skip-pytest` printed `ACCEPTED`.
- Full review notes are in `docs/REVIEW.md`.

Push note: the final pushed branch HEAD is reported in the handoff response because a commit cannot reliably embed its own final hash without changing that hash.

## v1.23 - 2026-07-03

### round2 handoff audit fixes after external review

External review found several real round2 cold-start defects. This entry records
the fixes made on `handoff/phase2-round2` after the original v1.22 push.

Fixed:

- M2 evaluation no longer drops A1. `repro_rep2/scripts/lora_phase0.py`,
  `s07_heldout_eval.py`, `s07_sibling_arena_eval.py`, and `round2/eval_round2.py`
  now support/pass `--base-adapter-dir`, so evaluation can load
  `M0 + A1 + A2` instead of raw `M0 + A2`.
- `build_t2.py` is AST fail-closed. Rows missing `ast_pass` now assert with
  examples instead of silently entering T2.
- `build_replay2.py` now derives its default target size from `2 * |T2|` when
  `--t2-jsonl` is supplied, and every replay row is stamped with
  `source=replay_base_success` and `partition=replay` so the KL-anchor replay
  path fires.
- `round2/common.py` no longer lets rows with missing/falsy `episode_id` bypass
  leakage asserts.
- `round2/collect_failures.py` now has a real full mode: it loads M1, runs T=0
  over the train share, writes F2 failures, and reports F1-vs-F2 composition
  summaries.
- `round2/reconcile.py` no longer compares manifest values to themselves. Full
  `ACCEPTED` now requires model re-evaluation of M1 on heldout and sibling
  surfaces. `--skip-model-check` is package-only and prints `PACKAGE_ONLY`.
- `round2/loop/evolution_loop_round2.py` now points at the real
  `EDG-EXP2-struct/scripts/evolution_loop.py` and passes its actual
  `--input/--output/--log/--model-id/--seeds` interface rather than the
  nonexistent `evolution_main.py`/`--adapter` interface.
- Focused pytest now includes round2 contract tests for M2 eval stack rendering,
  replay KL markers, and episode-id fail-closed behavior.

Remaining explicit caveat:

- The upstream EXP2 patch loop is not LoRA-adapter-aware. The repaired round2
  path uses M1 for F2 collection, then runs the patch loop on F2. Conditioning
  patch search itself on M1 remains a separate implementation task before
  production Phase 2 use.

Local checks:

- `python3 -m compileall -q round2 repro_rep2/scripts`
- `python3 -m pytest -q repro_rep2/tests/test_repro_rep2_asserts.py repro_rep2/tests/test_round2_contracts.py`
  returned `5 passed`.

## v1.24 - 2026-07-03

### EXP2 loop adapter-aware repair for Phase 2 handoff

The v1.23 remaining caveat is resolved. Patch search is no longer run through
the raw upstream EXP2 loop.

Fixed:

- Added `round2/loop/evolution_loop_m1.py`, a vendored EXP2 loop whose only
  model-loading path is `base -> PeftModel(A1) -> merge_and_unload()` in memory.
  It resolves A1/M1 from `repro_rep2/MANIFEST.json` unless
  `--base-adapter-dir` is supplied, and exits if no M1 adapter is available.
- `round2/loop/evolution_loop_round2.py` now delegates to the M1-aware vendored
  loop. The dry-run command no longer points at
  `EDG-EXP2-struct/scripts/evolution_loop.py`.
- `round2/collect_failures.py` now uses the same merged-M1 eval path and writes
  complete F2 episode records plus a `failures` list consumable by the loop.
- Added `round2/loop/no_patch_equivalence_smoke.py`. On a GPU execution host it
  checks that the loop entry, with no patch context, exactly matches
  `collect_failures.py` raw outputs for the first 10 F2 episodes.
- `round2/build_t2.py` can now generate teacher-2 samples directly from F2 plus
  H2 loop output using the same M1-aware loop entry:
  `teacher-2 = merged(M1) + H2 patch-in-context`.
- Added `round2/loop/teacher2_forward.py` and wired `round2/eval_round2.py` so
  teacher-2 heldout forward also uses the same M1-aware entry.
- `docs/PHASE2_HANDOFF.md`, `round2/README.md`, `round2/loop/README.md`, and
  `docs/REVIEW.md` now state that any raw-M0 patch search or validation run is
  invalid.
- Added tests for the loop contract: wrapper dry-run must call
  `round2/loop/evolution_loop_m1.py`, and the vendored loop must include
  `PeftModel.from_pretrained`, `merge_and_unload()`, and the raw-M0 guard.

Local checks:

- `python3 -m compileall -q round2 repro_rep2/scripts/lora_phase0.py repro_rep2/scripts/s07_heldout_eval.py repro_rep2/scripts/s07_sibling_arena_eval.py`
- `python3 -m pytest -q repro_rep2/tests/test_repro_rep2_asserts.py repro_rep2/tests/test_round2_contracts.py`
  returned `7 passed`.
- Static grep over `round2/loop` found one model load path and one direct
  `model.generate` site, both inside `round2/loop/evolution_loop_m1.py`.

## v1.25 - 2026-07-03

### round2 missing-producer closure and dynamics scriptization

Second external review found two pipeline files that were consumed but never
produced (so a literal cold-start would stop at replay build and heldout eval),
plus Gate 2 / retention_2 that were prose rather than code. All resolved.

Fixed:

- Replay source pool is now produced. `round2/build_m1_success.py` writes
  `m1_train_success.jsonl` as `M1 T=0 @ {capped-120 ∪ eliminated}` keeping only
  V=1, reporting the V=0 exclusion count, and asserting disjointness from
  heldout/D_val/old400/sibling300. `capped-120` is frozen as
  `repro_rep2/data/episode_ids/capped120_replay.json` (the 120 unique round-1
  replay episodes, verified disjoint from every eval surface). Dumping only
  `eliminated` is explicitly disallowed. `collect_failures.py` now emits
  `eliminated_episode_ids` to feed the pool.
- `t1_t2.jsonl` is now produced. `build_t2.py` unions `distill_main_core`
  (148 rows, 74 episodes) with T2, deduped by `episode_id`, and `assert_main_arm`
  confirms no STaR fork by inspecting only role fields (source/arm/round2_source),
  never free text.
- Gate 2 is scriptized. `round2/classify_gate2.py` classifies
  compound/converge/collapse by a 5-seed paired-bootstrap CI on heldout repair
  plus the forget gate; classification is mechanical (no narrative override).
- `retention_2 = repair(M2)/repair(M1+H2)` is now written by `eval_round2.py`
  to `<prefix>.retention2.json` after the M2 heldout and teacher-2 forwards.
- `docs/PHASE2_HANDOFF.md` Gate 2 / retention_2 language now points at the
  scripts rather than restating the rule in prose.

Tests:

- KL reference lock: `test_kl_reference_is_merged_m1_not_raw_m0` asserts the
  merge-before-apply-lora ordering, the `disable_adapter` KL path, and the
  `merged_base_adapter` metadata; `test_train_round2_dry_run_anchors_kl_to_m1`
  asserts the launcher passes `--base-adapter-dir M1` with `--kl-anchor-lambda 2`.
- Input-file existence: `test_build_t2_produces_t2_and_t1_t2` and
  `test_build_m1_success_dry_run_produces_replay_source` close the
  command-string-only blind spot by asserting the files themselves are written.

Local checks:

- `python3 -m compileall -q round2 repro_rep2/scripts/lora_phase0.py repro_rep2/scripts/s07_heldout_eval.py repro_rep2/scripts/s07_sibling_arena_eval.py`
- `python3 -m pytest -q repro_rep2/tests/test_repro_rep2_asserts.py repro_rep2/tests/test_round2_contracts.py`
  returned `11 passed`.
- Dry-run walk of the full `round2/README.md` pipeline: every consumed
  intermediate now has a producing step.
