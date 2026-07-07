# BASELINES — every arm, every reference number, and where it comes from

Audience: someone executing or auditing Phase 1/2 without the research
history. This file answers "what is this baseline, and why is it there".

## 0. The task, the model, the judge

- **Task/benchmark**: BFCL v4 (Berkeley Function Calling Leaderboard),
  Multiple-category function calling. The benchmark data is vendored at
  `EDG-EXP1/data/raw/bfcl/` (episodes + `possible_answer` ground truth).
  Nothing is fetched from the network at run time.
- **Base model** `M0`: Qwen/Qwen2.5-7B-Instruct, frozen. All capability
  changes live in LoRA adapters loaded on top.
- **The only judge** `V`: the BFCL AST matcher (function choice correct ∧
  required arguments present ∧ argument values inside the accepted answer
  set). It is a task-level oracle against human ground truth — NOT a format
  check. No LLM judge exists anywhere in data filtering, validation,
  evaluation, or partitioning; this is a frozen project rule.
- **Scaffold** `H1`: 39 accepted natural-language patches produced by the
  EXP2 self-evolution loop (diagnose → generate patch → AST-verify → accept).
  Frozen at `EDG-EXP2-struct/results/evolution_main_merged.json`. When the
  scaffold is "on", relevant patches are injected into the prompt
  (`[PATCH i] ...`), costing ~12.5k tokens per episode on average.

## 1. Data splits (frozen; leakage enforced by code asserts)

| split | size | role |
|---|---|---|
| `D_train` share | 313 failures | failure collection + all training data |
| `D_val` | 156 failures | patch acceptance + model selection only, never trained on |
| `D_heldout` | 158 failures | THE exam. Forward passes only |
| sibling-300 | 300 base successes | regression arena (`forget` = 1 − success rate ratio) |
| old-400 | 400 base successes | legacy regression continuity surface |

`pass@16` partition of heldout failures (base model, T=0.8, 16 samples):
**scaffold-only 47** (sampling never succeeds, scaffold repairs),
sampling-rescuable 10, neither 101.

## 2. The arms

| arm | what it is | status | numbers live at |
|---|---|---|---|
| **A1 base** | M0, no patches, no LoRA. The floor. | measured in Phase 0 | baseline records in `EDG-EXP1/results/a2_failures.json` |
| **A2 scaffold-on** | M0 + H1 patches in context. The teacher and the retention denominator. | frozen | `repro_rep2/data/s07_teacher_reanchor_heldout.json`: **50/158 heldout repairs** |
| **A3 EDR-main** | LoRA trained on scaffold-taught, AST-verified repairs (+replay, KL anchor). The protagonist. | frozen (rep2, 5 seeds) | `repro_rep2/artifacts/main_rep2_seed*/`: heldout **50.4/158 mean**, retention ≈ 1.008, sibling forget **0.045** |
| **A4 STaR** | LoRA trained on the base model's own lucky samples (pass@16 hits), row-matched to A3. The self-sampling contrast (STaR/RFT-style). | frozen (Phase 0) | C2 headline: on heldout scaffold-only, A3 − A4 = **+0.153, 95% CI [+0.064, +0.251]** |
| **A5 unverified** | A3's pipeline with the AST filter OFF (teacher outputs admitted regardless of verdict), row-matched to 444. Tests whether the verifier is necessary. | **Phase 1 — you run this** | `phase1_outputs/a5/` |
| **A6 retrieval-patch** | Don't train; retrieve top-k patches per failure and inject. The "why not RAG the patches" practitioner baseline. BM25 + pinned dense embedder, k∈{1,3}, best cell enters the context-cost frontier. | **Phase 1 — you run this** | `phase1_outputs/a6/` |
| **A7 replay-ablation** | A3's recipe minus replay rows. Because the KL anchor fires on replay rows, A7 is by definition "minus replay minus KL" (v1.26). Tests what replay buys. | **Phase 1 — you run this** | `phase1_outputs/a7/` |
| **A8 data-scale** | A3's data at 25% / 50% (episode-level subsets, replay ratio 2:1 preserved). 100% = A3 itself (artifacts reused, not retrained). | **Phase 1 — you run this** | `phase1_outputs/a8/` |
| **A11 placebo-patch** | H1 patches token-scrambled (length + vocabulary kept, information destroyed), injected exactly like A2, forward only. Kills the "patches just perturb the prompt distribution" explanation. | **Phase 1 — you run this** | `phase1_outputs/a11/` |
| A9 multi-round | M2 (= M0+A1+A2), the second EDR iteration. | **Phase 2 — you run this** | `round2_outputs/` |
| A10 multi-model | 3B/14B/other-family scale curve. | Phase 3, out of scope here | — |

Roles in one line each: **A4** answers "is scaffold teaching different from
self-sampling luck"; **A5** answers "does the verifier matter"; **A6** answers
"why internalize instead of retrieving"; **A7** answers "does replay protect
old capability"; **A8** answers "how does the effect scale with data"; **A11**
answers "is the scaffold effect just prompt noise".

## 3. Preregistered comparison rules

- All ablation arms use the LOCKED rep2 recipe (rank 16, lr 5e-5, epochs 3,
  replay 2:1, KL λ2, guards on) — no per-arm tuning, ever. Ruling: v1.26.
- Trained arms run 3 seeds {20260704, 20260705, 20260706} (overlapping A3's
  block for paired power); every 3-seed number carries a `PROBE` tag and is
  upgraded to 5 seeds only if it becomes a headline claim.
- Differences are judged by episode-level paired bootstrap CI (10k resamples).
  Point estimates and "looks better" are not conclusions.
- Regression hard gate: `forget ≤ 0.02` on the sibling arena. Phase 0's A3
  did NOT pass it (0.045) — that honest negative is part of the frozen
  result; Phase 1 arms are reported against the same gate.
- Gate-1 outcome branches (all preregistered, all publishable): A5 worse than
  A3 → verifier necessity established; A5 not worse → honest finding with
  wording downgrade. A6 approaching A3 at low context → the frontier's real
  shape. Any A6 cell repairing MORE than A2's full injection → recorded as
  patch-interference evidence. A11 placebo ≈ 0 → perturbation explanation
  excluded; > 0 → mechanical deduction of the net teaching gain.

## 4. Reference numbers you can check against

| quantity | value | source |
|---|---|---|
| teacher heldout repair (A2) | 50/158 = 0.3165 | `repro_rep2/data/s07_teacher_reanchor_heldout.json` |
| A3 heldout repair (5-seed mean) | 50.4/158 = 0.319 | `repro_rep2/artifacts/*/heldout.json` |
| A3 sibling forget (5-seed mean) | 0.0453 | `repro_rep2/artifacts/*/sibling.json` |
| M1 designated seed | 20260704 (heldout 51/158, sibling 292/300) | `repro_rep2/MANIFEST.json` |
| C2 scaffold-only CI (A3−A4) | +0.153 [ +0.064, +0.251 ] | CHANGELOG v1.20 |
| H1 accepted patches | 39 | `evolution_main_merged.json`, split seed 20260630 |
| A3 training rows | 444 = 148 core + 296 replay | `repro_rep2/data/distill_main_replay2_capped.jsonl` |
