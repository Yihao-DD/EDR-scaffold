# Overnight BFCL Gate Setup Summary

## Scope

All entries below are PROBE signal records. This summary covers mechanical BFCL connection, local AST signal checks, natural failure collection, deterministic failure splitting, and gate scaffold files.

## A0 BFCL Load

PROBE signal (n=808).

| item | value |
|---|---:|
| connected categories | 4 |
| simple_python episodes | 400 |
| simple_java episodes | 100 |
| simple_javascript episodes | 50 |
| live_simple episodes | 258 |
| total connected episodes | 808 |

| assert | result |
|---|---|
| category data readable | T |
| query readable for connected episodes | T |
| ground-truth function readable for connected episodes | T |
| ground-truth arguments readable for connected episodes | T |
| offline AST known-correct case returns True | T |
| offline AST known-wrong case returns False | T |

Files: `results/a0_bfcl_load.json`, `logs/a0.md`.

## A1 Signal Check

PROBE signal (n=3232 synthetic prediction cases from 808 episodes).

| prediction case | n | router True | router False | validator True | validator False |
|---|---:|---:|---:|---:|---:|
| correct | 808 | 808 | 0 | 808 | 0 |
| function correct, arguments wrong | 808 | 808 | 0 | 0 | 808 |
| function wrong | 808 | 0 | 808 | 0 | 808 |
| function correct, arguments partial | 808 | 808 | 0 | 0 | 808 |

| assert | result |
|---|---|
| router signal contains both True and False | T |
| validator signal contains both True and False | T |
| correct case has router=True for all rows | T |
| function-wrong case has router=False for all rows | T |
| function-correct/arguments-wrong case has router=True and validator=False for all rows | T |

Files: `results/a1_signal_check.json`, `logs/a1.md`.

## A2 Natural Failures

PROBE signal (n=808).

| metric | value |
|---|---:|
| total episodes | 808 |
| natural failures | 278 |
| call accuracy | 0.6559405941 |
| failure rate | 0.3440594059 |
| router_fail | 2 |
| validator_fail | 276 |
| router_fail rate | 0.0024752475 |
| validator_fail rate | 0.3415841584 |

| category | total | failures | router_fail | validator_fail |
|---|---:|---:|---:|---:|
| live_simple | 258 | 108 | 1 | 107 |
| simple_java | 100 | 28 | 0 | 28 |
| simple_javascript | 50 | 18 | 1 | 17 |
| simple_python | 400 | 124 | 0 | 124 |

| statistic | min | mean | max |
|---|---:|---:|---:|
| query length, words | 1 | 19.8836633663 | 212 |
| ground-truth parameter count | 0 | 2.7970297030 | 10 |

| assert | result |
|---|---|
| `n_failures > 0` | T |
| `n_failures < n_total` | T |
| one-failure-type-missing marker needed | F |

Files: `results/a2_natural_failures.json`, `logs/a2.md`, `logs/a2_run.out`, `logs/a2_run.err`.

## B0 Failure Splits

PROBE signal (n=278 natural failures).

| split | n |
|---|---:|
| train_update | 139 |
| validation | 69 |
| held_out | 70 |

| assert | result |
|---|---|
| train/update and validation disjoint | T |
| train/update and held-out disjoint | T |
| validation and held-out disjoint | T |
| all failure ids accounted for | T |

Files: `results/b0_splits.json`, `logs/b0.md`.

## B1-B5 Scaffold

| step | artifact | assert result |
|---|---|---|
| B1 budget framework | `gate_scaffold/budget_meter.py`, `results/b1_budget_framework.json`, `logs/b1.md` | T |
| B2 eight arm skeletons | `gate_scaffold/arms_skeleton.py`, `results/b2_arms_skeleton.json`, `logs/b2.md` | T |
| B3 patch operator skeletons | `gate_scaffold/patch_operators_skeleton.py`, `results/b3_patch_operators_skeleton.json`, `logs/b3_patch_operators.md` | T |
| B4 metrics schema | `gate_scaffold/metrics_schema.py`, `results/b4_metrics_schema.json`, `logs/b4.md` | T |
| B5 anti-shortcut checklist | `results/b5_anti_shortcut_rules.json`, `logs/b5_anti_shortcut_rules.md` | T |

Locked `NotImplementedError` cores:

- patch generation
- utility / rhat estimation
- oracle allocator
- repair experiment arms

## Blocked And Decisions

| directory | contents |
|---|---|
| `BLOCKED/` | empty at final verification |
| `DECISION_NEEDED/` | `a0_bfcl_categories.md` |

`DECISION_NEEDED/a0_bfcl_categories.md` records category inclusion/exclusion evidence for BFCL categories that were excluded or deferred from the single-turn single-function scope.

## Verification

| location | command | result |
|---|---|---|
| local `E:\EDGscaffold\EDG-BFCL` | `PYTHONPATH=. pytest tests -q` | 18 tests, exit code 0 |
| server `/root/autodl-tmp/EDG-BFCL` | `PYTHONPATH=. /root/miniconda3/bin/pytest tests -q` | 18 tests, exit code 0 |
| local markdown scan | forbidden judgment-word scan over `logs/` and `DECISION_NEEDED/` | no matches |
| server markdown scan | forbidden judgment-word scan over `logs/` and `DECISION_NEEDED/` | no matches |

## Sync

Artifacts are present in both `E:\EDGscaffold\EDG-BFCL` and `/root/autodl-tmp/EDG-BFCL`.

以上全部为机械接入与脚手架产物;四块判断核心(patch 生成 / 收益估计 / oracle / 修复实验)均未实现(NotImplementedError);任何判断(信号是否够用 / 失败率是否合适 / 是否进入 gate)留给人类与审计,本 agent 未做任何此类判断。
