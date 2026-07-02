# EDR 方法论总述

版本:2026-07-02
依据:`PROJECT_MASTER_PLAN.md` v1.2 + `CHANGELOG.md` 已登记的执行期修订。
用途:给项目内部、审计者、合作者快速理解我们到底在做什么、为什么这样做、哪些数字能 claim、哪些只能当 `PROBE`。

本文不是结果汇总表。结果数字、逐轮判决和时间顺序以 `CHANGELOG.md`、`EDG-EXP3-distill/logs/`、`EDG-EXP3-distill/results/` 为准。本文描述的是方法论和实验纪律。

---

## 1. 项目一句话

**EDR = Evolve -> Distill -> Retire。**

我们先用外部 scaffold 修复小模型在 function calling 上的失败,再把这些被确定性 verifier 认证过的修复轨迹蒸馏进模型权重,最后把 scaffold 完全卸载,只用模型本身和 LoRA adapter 做评估。

核心问题不是“scaffold 在场时有没有帮助”。它当然有帮助。真正的问题是:

> scaffold 发现并验证过的修复,能不能变成模型权重里的能力,并且不破坏模型原本会做的题?

这就是 “scaffold is meant to come down” 的含义。Scaffold 是临时教师,不是永久假肢。

---

## 2. 方法论对象

### 2.1 Base Model

`M_0` 是冻结的小模型 base。当前 Phase 0 使用的是 Qwen2.5-7B-Instruct 这一量级的模型。base 权重本身不直接改写。

### 2.2 Scaffold / Harness

`H_r` 是第 `r` 轮 evolve 得到的 harness,包含 accepted patches。它在 teacher 生成阶段可以帮助模型修复失败,但在 distilled model 的主评估中必须完全卸载。

### 2.3 LoRA Adapter

`A_r` 是第 `r` 轮 distill 训练出的 LoRA adapter。评估模型写作:

```text
M_r = M_0 + A_1 + ... + A_r
```

adapter 独立存盘、可加载、可卸载,不做不可逆 merge。这延续了项目一贯的 rollback 纪律。

### 2.4 Verifier

全项目唯一裁判是 BFCL AST verifier。它不是单纯格式检查,而是与 ground truth 的任务级匹配:函数选择正确、必需参数齐全、参数值落在可接受答案集内。

不使用 GPT judge、LLM judge、人类偏好 judge 来做训练过滤、在线验证、分区或 headline 评估。

---

## 3. 三个科学主张

### C1: 存在性

frozen 小模型经 scaffold 修复出来的能力,能否蒸馏进权重,并在 scaffold 完全卸载后保留。

判定口径:

```text
retention = repair(M_r 无 patch, heldout failures) / repair(M_{r-1}+H_r, heldout failures)
```

同时必须满足遗忘硬闸。repair 高但 forget 破闸,不能算部署级成功。

### C2: 机制差异

scaffold 教会的数据,和 base 自己采样碰对的数据,是否在类别上不同。

核心分区来自 base pass@16:

- **sampling-rescuable**:base 无 scaffold,温度采样 16 次,至少一次过 verifier。
- **scaffold-only**:base pass@16 全灭,但 scaffold 能修复。
- **neither**:base pass@16 和 scaffold 都没有修复。

C2 的主裁决场是:

```text
heldout scaffold-only 分区上的 A3(EDR-main) vs A4(STaR)
```

如果 EDR-main 在这个区显著优于 STaR,说明 scaffold-taught 数据确实转移了 self-sampling 得不到的能力。

### C3: 动力学与规模

多轮 Evolve -> Distill -> Retire 后,能力是复利、收敛还是崩塌? 这种行为如何随模型规模变化?

C3 不预设正结果。复利、健康收敛、材料枯竭、崩塌,只要测量干净,都是结果。

---

## 4. 方法论贡献

这个项目不只是一条 agent pipeline,还显式交付一套 scaffold-to-weight internalization 的评估范式:

- Wipe Test: scaffold 完全卸载后的能力保留。
- pass@16 scaffold-only 分区:区分 self-sampling 够得到和够不到的失败。
- STaR/RFT 对照:同样训练预算下,比较 scaffold-taught 与 self-sampled。
- 回归硬闸:修复失败不能以打碎原成功为代价。
- 2x2 泛化分层:函数是否见过、错误类型是否见过。
- teacher_agree:distilled 输出是否真的复现 teacher 行为。
- placebo-patch:排除“只是 prompt 扰动”这类替代解释。
- 预注册 Gates 和失败分支:结果不好也必须交付。

---

## 5. 数据治理

### 5.1 数据划分

| 划分 | 用途 | 训练权限 |
| --- | --- | --- |
| `D_train` | 失败采集、evolve、蒸馏数据来源 | 可以进训练 |
| `D_val` | patch 接受、LoRA 选点、clean validation | 不许进训练 |
| `D_heldout` | 最终主考场 | 绝不进训练,选点前不碰 |
| `R_success_eval` | base 成功回归评估 | 不许进训练 |
| replay pool | 防遗忘 replay | 必须与评估份分离 |
| `R_other` | 跨类别回归 | 只评估 |

### 5.2 三条硬 assert

v1.2 后,训练集 episode 必须同时满足:

```text
training intersect D_val = empty
training intersect D_heldout = empty
training intersect R_success_eval = empty
```

这三条必须代码 assert 化并进 pytest。不能靠口头纪律。

### 5.3 D_val 泄漏事故后的治理修订

旧总纲存在自相矛盾:

- Step 0.0 写过 `D_train ∪ D_val` 失败集重跑修复清单。
- 治理表又写 `D_val` 不进训练样本。
- assert 清单当时只覆盖 heldout 和 R_success_eval,漏了 `training ∩ D_val`。

因此第一版污染网格里 main/star 分别吃进了 D_val episode。处置是:

- 污染 checkpoint 和日志归档到 `contaminated_gridv1/`。
- 全部标 `PROBE (contaminated)`。
- 不作为选点、Gate、论文数字。
- 只允许用于干净 `R_success_eval` 上的遗忘类型诊断。
- clean pipeline 改为 train-only 蒸馏池。

### 5.4 train-only 规则

当前 Phase 0 的蒸馏数据池只来自 train 份修复:

```text
train repaired episodes = 74
```

val 份修复只记录、只用于评估/选点,绝不进任何训练集。

---

## 6. Phase 0 数据构建

### 6.1 Step 0.0: 修复清单

对 `D_train ∪ D_val` 的失败可重跑 teacher/harness,但必须 train/val 分开记账。

进入蒸馏池的只有:

```text
D_train repaired episodes
```

进入 clean Phase 0 的 train 份修复数为:

```text
74
```

这满足 v1.2 的直行条件:train 份原始修复不少于 50。

### 6.2 Step 0.1: pass@16 分区

对失败 episode 用 base 无 scaffold 采样:

```text
temperature = 0.8
samples = 16
judge = BFCL AST verifier
```

输出三区:

- sampling-rescuable。
- scaffold-only。
- neither。

train 修复集上的关键 reach 数字:

```text
64 / 74 = 86.49% 是 scaffold-only
10 / 74 = 13.51% 是 sampling-rescuable 且被 scaffold 修复
```

这说明大部分 scaffold 教学材料,base 自采样本身产不出来。

### 6.3 Step 0.2: EDR-main 数据

对 train repaired episodes:

1. 输入是原始 no-patch prompt。
2. teacher 是 `M_0 + H_1`。
3. 目标输出必须 AST 过 verifier。
4. T=0 生成一条。
5. T=0.8 增广采样,最多到 x8。
6. 对输出规范化并去重。

实际漏斗:

```text
T=0.8 x8: 592 sampled
AST-valid: 553
per-episode dedup: 74 unique outputs
```

结论:teacher 输出在当前任务上近似点质量分布。继续加采不能制造不存在的输出多样性。

EDR-main core 当前结构:

```text
148 rows = 74 unique outputs x 2 repeat weight
```

### 6.4 Step 0.2: STaR 数据

STaR 只使用 train 份 base self-sampling 成功轨迹。它回答:

> 如果不用 scaffold,只训练 base 自己采样碰巧做对的轨迹,能学到多少?

当前 row-matched 后:

```text
148 rows
31 unique train episodes
```

这不是 unfair。row 数配平是训练预算公平;unique episode 少是 self-sampling 的内生限制,正是 C2 要测的现象。

### 6.5 配平方向

v1.2 写死:

```text
配平方向永远向下
```

哪边少,另一边下采样到它。不得为了某一臂扩大数据量。

---

## 7. Replay 与遗忘控制

### 7.1 为什么需要 replay

distill 修复失败时,模型可能学到“参数值要改”的泛化规则,并把这个规则错误作用到原本做对的 episode 上。这就是 forgetting/interference。

因此训练集中混入 base 原成功轨迹作为 replay,用于钉住原能力。

### 7.2 当前 replay 数据变体

| 变体 | 总 rows | core rows | replay rows | 说明 |
| --- | ---: | ---: | ---: | --- |
| clean grid default | 222 | 148 | 74 | replay 50% |
| replay1 | 296 | 148 | 148 | 1:1 |
| replay2 | 444 | 148 | 296 | 2:1 |
| capped replay2 | 444 | 148 | 296 | replay 池避开 capped arena |
| capped targeted replay2 | 444 | 148 | 296 | function-targeted replay |

当 unique replay 不足以填满 row 数时,允许 repeat weighting,但必须报告 unique 数和重复结构。

---

## 8. 实验臂

### A1: Base

`M_0`,无 scaffold,无 LoRA。地板。

### A2: Scaffold-On

`M_0 + H_1`,scaffold 在场。teacher 和 retention 分母。

### A3: EDR-main Distilled

LoRA 训练于 scaffold-taught、AST-verified train-only repairs + replay。主角。

### A4: STaR Distilled

LoRA 训练于 base self-sampled、AST-passing train-only trajectories + replay。C2 核心对照。

### A5: Unverified Distilled

未来 Phase 1 臂。关闭 AST 过滤,检验 verifier 必要性。

### A6: Retrieval-Patch

未来 Phase 1 臂。检索少量 patch 注入,形成 context cost vs performance 前沿。

### A11: Placebo-Patch

未来 Phase 1 臂。打乱 patch token,保留长度/词表扰动,摧毁信息,纯前向评估。用于排除“scaffold 只是 prompt 分布扰动”的解释。

---

## 9. 训练网格

### 9.1 v1.2 clean grid

污染事故后,网格保守化为:

```text
rank in {8, 16}
lr in {2e-5, 5e-5, 1e-4}
epochs in {1, 2}
replay = 50%
```

每臂 12 点,main 和 STaR 对称。单 seed grid,选定后再 5 seeds。

### 9.2 选点规则

选点只看 clean `D_val`,且训练与 `D_val` 零交集。

预注册规则:

```text
在满足 r_success >= 0.98 的点里,取 val repair 最高者。
若全不满足,取回归最好档 + BLOCKED 报告。
```

### 9.3 升级规则

如果 12 点全破 forget 闸:

```text
取 repair 最高点
replay 提到 1:1
lr 减半
epochs 保持
重跑一点
```

仍破闸则进入 BLOCKED。

### 9.4 BLOCKED 第一线

s04 升级点仍破闸后,进入 BLOCKED。

第一线配方扩展是 KL-anchor:

```text
lambda in {0.5, 1, 2}
replay in {1:1, 2:1}
```

KL-anchor 在 replay/base-success 输入上惩罚模型偏离 base 分布。它针对的病因假设是:修复训练侵蚀了原成功 episode 附近的 margin。

### 9.5 ep3 机制探针

额外跑 ep3 replay1,测试“更长、更温和、更多 replay”是否会让原成功区固结回来。

这不是无限制调参,而是 BLOCKED 菜单里已登记的机制检查。

### 9.6 capped + targeted 最后一发

因为旧 `R_success_eval` 已被多次诊断读取,再用它设计 targeted replay 会有 adaptive contamination。v1.13 后改用 capped-pool 方案:

```text
226 unique replay-success rows
-> 120 allowed replay rows
-> 106 capped-arena rows
```

capped D3 先跑四角格校准。D1 targeted replay 只允许每臂一发:

1. 取已完成非 targeted 配置中 capped forget 最低者。
2. 只把 uniform replay 换成 targeted replay。
3. 其他 rank/lr/epoch/lambda/replay 体量逐字节保持。
4. 成功/partial/failure 按预注册判据机械对号。

targeted replay 的选择规则:

```text
train-failure exact function match 优先
coarse family 补足
剩余再 uniform
```

不得从 heldout 或最终考场反推训练。

---

## 10. 评估指标

### 10.1 repair

```text
repair(M, S) = M 在原失败集合 S 上 T=0 前向且 AST 通过的比例
```

### 10.2 retention

```text
retention = repair(distilled no-patch, heldout failures)
          / repair(scaffold-on teacher, heldout failures)
```

当前 harness 的 heldout teacher 分母已实测:

```text
heldout failures = 158
teacher repairs = 50
teacher repair rate = 0.3165
```

后续 D2 retention 以这个 harness-specific 分母为准。

### 10.3 forget

```text
forget = 1 - success(M, regression_arena) / success(M0, regression_arena)
```

在 old arena 上就是:

```text
new_wrong / 400
```

硬闸:

```text
forget <= 0.02
```

### 10.4 slice63

`slice63` 是 clean validation 中 teacher 修复过的 63 个失败 episode。它是 val 侧 retention 预演,不是 heldout claim。

### 10.5 validation 四分层

clean val 的 156 个失败分成:

| 分层 | n | 含义 |
| --- | ---: | --- |
| slice63 scaffold-only | 55 | teacher 修复,base pass@16 全灭 |
| slice63 sampling-rescuable | 8 | teacher 修复,base pass@16 可救 |
| slice93 neither | 84 | teacher 未修,base pass@16 也不可救 |
| slice93 sampling-rescuable | 9 | teacher 未修,base pass@16 可救 |

这四格是 reporting-only watch,不改变选点规则。

### 10.6 teacher_agree

比较 distilled 输出与 scaffold-on teacher 输出的 normalized function call 是否一致。

用途:

- 一致高:说明模型内化的是 teacher 行为。
- 一致低但 AST 成功:说明模型可能学到了不同成功路径。

### 10.7 overall

overall 是成功保持 + 失败修复后的整体 BFCL accuracy。它可以说明净能力变化,但不能豁免 forget 硬闸。

### 10.8 forget_heldout

D2 必须显式报告 heldout base-success 上的回归:

```text
forget_heldout
```

这保证最终回归 claim 有处女地指标,不完全依赖探索期被诊断读取过的 arena。

---

## 11. Gates 与判决纪律

### 11.1 Gate 0

主判据:

```text
retention >= 0.60: 强信号
0.40 <= retention < 0.60: 可用
retention < 0.40: 一轮重试后仍低则停线/人决策
```

但任何 retention 都必须受回归硬闸约束:

```text
forget <= 0.02
R_other 无显著下降
```

C2 判据:

```text
heldout scaffold-only 上 A3 > A4,5-seed paired CI 分离
```

### 11.2 BLOCKED 判据

KL/BLOCKED 阶段:

```text
success = forget <= 0.02 and slice63 >= 0.40
partial = forget <= 0.02 and slice63 < 0.40
failure = 无合格点
```

若失败,只能按已登记的最后一发 targeted/capped 流程走。不得现场发明新杠杆。

### 11.3 targeted 最后一发判据

```text
success = active arena forget <= 0.02 and slice63 >= 0.40
partial = active arena forget <= 0.02 and slice63 < 0.40
failure = active arena forget > 0.02
```

若 failure:

```text
配方探索硬停
```

论文转入 frontier/负结果叙事:本配方族能创造修复能力,但未能在当前限制下把遗忘压到部署级闸内。

---

## 12. 当前机制假设

这些是 `PROBE` 机制框架,不是最终 claim。

### 12.1 点质量输出

teacher 采样大量 AST-valid 输出后,按 episode 去重仍塌缩成每 episode 一个 normalized call。这说明任务正确答案通常是窄点,不是宽答案集。

### 12.2 reach vs transfer

C2 分两层:

1. **Reach**:base self-sampling 是否能产生训练信号。
2. **Transfer**:训练后 EDR-main 是否比 STaR 更能转移到 scaffold-only 区。

当前 reach 层很强:train 修复里 86.49% 是 scaffold-only;heldout teacher 修复里 scaffold-only 占比更高。transfer 层仍以 heldout 5-seed 为最终裁决。

### 12.3 水位模型

clean grid 监控显示:更高训练剂量会把更多低先验失败推过正确边界,也会把更多低 margin 原成功推向错误邻居。

因此 repair 和 forget 是同一场分布移动的两岸。项目工程问题变成:

```text
如何保留 transfer,同时压住原成功区的边界侵蚀?
```

### 12.4 generic interference

main 和 STaR 的教学内容不同,但 newly-wrong 集合和漂移子类型高度重合。工作解释是:脆弱 episode 的错误方向预先存在于 base 概率地形里,微调只是把它推过去。

### 12.5 KL-anchor 的动机

如果病因是 base-success 区的 margin erosion,则 replay 只是在重复正确答案,而 KL-anchor 直接约束输出分布不要离 base 太远。它是目前最贴病因的配方扩展。

### 12.6 targeted replay 的动机与 caveat

targeted replay 试图优先保护训练侧 failure function/family 相关的成功 episode。

caveat:我们已经看过 old regression arena 的诊断,所以 targeted 不能继续用 old arena 当纯净裁判。必须使用 fresh/capped/heldout 层级重新约束。

---

## 13. 最终 D2 评估

选定代表点后,进入 D2 heldout 5-seed。

代表点规则:

- `rep1`:全库 slice63/transfer 最高代表点。
- `rep2`:forget 最低代表点;若 targeted 成功,则 targeted 是 rep2。

D2 必须报告:

1. heldout 158 failures 全集 repair。
2. heldout scaffold-only / sampling-rescuable / neither 分层。
3. retention ratio:distilled repair / 0.3165。
4. conditional recovery:teacher 修复的 50 个中 distilled 命中多少。
5. A3 vs A4 在 heldout scaffold-only 层的 paired bootstrap CI。
6. forget_heldout。
7. old regression continuity 指标。
8. R_other / overall / teacher_agree。

---

## 14. 反自欺纪律

项目历史里最贵的教训是:看起来像进展的信号,很多不是进展。

因此:

1. 裁判唯一:BFCL AST verifier。
2. 预注册不可变:Gate、臂、指标不能事后按结果改。
3. PROBE 必须标注:n 小、seed 少、污染、探索性都不能当主 claim。
4. CI 至上:点估计方向不是结论。
5. 配平强制:数据量、seeds、调参机会都要报。
6. 泄漏 assert 化:不靠记忆。
7. 判断密集步骤人在场:agent 准备材料,人做最终解释。
8. 不利结果同权交付:每阶段报告必须包含最不利数字。
9. 可满足性检查:任何新预注册数据条件必须先做池算术/可构造性检查。

---

## 15. 文件地图

### 权威规范

- `PROJECT_MASTER_PLAN.md`:当前总纲,v1.2。
- `CHANGELOG.md`:执行期修订与逐轮判决。
- `METHODOLOGY.md`:本文,方法论整理。

### 活动工作区

- `EDG-EXP3-distill/`:Phase 0 / EXP3 主工作区。

### 关键脚本

- `EDG-EXP3-distill/scripts/phase0_probe.py`:数据盘点、pass@16、构建与审计。
- `EDG-EXP3-distill/scripts/lora_phase0.py`:LoRA 训练、评估、KL-anchor、replay 变体。
- `EDG-EXP3-distill/scripts/heldout_pass16.py`:heldout pass@16 分区。
- `EDG-EXP3-distill/scripts/logprob_probe.py`:base log-prob 机制探针。
- `EDG-EXP3-distill/scripts/logprob_followups.py`:log-prob 后续分析。
- `EDG-EXP3-distill/scripts/v13_arena_targeted.py`:v1.13 arena 与 targeted replay 构建。
- `EDG-EXP3-distill/scripts/v13_fresh_arena_eval.py`:fresh/capped arena 评估工具。
- `EDG-EXP3-distill/scripts/eval_capped_arena.py`:capped arena 纯前向评估。

### 结果与日志

- `EDG-EXP3-distill/logs/deferred_metrics.md`:探索性指标登记。
- `EDG-EXP3-distill/logs/s03_monitoring.md`:clean grid 监控。
- `EDG-EXP3-distill/logs/s04_upgrade_monitoring.md`:升级点监控。
- `EDG-EXP3-distill/logs/s05_blocked_monitoring.md`:BLOCKED/KL 监控。
- `EDG-EXP3-distill/results/`:JSON 结果与审计产物。
- `EDG-EXP3-distill/contaminated_gridv1/`:污染网格归档,只作 `PROBE (contaminated)` 诊断。

---

## 16. 对外报告口径

可以说:

- 我们研究的是 scaffold-to-weight internalization,不是单纯 scaffold-on 性能。
- 训练数据必须由确定性 verifier 认证。
- scaffold-only 区是 C2 的核心战场。
- STaR 是 self-sampled 对照,不是要在所有意义上打败的论文或数据集。
- repair 和 forget 必须一起报。
- 若忘得太多,即使 overall 净值为正,也不能算通过部署级闸。
- 若最终过不了 forget 闸,frontier 和负结果仍然是有效科学产出。

不能说:

- 单 seed validation 结果证明 C2。
- 污染网格方向可以引用。
- val 选点结果等于 heldout 结论。
- STaR 在 scaffold-only 也能转移就代表 C2 失败;最终要看 heldout 5-seed scaffold-only paired CI。
- 只报 repair 不报 forget。
- 用 old regression arena 的 targeted 后结果当完全干净 gate。

---

## 17. 审计前检查清单

任何数字外发前,逐条检查:

- 是否来自 clean run?
- 是否含 `D_val` 训练泄漏?
- 是否 heldout 零接触直到选点锁定?
- 是否标注了 val / heldout / old regression / capped arena / PROBE?
- A3/A4 是否 row-matched?
- unique episode 数是否同时报告?
- scaffold 是否在 distilled 评估时完全卸载?
- 是否只用 BFCL AST verifier?
- 是否报告 forget?
- 是否满足或明确破坏 `forget <= 0.02`?
- 若是探索性分析,是否已先在 `deferred_metrics.md` 登记?
- 若是新数据条件,是否做了池算术/可构造性检查?

这个清单本身就是方法论的一部分。EDR 的目标不是让数字好看,而是让每个结论站得住。
