# 起步指令:转向 Structure-over-Information 实验(保留环境,换 patch 与目标)

## 0. 先停、先保留、再转(读这一段,不要全弃)

你正在跑"实验一:预算化 harness patch 调度 + 解耦质检"。**现在转方向,但不是推倒重来——大部分地基要保留。**

**保留(继续用,别删):**
- BFCL Multiple AST + Simple AST 环境接入
- AST 验证器:router_success / validator_success / call_success
- frozen Qwen2.5-7B-Instruct 推理管线
- 自然失败采集(Multiple 上 router_fail / validator_fail 分布)
- 轻量 HarnessFix 式诊断器骨架(诊断失败类型:router 问题 vs validator 问题)

**停掉/搁置(本方向不需要):**
- "预算化调度"主线(successive halving / 先验证哪个 patch 那套)→ 搁置,不再是核心
- 解耦质检从"主实验同时跑的核心臂" → 降级为一次性 ablation

**新增/改变(本方向的核心):**
- patch 操作空间换成两个对照族:**结构化 patch (STRUCT) vs 等信息量自然语言 patch (NL)**
- 主实验目标从"调度更省" → **"结构化修复在 7B 上 held-out 赢过等信息量自然语言修复"**

把当前实验一的产出归档到 `EDG-EXP1/`(别删),新方向开 `EDG-EXP2-struct/`,复用 EXP1 的环境/采集代码(import 或软链,不重写)。

---

## 1. 这个新方向要证明什么(一句话)

现有 harness 自进化的修复是"注入信息"(更多描述/示例/规则);我们主张小模型该"注入结构"(把失败的决策拆成固定槽位模板)。**核心实验:同一个失败,生成携带相同信息但结构化程度不同的两种 patch,证明结构化的那种在 frozen 7B 上 held-out 涨得更多。**

---

## 2. 核心:两个对照 patch 族(信息相同,只差结构)

对每个自然失败,诊断出失败类型后,生成**两个版本的 patch,信息量配平,唯一差别是结构化程度**:

**STRUCT(结构化槽位模板):**
- router 失败 →
  ```
  [STEP 1] User wants to: ___
  [STEP 2] Candidate functions:
           - fn_A does: ___
           - fn_B does: ___
  [STEP 3] Best match: ___
  ```
- validator 失败 →
  ```
  [PARAM 1] name: ___  type: ___  value from query: ___
  [PARAM 2] name: ___  type: ___  value from query: ___
  ```

**NL(自然语言散文,同信息):**
- router 失败 → "When the user wants X, fn_A does ... and fn_B does ...; pick the one matching the intent."
- validator 失败 → "Extract each parameter with correct type; date should be ISO, count should be integer."

**铁律(对照有效性的前提):STRUCT 和 NL 必须携带相同信息**(同样的函数说明 / 参数规则),**唯一差别是 STRUCT 拆成固定槽位、NL 是连续散文**。每条 patch 报告 token 数,确认两族 token 数可比(配平)。**如果 NL 信息更多或更少,对照无效——这是头号要守的。**

---

## 3. 自进化 loop（两条轨道并行）

沿用 HarnessFix 式 loop,但跑两条独立轨道:
1. frozen 7B 在 BFCL Multiple 自然失败(已有采集)。
2. 诊断失败类型(router / validator)。
3. 生成 STRUCT patch 和 NL patch。
4. 本地 AST 验证:patch 后 validation 集重跑,涨且不回归则接受。
5. 迭代。
6. held-out 测。

**轨道 A:只用 STRUCT patch 自进化。轨道 B:只用 NL patch 自进化。比较两条轨道 held-out 终点。**

---

## 4. 对照组 + 指标

**对照组:** no-evolution / NL-evolution(轨道B,代表"注入信息") / STRUCT-evolution(轨道A,本工作) / oracle(穷举实测上界)。

**指标:**
- held-out pass rate（主，多 seed + CI）
- **核心:STRUCT-evo vs NL-evo 的 held-out 差距**（"结构 > 信息"的直接证据）
- **context 成本:每条 patch 加的 token 数；STRUCT vs NL 的 context 占用；context 增量对 7B 性能净影响**（自己测，不引外部数字）
- 按失败类型拆分：router 上 / validator 上，STRUCT vs NL 各自的差距

---

## 5. 一次性 ablation（不是主线，做一次即可）

- **信息量配平 sanity**：确认 STRUCT 和 NL 的 token 数 / 内容覆盖可比（否则 STRUCT 赢可能只因信息多）。**这个最重要,先做。**
- **解耦 sanity（一次）**：把 patch 里函数名/参数名解耦成无关符号,确认 STRUCT-evo 涨点不来自字面 token 重合。干净就过,写一句。

---

## 6. 阶段性 gate（跑前定）

**主判据：** STRUCT-evo 的 held-out 是否显著 > NL-evo？
- 显著 > → "结构 > 信息" 在 7B 成立,proof-of-concept 成,贴结果。
- ≈ → 先查信息量是否配平、Multiple 失败结构是否够丰富,贴数讨论。
- < → 7B 上结构化反而有害（重要负结果,但先排查 patch 设计）。

**配平前置 gate（最先做）：** 先确认能生成"信息相同、只差结构"的 STRUCT/NL 对,且 token 数可比。如果配平做不到（比如 STRUCT 模板天然比 NL 长很多）,先停下贴出来——因为配不平,整个对照无效。

---

## 7. 纪律
- frozen 7B 权重不动；patch 可版本化/可回滚；本地 AST 验证,绝不 GPT judge。
- STRUCT/NL 信息量必须配平（核心）。
- 模型抽象成可替换接口（为后续多模型铺路）。
- 记录 base 能力 / STRUCT 收益 / NL 收益 / context 成本（后续画相变曲线要用）。
- 所有数字标 PROBE；n<30 或单 seed 一律 PROBE。
- 复用 EXP1 环境/采集,不重写；旧产出归档不删。

---

## 8. 执行顺序（先做配平,这是命门）

1. **先做配平验证**：用几个失败样本,手工 + 自动生成 STRUCT 和 NL 两版 patch,确认能做到"信息相同、token 数可比"。**贴出 3-5 对样本 + token 数对比。** 配不平就停下讨论。
2. 配平成立 → 实现 patch 生成器（两族）。
3. 跑两条自进化轨道（STRUCT-evo / NL-evo）+ no-evo + oracle，多 seed。
4. 出 held-out 对比 + context 成本 + 失败类型拆分。
5. 一次性 ablation（配平 sanity + 解耦 sanity）。
6. 按 gate 判读,贴结果。

**先做第 1 步配平验证,贴 3-5 对 STRUCT/NL 样本和它们的 token 数——这一步决定整个对照成不成立。配得平再往下,配不平先停下贴给我。**
