# 实验计划:Structure-over-Information Scaffold Evolution for Small Models

> 项目:frozen 小模型通过外部脚手架自进化变强,核心创新在"修复操作注入结构而非信息"。
> 两阶段:阶段一(7B 单模型,自有算力)证明 work;阶段二(多同规模模型,公司算力)画结构相变曲线冲顶会。

---

## 0. 一句话主张(论文核心)

现有 harness 自进化(HarnessFix/Self-Harness)的修复操作本质是**注入信息**(更多 tool 描述、示例、规则)——这建立在"模型内部能组织好信息"的前提上,对强模型成立。但**小模型缺的不是信息而是组织能力**,且更多信息撑大 context、小模型对 context 增长敏感。因此我们提出针对小模型的脚手架自进化:修复操作**注入结构**(把模型反复失败的决策拆成它能稳定填充的固定槽位模板)而非注入信息;并发现一个**脚手架结构相变**——随模型规模下降,最优脚手架从"自然语言、让模型自组织"转向"显式结构化、固定槽位"。

---

## 1. 文献地基(已核实可信度,2026-06-30)

**结实(可当地基):**
- **Guided-Structured Templates(EMNLP 2025 Main,正刊,DOI 确认)**:证明 free-form CoT 对 function-calling 适得其反、结构化推理模板跨模型 +3-12%。**这是"结构 > 信息"主张的核心支柱,最硬。** 那个"小模型加 CoT 降分"(Qwen-2.5-14B 77.33→62.25)也出自此。

**可用但 preprint(措辞留余地,不说"已发表顶会"):**
- **HarnessFix(arXiv 2606.06324,中科院软件所)**:trace→诊断到 harness layer→scoped repair→验证,4 benchmark +15.2-50%。**作为改进底座和 baseline。** 全文已读,方法实在。
- **Self-Harness(arXiv 2606.09498,上海AI Lab)**:weakness mining→proposal→validation。related work。

**有裂缝(降权 + 加限定,见下):**
- **NLT(arXiv 2510.14453,PokketCoach 公司出品,评测域 customer service/mental health,frontier 模型)**:表面主张"自然语言 > 结构化",**和本工作表面矛盾**。处理:NLT 说的是**输出格式轴**(JSON vs 自然语言句子),本工作说的是**推理引导轴**(决策拆槽位 vs 给描述自己想),两轴正交。只用 NLT 支撑"小模型对格式约束/context 长度敏感"这个窄点,**不用它支撑"结构化有益"**,并在论文里明确区分两轴。

**不可当 citation(仅做叙事 motivation):**
- "脚手架溶解"(个人博客 leehanchung)、"context 增长导致小模型暴跌的具体 token 数字"(二手转引未核实 Modarressi 原文)。**这些要么自己核实原文,要么在本实验里直接测量自证——不依赖二手数字。**

---

## 2. 核心区分(必须在论文和实验里讲清,否则被 NLT 打)

| 轴 | NLT 说的 | 本工作说的 | 关系 |
|---|---|---|---|
| **输出格式轴** | 自然语言句子 > JSON(别被语法束缚) | 不主张这个 | 正交,可并存 |
| **推理引导轴** | 不涉及 | 结构化槽位模板 > 一段自然语言描述(把决策拆成可稳定填的步骤) | **本工作的核心** |

本工作的"结构化"= 推理引导层面把决策拆成固定槽位;不是输出格式层面的 JSON。

---

## 3. 两阶段总览

### 阶段一(自有算力,7B 单模型):证明 work
**目标**:产出一个"硬的"proof-of-concept——证明结构化脚手架修复在 7B 上让 held-out 涨点,且**赢过等信息量的自然语言修复**。这个 proof-of-concept 用来换公司算力做阶段二。

**硬标准(可信到能换算力)**:
1. 涨点真:7B 自进化后 held-out 涨,多 seed + 置信区间。
2. 结构 > 信息:同信息量下,结构化 patch 赢自然语言 patch(核心对照)。
3. context 成本量化:测结构化 patch 加了多少 context、净影响多少(自证,替代二手数字)。
4. 非 artifact:解耦 ablation 做一次,确认涨点不来自字面 token 重合。

### 阶段二(公司算力,多同规模模型):画结构相变
**目标**:size sweep(Qwen2.5-7B/3B/1.5B、Llama-3.1-8B、Mistral-7B 等),证明随模型变小,结构化优势越大、自然语言优势越小,存在可测的**结构相变点**。这条曲线是顶会 novelty 核心。

---

## 4. 阶段一实验设计(本轮要做的)

### 4.1 复用本地 agent 已有产出(不推倒重来)
从"实验一"保留:
- BFCL Multiple/Simple 环境接入 + AST 验证器(router_success / validator_success / call_success)。
- frozen Qwen2.5-7B-Instruct 推理管线。
- 自然失败采集(Multiple 上 router_fail/validator_fail 分布)。
- 轻量 HarnessFix 式诊断器骨架(诊断失败类型:router 问题 vs validator 问题)。

搁置/降级:
- "预算化调度"主线 → 搁置(本工作核心不是调度)。
- 解耦质检从"主实验核心臂" → 降级为一次性 ablation。

### 4.2 核心改动:patch 操作空间换成"结构 vs 信息"两个对照族

**这是整个实验的核心。** 对同一个失败,生成两种 patch,**携带相同的信息,只差结构化程度**:

**结构化 patch(STRUCT):把决策拆成固定槽位模板。**
- router 失败 → 注入函数选择槽位模板:
  ```
  [STEP 1] User wants to: ___
  [STEP 2] Candidate functions and what each does:
           - fn_A: ___
           - fn_B: ___
  [STEP 3] Best match: ___
  ```
- validator 失败 → 注入参数填充槽位模板:
  ```
  [PARAM 1] name: ___  type: ___  value from query: ___
  [PARAM 2] name: ___  type: ___  value from query: ___
  ```

**自然语言 patch(NL):同样的信息,写成一段描述,不拆槽位。**
- router 失败 → "When the user wants X, consider that fn_A does ... and fn_B does ..., and pick the one that best matches the user's intent."
- validator 失败 → "Make sure to extract each parameter with the correct type; for example the date parameter should be in ISO format and the count parameter should be an integer."

**关键约束(否则对照无效):STRUCT 和 NL 必须携带相同信息量**(同样的函数说明、同样的参数规则),**唯一差别是 STRUCT 拆成固定槽位、NL 是连续散文**。信息量用 token 数 / 内容覆盖度配平,在日志里报告两者的 token 数确认可比。

### 4.3 自进化 loop(沿用 HarnessFix 式,但 patch 是上面两族)
1. frozen 7B 在 BFCL Multiple 自然失败。
2. 诊断失败类型(router / validator)。
3. 对每个失败,生成 STRUCT patch 和 NL patch 两个版本。
4. 本地 AST 验证:patch 后在 validation 集重跑,涨且不回归则接受。
5. 迭代,接受的 patch 进 harness。
6. held-out 测。

**两条独立的自进化轨道并行**:一条只用 STRUCT patch、一条只用 NL patch,**比较两条轨道的 held-out 终点**。

### 4.4 对照组
- **no-evolution**(baseline harness,不进化)
- **NL-evolution**(只用自然语言 patch 自进化)—— 这是"注入信息"路线,代表 HarnessFix 式
- **STRUCT-evolution**(只用结构化槽位 patch 自进化)—— 本工作
- **oracle**(穷举所有 patch 实测取最高,上界)

### 4.5 指标
- **held-out pass rate**(主):no-evo / NL-evo / STRUCT-evo / oracle,多 seed + CI。
- **核心对照:STRUCT-evo vs NL-evo 的 held-out 差距**(这是"结构 > 信息"的直接证据)。
- **context 成本**:每条 patch 加的 token 数;STRUCT vs NL 的 context 占用对比;context 增量对 7B 性能的净影响(自证,替代二手数字)。
- **按失败类型拆分**:router 失败上 STRUCT vs NL、validator 失败上 STRUCT vs NL(可能两类失败的结构化收益不同)。

### 4.6 一次性 ablation(降级的解耦质检 + 必要 sanity)
- **解耦 sanity(一次,非主线)**:把 patch 里的函数名/参数名解耦成无关符号,确认 STRUCT-evo 的涨点不是来自 patch 文本和 benchmark 的字面重合。做一次,干净就过,写进 ablation 一句话。
- **信息量配平 sanity**:确认 STRUCT 和 NL 的 token 数 / 内容覆盖可比(否则赢可能只因信息多)。

---

## 5. 阶段一判读

| 结果 | 含义 | 下一步 |
|---|---|---|
| STRUCT-evo 显著 > NL-evo(held-out),且 context 成本可接受 | **"结构 > 信息"成立** | proof-of-concept 成,写技术报告换算力,进阶段二 |
| STRUCT-evo ≈ NL-evo | 结构化没有独立优势 | 检查:是否信息量没配平?是否 7B 在 Multiple 上失败结构不够丰富?调整或转 |
| STRUCT-evo < NL-evo | 7B 上结构化反而有害 | 重要负结果,但需排查 patch 设计;可能相变点在更小模型,7B 还在自然语言占优区 |

注:即使 7B 上 STRUCT ≈ NL,也不代表方向死——相变点可能在更小模型(3B/1.5B)。但阶段一要先在 7B 上看到信号或明确的趋势,才值得上阶段二。

---

## 6. 阶段二(公司算力,简述,后续细化)
- size sweep:7B/3B/1.5B/8B/Mistral-7B,每个模型跑 STRUCT-evo vs NL-evo。
- 画相变曲线:x 轴模型能力(或参数量),y 轴 STRUCT-evo 相对 NL-evo 的优势。
- 主张:优势随模型变小而扩大,存在相变点。
- 这条曲线 = 顶会 novelty。

---

## 7. 纪律(沿用)
- frozen 模型,权重不动;patch 可版本化/可回滚;本地 AST 验证,绝不 GPT judge。
- 所有数字标 PROBE;n<30 或单 seed 一律 PROBE。
- STRUCT 和 NL 必须信息量配平(核心对照有效性的前提)。
- 模型抽象成可替换接口(为阶段二 size sweep 铺路)。
- 记录能支撑阶段二相变分析的量(base 能力、STRUCT/NL 各自收益、context 成本)。
- 不删旧文件,复用实验一的环境/采集产出。

---

## 8. 文献引用纪律
- "结构 > 信息" 核心引 Guided-Structured Templates(EMNLP 2025,正刊)。
- HarnessFix/Self-Harness 措辞:"recent preprint",不说已发表顶会。
- NLT 只用于"小模型对格式/context 敏感",并明确区分输出格式轴 vs 推理结构轴。
- context 敏感性论点:自己实验测量自证,不引二手 token 数字。
