# EDR 项目总纲:Evolve → Distill → Retire
## 面向顶会的完整研究程序 · v1.2 · 2026-07(修订记录见文末 CHANGELOG)

> 本文档是项目的唯一权威规范。本地 agent 执行任何步骤前必须通读本文档。
> 一切判据在此预注册;跑完对号入座,事后不许改。
> 语言:简体中文 + English 技术术语。所有未达 PUBLICATION 标准的数字标 `PROBE signal (n=X)`。

---

# Part I · 项目身份与目标

## 1. 一句话定义

**EDR(Evolve–Distill–Retire)**:一个自包含的小模型自进化循环——外部 scaffold 修复模型的失败(Evolve),被确定性 verifier 认证的修复轨迹蒸馏进模型权重(Distill),scaffold 随后卸载(Retire),模型在更强的自身上继续进化。**Scaffold 是临时教师,不是永久假肢。**

论文标题候选:
- *Scaffolding Is Meant to Come Down: Internalizing Self-Evolved Repairs into Small Model Weights*
- *Evolve, Distill, Retire: Self-Contained Self-Improvement for Small Function-Calling Agents*

## 2. 目标 venue 与达标标准

- **主目标**:ICML 2027 / ACL 2027 级别主会(截稿约 2027 年 1-2 月,**确切日期需临近时核实**)。
- **弹性目标**:若 Phase 0-2 结果强,冲 ICLR 2027(截稿约 2026 年 9 月下旬,需核实);不够则该窗口出 arXiv preprint 占位。
- **兜底**:任何时点,Phase 0-1 的诚实结果 = 一篇扎实 workshop / preprint。

**顶会达标的自我要求(缺一即降级投稿目标,不硬凑)**:
(a) 可证伪的科学主张而非"我做了个系统";(b) 完整对照臂系(含 STaR 对照、未验证数据对照、检索式 patch 基线);(c) ≥3 个模型规模的证据;(d) 多轮动力学刻画;(e) 全部机制有 ablation 隔离;(f) 统计协议达标(≥5 seeds、paired bootstrap CI);(g) 诚实的失效模式分析章节。

## 3. 科学主张(可证伪形式,论文的三根柱)

**C1(存在性)**:frozen 小模型自进化 scaffold 教会的修复,可以以高保留率蒸馏进权重,并在 scaffold 完全卸载后保持(Wipe Test)。
*证伪条件*:保留率持续 <40%(经一轮增广重试后)。

**C2(机制,最锋利的一根柱)**:scaffold 教会的数据与模型自采样碰对的数据在**类别上不同**——在采样不可达的失败上(pass@16 全灭),scaffold-taught 蒸馏显著优于 self-sampled 蒸馏(STaR/RFT)。**"蒸馏教会的,不是蒸馏碰对的。"**
*证伪条件*:唯 scaffold 可救区上两臂 CI 重叠。

**C3(动力学与规模)**:evolve→distill 循环的复利/收敛/崩塌行为,及其随模型规模的变化,可被系统刻画。
*注*:C3 无"失败"——复利、收敛、崩塌三种结果都是有效刻画,只要测量干净。

**明确不 claim 的(过度声称是顶会毙稿主因)**:
- 不 claim 对 xLAM/FunReason 等人工数据 fine-tune 方法的 SOTA;只报"自产数据缩小了与人工数据 fine-tuning 的差距 X%"。
- 不 claim 泛化到 reasoning/math 任务;scope 限定 function calling。
- 不 claim "scaffold-free 永远更好";我们刻画 context 成本 × 性能的前沿,检索式基线是前沿上的合法点。

**方法论贡献(第四条,与 C1–C3 并列写进论文贡献列表)**:内化评估 protocol 本身——**Wipe Test + 2×2 泛化分层 + 唯 scaffold 可救区对比 + placebo-patch 对照 + teacher 行为一致率**——作为任何 scaffold→weights 内化工作可直接复用的评估范式显式交付。这把项目从"一个 agent pipeline"升格为"scaffold-to-weight internalization 的一般评估范式 + 实证规律":别人拿走定义与 protocol,可在另一个 agent 系统上复现同样的判定。

---

# Part II · Philosophy 与设计理念

## 4. 四层 philosophy(由实证向理念递进,每层贴着本项目的数据)

**第一层(实证起点)**:小模型缺的能力,有一部分靠 prompting 永远够不到、有一部分靠外部结构可以临时补上。本项目实测:frozen 7B 在 BFCL Multiple 自然失败上,no-evolution=0.0000,scaffold 自进化后修复 24%(NL-evo held-out 0.2405,5 seeds)。

**第二层(问题)**:外部结构的补偿有累积成本。本项目实测:accepted patches 累积 12.5k–15.3k tokens context;小模型对 context 增长敏感。**scaffold 越教越多,负担越背越重——外挂式自进化没有终点。**

**第三层(主张)**:自进化的完成态,不是"scaffold 长得足够大",而是"scaffold 把自己教到不再被需要"。Scaffold 在建筑学里本就是施工期临时结构——**建筑落成之日,即脚手架拆除之时**。方法上:被确定性 verifier 认证的修复轨迹,是把结构性知识转移进权重的合法通道。

**第四层(边界,来自本项目最贵的教训)**:并非任何自我生成的数据都可训。CARE 数学线证明:**无外部验证信号的内部闭环不可靠**(自评是镜子,一致性选择稳定错误)。因此 EDR 的每一条蒸馏样本必须过确定性 AST verifier——**负结论在此成为设计原则,而非历史包袱**。未验证数据对照臂(Phase 1)把这条原则变成一个实验结果。

## 5. 本项目历史如何喂养本设计(agent 必须理解的因果链)

| 历史结论(已实证) | 在 EDR 中的角色 |
|---|---|
| 小模型无验证内部闭环不可靠(CARE 数学线) | 蒸馏数据必须 AST 认证;设未验证对照臂证明必要性 |
| 语义诊断可退化为字符串匹配(名字-解耦,4.70×→0.204×) | EDR 不依赖语义诊断;蒸馏数据由**实测修复**定义,不由诊断定义 |
| patch 形式无差异(STRUCT≈NL≈NL-padded,三方配平) | scaffold 选最简的 NL-evo;形式不是变量,**内化才是** |
| scaffold work(0→24%)但 context 成本 12-15k tokens | Evolve 环节直接复用;context 成本是 Retire 的动机且被量化 |
| 完美信号常是海市蜃楼(六次假信号史) | 全部 gate 预注册;CI 不分离不许下结论;PROBE 纪律 |

## 6. Related work 定位(已核实状态;引用措辞纪律)

| 工作 | 状态 | 与 EDR 的区分句(写死,related work 直接用) |
|---|---|---|
| **SIA** (arXiv 2605.27276, Hexo Labs, 开源, preprint) | 最近对手 | SIA 在 120B 模型上以 frontier 模型(Claude Sonnet 4.6)为 Meta/Feedback-Agent,每步在"改 scaffold"与"训权重"间二选一,scaffold 永远在场;EDR 是 7B 自包含(无 frontier teacher)、以本地确定性 verifier 认证数据、以"教→学→**退休**"为循环结构、以 Wipe Test 为核心指标。SIA 证明双杠杆有效;EDR 回答 SIA 未问的问题:内化后 scaffold 能否拆除,小模型能否不靠 frontier 教师完成循环。 |
| **Skill-to-LoRA** (arXiv 2606.16769, preprint) | 近邻 | 它把**人写的静态 skill** 编译进 LoRA 省 token;EDR 蒸馏的是**自进化 loop 自己发现并被 verifier 认证的修复**,且嵌在 evolve→distill 的循环中,并检验对未教实例的泛化。 |
| **STaR / RFT / ReST 系** | 方法对照 | 它们蒸馏模型**自己采样碰对**的轨迹(sharpening);EDR 蒸馏模型原本 0% 成功、**scaffold 教会后才成功**的轨迹(结构→权重的能力转移)。C2 是这条区分的直接实验。 |
| **Context distillation / PromptIntern / Experience Stripping** | 技术谱系 | 训练技术同源;EDR 的差异在数据来源(自进化自产 + verifier 认证)与循环结构(多轮 evolve-distill + retire)。 |
| **PACE** (arXiv 2605.23019, preprint) | 约束同源 | 同守"不依赖 frontier teacher";PACE 进化 prompt/control 且 scaffold 永在,EDR 内化并卸载。 |
| **TMEM** (arXiv 2606.04536, preprint) | 近邻 | 经验→parametric memory(LoRA);EDR 蒸馏的是修复行为而非记忆条目,且有 STaR 对照与泛化分层。 |
| **"Wipe Test"/Internalization Gap**(社区博客概念,2026-05;不作 citation,仅术语) | 评估框架 | ALFWorld-3B 上 skill 检索有/无 = 80.5/60.2 的 20 分 gap 是社区认识到的现象;EDR 把通过 Wipe Test 作为设计目标。 |

**引用纪律**:上表除注明外全部是 preprint——措辞用 "recent preprint",不写 "published at [venue]"。任何新引入的文献,先核实真伪与状态再进 related work(本项目有过 15/15 全真但也有过公司 preprint 冒充学术支柱的经历)。

---

# Part III · 系统形式化定义

## 7. 记号与 EDR 循环

- `M_0`:frozen base(Qwen2.5-7B-Instruct)。权重文件永不改动。
- `A_r`:第 r 轮训练出的 LoRA adapter(独立存盘,带版本号,可加载可卸载——**adapter 是权重侧的 patch,延续本项目 rollback 纪律**)。
- `M_r := M_0 ⊕ A_1 ⊕ … ⊕ A_r`(评估时按栈加载;不做不可逆 merge,保证任意回滚到 M_{r-1})。
- `H_r`:第 r 轮 Evolve 产出的 harness(accepted patches 集,沿用 EXP2 的 loop:诊断→生成→AST 验证→接受/回滚)。
- `V(·)`:BFCL AST verifier(router_success ∧ argument-match ⇒ call_success)。全项目唯一裁判,绝不 GPT judge。

**EDR 第 r 轮(r = 1, 2, …)**:
1. **Evolve**:在 `D_train` 上采集 `M_{r-1}` 的自然失败 `F_r`;跑自进化 loop 得 `H_r`(验证/接受在 `D_val` 上)。
2. **Teach(数据生成)**:`T_r = { (x, y) : x ∈ F_r 被修复的 episode 的原始无-patch prompt, y = [M_{r-1} ⊕ H_r](x), V(y)=1 }`,加温度增广(T=0 一条 + T=0.8 采 4–8 条,只留 V=1,去重)。
3. **Distill**:`A_r = LoRA-train(M_{r-1}, T_r ∪ Replay_r)`;`Replay_r` = `M_{r-1}` 自身成功轨迹按 30–50% 比例混入(防遗忘正则)。
4. **Retire**:`H_r` 归档卸载;所有主评估在 **`M_r` 无任何 patch** 状态下进行(Wipe Test)。
5. 循环:`M_r` 在 `D_train` 上产生新失败集 `F_{r+1}`(更小、更难),进入下一轮。

**关键不变量(每轮 assert)**:
- 蒸馏样本 input = 原始无-patch prompt,与 base 所见 byte 级一致;patch 只出现在生成 y 的过程中,绝不出现在训练 input。
- `episodes(T_r) ∩ episodes(D_heldout) = ∅`;`episodes(T_r) ∩ episodes(R_success_eval) = ∅`;`episodes(T_r) ∩ episodes(D_val) = ∅`(v1.2 新增;泄漏事故暴露 v1.0 assert 清单缺此项)。三条并列,代码 assert + pytest,每次构建数据强制执行。
- 每轮的 `H_r`、`A_r`、评估结果、随机种子全部版本化存档,任意轮可复现。

## 8. 数据划分与治理(全项目冻结,任何 Phase 不得改动)

| 划分 | 内容 | 用途 | 铁律 |
|---|---|---|---|
| `D_train` | BFCL Multiple(live_multiple 为主)训练份 | Evolve 失败采集 + 蒸馏数据来源 | — |
| `D_val` | 同上验证份 | loop patch 接受判定 + LoRA 选点 | 不进训练样本 |
| `D_heldout` | EXP2 已冻结的 held-out | **唯一主考场**(Wipe Test) | 任何轨迹绝不进训练;全程只在评估时前向 |
| `R_success` | base 成功 episode 抽 300–500 | 遗忘检测(评估份) | 评估份不进训练;replay 用其**不重叠**的另一份 |
| `R_other` | BFCL Simple 全量 + 可得的其他类别 | 跨类别回归检测 | 全程只评估 |
| 划分种子 | 固定并记录 | 全 Phase 复用同一划分 | 中途改划分 = 全部重跑 |

## 9. 度量词典(全项目统一定义;报告里出现的每个数字必须能溯源到本表)

| 度量 | 定义 |
|---|---|
| `repair(M, S)` | M 在集合 S(原失败 episode)上 T=0 前向、V=1 的比例 |
| `retention_r` | `repair(M_r 无patch, D_heldout) / repair(M_{r-1}⊕H_r, D_heldout)` —— Wipe Test 保留率,**主指标** |
| `wipe_gap_r` | `repair(M_{r-1}⊕H_r) − repair(M_r 无patch)`,同 heldout |
| 泛化 2×2 分层 | heldout 修复按 (函数是否在 T_r 出现过) × (错误类型是否在 T_r 出现过) 切四格:SS / SN / NS / **NN**;NN 格 >0 且 >base = 权重学到规律的判据 |
| `pass@16 分区` | train 失败集上 base T=0.8 采 16 条:**采样可救区**(≥1 条 V=1)/ **唯 scaffold 可救区**(0/16 且被 H 修复)/ 双否区 |
| `forget` | `1 − success(M_r, R_success_eval)/success(M_0, R_success_eval)`;硬性判据 forget ≤ 0.02 |
| `ctx_cost` | 每 episode 注入的 patch tokens 均值(scaffold-on ≈12.5k;distilled = 0;retrieval-k = 实测) |
| `overall(M)` | BFCL Multiple 整体 accuracy(成功保持 + 失败修复合并)—— benchmark 叙事数字 |
| `compound` | 复利:`repair(M_2 无patch) > repair(M_1 无patch)` 且 CI 分离(见 Gate 2 分类规则) |
| `teacher_agree` | heldout 上 distilled 输出与 teacher(`M_{r-1}⊕H_r`)输出的(函数名 + 参数集)一致比例;**修复集与 NN 格分别报**。修复集高一致 = 内化的是 teacher 的行为;NN 格 distilled 成功而与 teacher 路径不同 = 权重泛化出新解——两者皆为机制解释章节硬料。评估时顺手计算,零额外成本 |
| `placebo_repair` | 乱序 patch(token 级乱序:每个 accepted patch 内部 token 打乱,长度与词表分布保持、信息摧毁;乱序 seed 固定存档)以 scaffold-on 同样方式注入后,在失败集上的修复率。预期 ≈0;显著 >0 的部分记为**扰动效应**,教学净收益 = `repair(scaffold-on) − placebo_repair`,C2 叙事按此扣除 |
| 分层 retention | retention 按(**失败类型 × patch 类型**:router/validator 等)分层报告;另报修复构成中 parse-level vs semantic-level 的占比与各自内化率——"哪类 scaffold 能内化、哪类不能"即使在主结果平庸时也是独立科学结论,并顺带排除"LoRA 只学了格式适配"的替代解释 |

---

# Part IV · 实验总体设计

## 10. 实验臂全表(顶会完整臂系;各 Phase 按需启用)

| # | 臂 | 配置 | 回答什么 | 启用 Phase |
|---|---|---|---|---|
| A1 | base | M_0,无 patch | 地板 | 全部 |
| A2 | scaffold-on | M_0 ⊕ H_1(NL-evo,已知 0.2405) | 内化上界参照 / 被退休对象 | 全部 |
| A3 | **EDR-distilled** | M_1 无 patch | **主角**(C1) | 全部 |
| A4 | **STaR-distilled** | M_0 + LoRA(自采样碰对轨迹,量配平) | **C2 核心对照** | P0 起 |
| A5 | **unverified-distilled** | 同 A3 数据管线但**关闭 AST 过滤**(混入未验证输出,量配平) | 验证信号必要性(第四层 philosophy 的实验化) | P1 |
| A6 | **retrieval-patch** | M_0 + 按失败相似度检索 top-k patch 注入(k∈{1,3}) | context-成本前沿的中间点;封"为什么不 RAG patches"的审稿问题 | P1 |
| A7 | replay-ablation | A3 去掉 replay 混入 | replay 对遗忘的作用 | P1 |
| A8 | data-scale | A3 在 {25%, 50%, 100%} 蒸馏数据量 | 数据量 scaling 小曲线 | P1 |
| A9 | multi-round | M_2, M_3(无 patch) | C3 动力学 | P2 |
| A10 | multi-model | 3B / 14B / Llama-3.1-8B 各自完整 EDR | C3 规模 | P3 |
| A11 | **placebo-patch** | M_0 + 乱序 patch(长度/词表分布保持,信息摧毁;**不训练,仅前向**) | 排除"scaffold 只是扰动了 prompt 分布"的替代解释;C2 的 placebo 对照 | P1 |

**臂间公平铁律**:A3/A4/A5 训练数据条数配平(±10%)、同网格、同选点判据、同 seeds 数;任何一臂不得获得额外调参轮次。输的一方不能怪待遇。

## 11. 分层评估协议

每个臂在 `D_heldout` 上的 repair 一律输出五层:全集 / SS / SN / NS / NN(2×2)/ 以及按 pass@16 分区切的(采样可救区 / **唯 scaffold 可救区**)。**唯 scaffold 可救区上 A3 vs A4 的差距是 C2 的裁决数字。** 另:R_success、R_other、overall、ctx_cost 全臂必报;A3 加报 `teacher_agree`(修复集/NN 格两切片)与分层 retention(失败类型 × patch 类型;parse-level vs semantic-level 构成)。

## 12. 统计协议(顶会标准)

- Headline 数字(A3/A4 及 Gate 判定所涉):**≥5 training seeds**;超参网格允许单 seed,选点后重训 5 seeds。
- 臂间差异:heldout episode 级 **paired bootstrap**(≥10,000 重采样),报 mean ± 95% CI 与效应量;"显著" = CI 不含 0。
- 任何分层子集 n<30 → 该格数字标 `PROBE signal (n=X)`,不得单独支撑任何 claim。
- 评估全部 T=0(确定性);采样步骤(pass@16、增广)记录 temperature 与 seed。
- **禁止事后加指标救 claim**:报告里的指标集合 = 本文档度量词典;新增指标须在 `logs/deferred_metrics.md` 登记并说明动机,只能进"探索性分析"章节,不得进主结论。

---

# Part V · 分阶段执行(事无巨细)

## Phase 0 — 存在性验证(第 1 周,本地双 5090)

> 回答:C1 是否存在?C2 的战场有多大?
> (与已发的 AGENT_EXP3_distill_phase0.md 一致,此处为权威版;冲突以本文档为准。)

**Step 0.0 数据盘点(半天)**
1. NL-evo 最终 harness 在 `D_train ∪ D_val` 失败集上 T=0 重跑,得修复清单 `repaired_train_val.json`(**train/val 两份分开记账**)。**蒸馏数据池 = 仅 train 份修复**;val 份修复只记录、只作评估点,绝不入任何训练集(v1.2 修订:v1.0 此句原文"train∪val"与 Part III·8 治理表"D_val 不进训练样本"自相矛盾,泄漏事故根因在此)。
2. 固化 R_success(评估份/replay 份不重叠切分)、R_other、D_heldout 三个回归/考场集。
3. 泄漏 assert 落码并入 pytest。
4. 统计修复 episode 的函数/错误类型分布(供 2×2 分层)。
**分支**:**train 份**原始修复 ≥50 条 → 直行;<50 → 停,报告(补 loop 数据方案)。

**Step 0.1 pass@16 probe(半天,可与 0.0 并行)**
1. `D_train ∪ D_val` 全部失败,base 无 patch,T=0.8 × 16 条,V 判。
2. 输出三区标签;**头号数字 = 唯 scaffold 可救区占修复集比例**。
3. 采样通过轨迹全部留存(= A4 的训练数据,不重采)。

**Step 0.2 蒸馏集构建(1 天)**
1. 主集:**仅 train 份**修复 episode × (T=0 ×1 + T=0.8 ×4,不足加采到 ×8),V=1 过滤,去重,目标 300–500 条;逐条标注(episode_id / 分区 / 函数 / 错误类型)。
2. Replay 30–50% 混入(来自 R_success 的 replay 份)。
3. STaR 集:0.1 留存轨迹中**仅 train 份**,量配平 ±10%;若一方不足,以**较小集为基准对另一方下采样**(配平方向永远向下,两集最终条数均报),同比例 replay。
4. 全部过泄漏 assert。

**Step 0.3 训练(1–2 天)**
- peft + transformers,bf16 + gradient checkpointing;5090 32GB 单卡足够;显存吃紧退 QLoRA 并注明。
- 网格(v1.2 修订;动机:v1.1 网格已完成点在**干净的** R_success_eval 上 forget 8–9%,远破 ≤2% 硬闸):rank∈{8,16} × lr∈{2e-5,5e-5,1e-4} × epochs∈{1,2},replay 50%,cosine,warmup 5%,单 seed,共 12 点。**升级规则(预注册)**:12 点全破 forget 闸 → 取其中 repair 最高点,replay 提至 1:1 且 lr 减半重跑一点;仍破 → BLOCKED 报告,人决策。
- 选点判据(D_val,预注册):val repair 提升最大 **且** R_success 子样 ≥0.98;满足者取 repair 最高;全不满足 → 取回归最好档 + BLOCKED 报告。(v1.2 起训练与 D_val 零交集,D_val 全体 156 个失败均为干净选点集;**选点锁定前 heldout 一个数字不跑**。)
- 选定配置 **5 seeds** 重训;A4 同流程独立训练。

**Step 0.4 四臂评估(1–2 天)**
- 前置:heldout 失败集补跑 pass@16 分区标签(base 模型,T=0.8×16,纯评估侧前向,不触任何训练;网格期间空闲卡先跑)——唯 scaffold 区的分层评估依赖此标签,v1.0 的 Step 0.1 只覆盖了 train∪val。
- A1/A2/A3/A4,全 T=0,按第 11 节分层全报 + 第 9 节度量全报。
- 评估时顺手计算 `teacher_agree`(A3 各 seed 对 A2 输出的函数名+参数集一致率,修复集与 NN 格分别报)与分层 retention(失败类型 × patch 类型)。
- **交付**:主表(四臂 × 全分层 × CI)、两个头号数字(retention_1、唯 scaffold 区占比)、`teacher_agree` 两切片、分层 retention 表、Gate 0 逐条对号、pytest 双机。

## Phase 1 — 机制隔离(第 2–4 周,本地)

> 回答:哪些机制是必要的?context 成本前沿长什么样?为 C1/C2 的每一句话配上 ablation。

**Step 1.1 未验证数据对照(A5,约 3 天)**:同管线关闭 AST 过滤(teacher 输出无论 V 与否按比例混入至量配平),同网格同 seeds。预期显著劣于 A3;**若不劣,C 第四层 philosophy 的实验化失败,如实报告并把"验证信号在此设定下非必要"作为发现**——不许悄悄收窄。
**Step 1.2 retrieval-patch 基线(A6,约 2–3 天)**:失败 query 嵌入(可用轻量 embedding 模型,仅离线索引,不参与在线判定),检索 top-k(k=1,3)patch 注入;报 repair + 实测 ctx_cost。产出 **context-成本 × 性能前沿图**(A2 全量 patch / A6 检索 / A3 零 context 三点)。
**Step 1.3 replay ablation(A7,1 天)**:去 replay 重训(1–3 seeds),报 forget 变化。
**Step 1.4 数据量 scaling(A8,1–2 天)**:25%/50%/100% 三点小曲线(各 3 seeds)。
**Step 1.5 定性分析(1 天,人在场完成)**:抽 10–20 个 NN 格成功例与失败例,人工归纳"权重学到了什么/没学到什么"——论文 analysis 章节素材。**此步判断密集,agent 只做抽样与整理,结论由人写。**
**Step 1.6 placebo-patch 对照(A11,1 天)**:对 `H_1` 的每个 accepted patch 做 token 级乱序(patch 内部 token 打乱,保持长度与词表分布、摧毁信息;乱序 seed 固定并存档),以 scaffold-on 完全相同的注入方式,在 train/val 失败集与 `D_heldout` 上**前向评估(不训练)**。预期 `placebo_repair ≈ 0`;若显著 >0,按度量词典规则计算教学净收益并在 C2 叙事中显式扣除,写入报告与论文。此臂封死"scaffold 修复 = prompt 分布扰动"的替代解释——是 EXP2 NL-padded(长度隔离)之后的下一级对照:**零信息扰动隔离**。

## Phase 2 — 动力学(第 4–7 周,本地)

> 回答:C3 前半——循环是复利、收敛还是崩塌?

**Step 2.1 第二轮 Evolve**:`M_1`(无 patch)在 `D_train` 重新采失败 `F_2`(预期更小更难);loop 进化出 `H_2`;记录 F_2 规模、修复率、与 F_1 的构成对比(哪些失败类型被内化消灭了、哪些残留)。
**Step 2.2 第二轮 Distill**:`T_2` 构建(同 0.2 规程)、`A_2` 训练(同 0.3 规程,5 seeds)。**若 |T_2| 过小(<30 原始修复)**:如实记录"第二轮可教材料枯竭"本身作为动力学发现,r=2 以现有量训练并标 PROBE。
**Step 2.3 轮次评估**:`M_2` 无 patch 全套评估;若资源允许推 r=3。
**Step 2.4 动力学分类(按 Gate 2 规则对号)** + 逐轮曲线:no-patch heldout repair / F_r 规模 / T_r 量 / forget,四条曲线随 r。

## Phase 3 — 规模曲线(公司算力到位后,第 2–3 个月)

> 回答:C3 后半——内化能力随模型规模如何变化?

- 模型:Qwen2.5-3B / Qwen2.5-14B / Llama-3.1-8B(跨家族点),各自完整跑 Phase 0 + Phase 2 前两轮。
- **规模点有效性规则(预注册)**:某模型的 Evolve 环节原始修复 <30 条 → 该点标"loop 无效",**不纳入规模曲线**,单独报告(沿用 3B router 全 0 的教训:模型太弱做不了实验 ≠ 方法失效)。
- 已知先验:3B loop 弱(0.08)但最需要摆脱 context——**"loop 越弱、内化的相对价值是否越大"是本 Phase 的看点**,两个方向都是发现。
- 14B 本地 QLoRA 可试;A100 到位则 bf16。
- 产出:retention / NN 泛化 / compound 三条曲线 × 模型规模。

## Phase 4 — 广度(公司算力,第 3–4 个月,论文的 generalization 章节)

- BFCL 其余类别(Live 系列等)作为 `D_transfer`:**问"内化的能力(A3/M_2)比外挂 patch(A2)跨类别迁移得更好吗"**——这是旧"多任务"方向的正确回归位置,skill-transfer 红海没占这个问法。
- 可选第二环境(如可低成本接入的确定性 verifier 环境);不可得则如实以 BFCL 内跨类别为限并在 limitation 声明。

## Phase 5 — 成文(与 Phase 1–2 并行动笔;preprint 第 8 周)

- 第 8 周末:arXiv preprint v1(Phase 0–2 内容)——**占位由 R5(窗口)决定,不等 Phase 3**。
- Phase 3–4 完成后升级 v2,投主目标 venue。
- 论文骨架映射见 Part XI。

---

# Part VI · 预注册 Gates(全部,现在写死;跑完对号,不许改)

**Gate 0(Phase 0 末)**
- 主判据 `retention_1`:≥0.60 强信号 → 直进 Phase 1;0.40–0.60 可用 → 进 Phase 1,叙事补 ctx_cost 维度,增广加强列入 Phase 1;<0.40 → 允许**一轮**增广/超参重试,再 <0.40 → **停,贴全部数字,人决策**(候选:退 workshop 叙事 / 换蒸馏配方 / 停线)。
- 回归硬闸:forget ≤0.02 **且** R_other 无显著下降;违者先修遗忘再报数,任何 retention 数字不豁免此闸。
- C2 判据:唯 scaffold 可救区上 A3 > A4,5-seed CI 分离。成立 → C2 立;打平 → **如实记录,不许解读为"接近成立"**;此时看该区占比:占比 ≥20% 修复集 → C2 收窄为"该区存在但内化无差异"的诚实发现;<20% → C2 撤下,论文以 C1+C3 立骨。
- 泛化观察(记录不 gate):NN 格 >0 且 >base → 单独标注为超预期强信号。

**Gate 1(Phase 1 末)**
- A5(未验证)显著劣于 A3 → 验证必要性成立(第四层 philosophy 实验化完成);不劣 → 如实发现,philosophy 第四层措辞降级为"在更弱验证下亦可行",**不许删臂不报**。
- A6(检索)与 A3 的前沿关系如实呈现:若 A6 以低 ctx_cost 逼平 A3 → 论文诚实呈现前沿并把 EDR 的独特值收窄到"零 context + 泛化(NN 格)+ 循环复利";**检索赢在某点不是失败,是前沿的形状**。
- A11(placebo)`placebo_repair ≈ 0` → "prompt 扰动"替代解释排除,C2 证据链闭合;显著 >0 → 按度量词典规则机械扣除(教学净收益 = scaffold-on 修复 − placebo 修复),各处叙事以净收益为准——**不设 kill,只设诚实扣除;扣除后净收益若失去显著性,按 R2 预案处理**。

**Gate 2(Phase 2 末,动力学分类规则)**
- **复利**:`repair(M_2) > repair(M_1)`(无 patch,heldout),CI 分离。
- **收敛**:CI 重叠且 `F_2` 规模 < 0.5×`F_1`(失败被消灭导致可教材料减少——健康收敛)。
- **崩塌**:`repair(M_2) < repair(M_1)` CI 分离,或 forget 破闸。
- 三类都写进论文;分类由上述规则机械判定,**不许叙事性改判**。

**Gate 3(Phase 3)**
- 规模点有效性:原始修复 ≥30 条才入曲线(见 Phase 3)。
- 曲线 claim 需 ≥3 个有效规模点;不足 → 规模章节降级为 case 报告。

**全局诚实规则**:任何 gate 的"不成立"分支都必须完整跑完并写入报告;不存在"结果不好就不报"的臂。

---

# Part VII · 反自欺协议(本项目血泪的成文化;agent 与人共同遵守)

本项目历史上的六类造假/假信号发生地:patch 生成逻辑、收益估计、oracle 实现、诊断语义捷径、符号级海市蜃楼(CI 内的符号翻转被读成趋势)、测量污染。对应铁律:

1. **裁判唯一**:V = BFCL AST,全项目唯一判定;任何环节不得引入 LLM judge,包括数据过滤、评估、分区。
2. **预注册不可变**:Part VI 的全部判据、Part III 的度量定义、Part IV 的臂配置,自本文档定稿起冻结;修改需在 `CHANGELOG.md` 记录动机并由人签字,且已跑数据不得按新判据重新裁决。
3. **PROBE 纪律**:n<30 或 seed<3 的任何数字必须带 `PROBE signal (n=X)` 标签;PROBE 不得出现在 Gate 判定与论文主结论。
4. **CI 至上**:符号、点估计、"看起来在涨"一律无效;只有 CI 分离才是差异(3B 的 +0.0038 教训)。
5. **配平强制**:任何两臂对比,数据量/调参轮次/seeds 必须配平并在报告中给出配平证据(EXP2 的 IU parity 纪律延续到数据量层面)。
6. **泄漏 assert 化**:一切"不许碰"用代码 assert + pytest 覆盖,不靠自觉。
7. **判断密集步骤人在场**:蒸馏过滤逻辑变更、gate 对号解释、新指标引入、定性分析结论——agent 准备材料,人做结论;agent 不得在无人时修改这四类。
8. **不利结果同权交付**:每份阶段报告必须包含"本阶段对主张最不利的三个数字"专节。

---

# Part VIII · 工程规范

- 目录:本地 `E:\EDGscaffold\EDG-EXP3-distill`,服务器 `/root/autodl-tmp/EDG-EXP3-distill`;复用 EDG-EXP2-struct 的环境/评估/loop 代码(import/软链,不复制不重写);旧产出永不删除。
- 版本:`adapters/r{r}_seed{s}/`(config + 权重 + 训练日志);`harness/H{r}/`;每轮 `results/` `logs/` 按 step 命名;git 全程。
- 复现:每个数字可由(代码版本 + 数据划分种子 + 训练种子 + config)完全复现;报告中给出复现命令。
- 双机:关键结果本地 + 服务器 pytest 双过;泄漏 assert 在测试中显式覆盖。
- 显存:7B bf16 LoRA 单 5090 直训;14B 先 QLoRA;A100 到位换 bf16 并重跑 headline seeds。
- 评估管线:LoRA 栈加载进已有 BFCL 评估代码;T=0;tokenizer/模板与 EXP2 逐字节一致(模板漂移会污染所有对比)。

---

# Part IX · 风险登记册(与 Gate 联动)

| 风险 | 触发信号 | 预案(预注册) |
|---|---|---|
| R1 保留率低 | Gate 0 主判据 | 分档处理;<0.40 一轮重试后人决策 |
| R2 STaR 打平 | Gate 0 C2 判据 | claim 收窄/撤 C2,C1+C3 立骨;区占比决定叙事 |
| R3 遗忘 | forget >0.02 | replay 比例↑ / lr↓ / rank↓,修复前不报 retention |
| R4 多轮崩塌/材料枯竭 | Gate 2 分类 | 三类皆可发;枯竭本身是动力学发现 |
| R5 窗口关闭 | 近邻月更 | 第 8 周 preprint 硬节点,不等 Phase 3 |
| R6 "只是 context distillation/RFT" | 审稿预期 | A4+A5+A6+A11 四臂 + pass@16 分区 + teacher_agree + 循环结构,全部实验化防御 |
| R7 检索基线逼平 | Gate 1 | 前沿图诚实呈现,独特值收窄到零 context + NN 泛化 + 复利 |
| R8 3B/小模型 loop 无效 | Phase 3 有效性规则 | 该点出曲线,单独报告,不硬凑 |

---

# Part X · 时间线

| 周 | 内容 | 硬节点 |
|---|---|---|
| 1 | Phase 0 全部 | Gate 0 对号 |
| 2–4 | Phase 1(A5/A6/A7/A8 + 定性) | Gate 1 对号 |
| 4–7 | Phase 2(r=2,可选 r=3) | Gate 2 分类 |
| 8 | **arXiv preprint v1** | 占位,不等后续 |
| 月 2–3 | Phase 3(公司算力) | Gate 3 |
| 月 3–4 | Phase 4 + 论文 v2 | 投主目标 venue(截稿日期临近时核实) |

---

# Part XI · 论文骨架 ↔ 实验映射(写作时逐节取数)

| 论文章节 | 取自 |
|---|---|
| Intro:context 成本困境 + "脚手架该拆" | 12.5–15.3k tokens 实测;第四层 philosophy |
| Method:EDR 循环形式化 | Part III |
| Exp 1(C1 存在性):四臂主表 + retention + 前沿图 | Phase 0 + A6(P1) |
| Exp 2(C2 机制):pass@16 分区 + 唯scaffold区 A3 vs A4 + A5 消融 + A11 placebo + teacher_agree | Phase 0/1 |
| Exp 3(C3 动力学):逐轮四曲线 + 分类 | Phase 2 |
| Exp 4(C3 规模):三曲线 × 模型 | Phase 3 |
| Analysis:NN 格定性 + 分层 retention("哪类 scaffold 能内化")+ 失效模式 + 不利数字专节 | Step 1.5 + 各报告 |
| Limitations:scope(function calling)、单验证器、检索前沿 | Part I·3 + Gate 1 |

**写作备忘(强制段落,成文时不可省)**:
1. **AST 澄清段(Method/Setup 节,verifier 首次出现处)**:必须明确写出——BFCL 的 AST 评估是与人工标注 ground truth 的**匹配判定**(函数选择正确 ∧ 必需参数齐全 ∧ 参数值落在可接受答案集),属**任务级 oracle**,不是语法/格式校验。外部审稿演习已证明"AST = 格式检查"的望文生义会真实发生;论文必须在该词首次出现处主动拆除此误解,否则"verified trajectory"一词会被无谓攻击。
2. **Protocol 贡献段(Intro 贡献列表第四条)**:显式以"a reusable evaluation protocol for scaffold-to-weight internalization"名义交付 Part I·3 的方法论贡献。

---

## 结语(给 agent 的最后一句)

本项目的全部纪律来自一个已付过学费的事实:**看起来像进展的信号,大多数不是进展**。你的职责不是让数字好看,而是让数字**可信**——每一个 gate 的"不成立"分支,和"成立"分支一样,都是这个项目要的答案。先跑 Phase 0 的 Step 0.0 + 0.1,贴两个头号数字。

---

# CHANGELOG

**v1.0 → v1.1(2026-07-01;修订发生在 Phase 0 任何数据回传之前——预注册完整性保持)**

动机:对一份外部审稿模拟意见的批判性吸收(采纳 4 项、拒绝 2 项、纠正其 1 项事实错误)。

采纳:
1. 新增 **A11 placebo-patch 对照臂**(Part IV·10 / Phase 1 Step 1.6 / Gate 1 / R6):零信息扰动隔离,封"scaffold = prompt 分布扰动"替代解释。成本 1 天,不训练。
2. 度量词典新增 **`teacher_agree`**、**`placebo_repair`**、**分层 retention**(Part III·9 / 第 11 节 / Step 0.4):把 "internalization" 从形容词变成测量值;"哪类 scaffold 能内化"成为主结果平庸时的兜底科学结论;顺带排除"格式适配"解释。评估侧零成本。
3. **方法论贡献第四条**(Part I·3 / Part XI 写作备忘 2):内化评估 protocol 作为可复用范式显式交付。写作层成本。
4. **AST 澄清强制段**(Part XI 写作备忘 1):BFCL AST = ground-truth 匹配含参数值判定,非语法校验;预防审稿人望文生义(外部意见自身已演示此误解)。

明确拒绝(记录在案,防止反复):
- ratio 型主指标("accuracy retained per scaffold token removed"):复合比值指标是审稿靶子;A2/A6/A3 的 context-成本 × 性能前沿图表达同一信息且更稳。
- "verifier 强度 × 内化"自变量实验:单 benchmark 单 verifier 无法支撑该自变量,硬做即范围爆炸;记入 future work。

事实纠正:外部意见称 AST verifier"只能证明格式合法"——对 BFCL 不成立(见写作备忘 1);相应"更换更强 verifier"的隐含建议不采纳。

时间线影响:零(A11 的 1 天被 Phase 1 吸收);Gate 判据无实质变更,新增判据均为"诚实扣除/加报"型,不改变任何既有 kill/pivot 线。

签字:Ian 批准,2026-07-01。

---

**v1.1 → v1.2(2026-07-01;修订发生在任何 heldout 评估之前——heldout 零接触,预注册核心完整)**

触发:执行期人工核对发现 `distill_main`/`distill_star` 分别含 63/17 个 `D_val` episode。根因判定:**文档自相矛盾,责任在总纲不在 agent**——v1.0 Step 0.0 写"train∪val 重跑取修复清单"(蒸馏池未限定),与 Part III·8 治理表"D_val 不进训练样本"冲突,且 Part III·7 assert 清单缺 val 交集项;agent 按 Step 0.0 字面执行,assert 全绿。

审计结论:heldout ∩ training = 0,R_success_eval ∩ training = 0,两条硬泄漏线干净——**主考场未受污染,损害限于选点层**。数字内部一致(63+17−8 重叠 = 72 union;0.6346=99/156、0.4679=73/156 确认 val 选点分母)。

决策:按 Option A/B 预写规则(train 份修复 74 ≥ 50 且 val_clean 84 ≥ 30)→ **Option A:val 全部退出训练**,恢复治理表原文效力。

去污染粗估(PROBE 级,记录在案防止方向信号被误引):假设已训 episode 近全召回,干净 val 上 main ≈ 36/93 = 0.387,star ≈ 56/139 = 0.403;90% 召回假设下 main ≈ 0.455,star ≈ 0.415。**两假设下区间重叠——污染网格的 main>star 方向信号不成立,不得在任何场合引用**;C2 裁决场不变(heldout 唯 scaffold 区分层对比)。

修订项:Step 0.0/0.2 蒸馏池与 STaR 集限 train 份 + 配平方向规则(向下);不变量新增第三条 assert(training ∩ D_val = ∅);Step 0.3 网格保守化(rank{8,16} × lr{2e-5,5e-5,1e-4} × epochs{1,2},replay 50%,12 点)+ 预注册升级规则,动机为干净 R_success 证据 forget 8–9%;Step 0.4 前置补 heldout pass@16 分区标签;选点判据说明补全。

污染产出处置:v1.1 网格全部 checkpoint 与日志**归档留存**(目录加 `contaminated_gridv1/` 标签 + README 指向本条),标 `PROBE (contaminated)`,永不作为选点/报告/论文依据;已完成 checkpoint 仅可用于遗忘类型诊断(R_success_eval 干净)。

时间线影响:Phase 0 +2–3 天;第 8 周 preprint 节点不变。

签字:Ian 批准(Option A 规则触发,2026-07-01)。
