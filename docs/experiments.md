# 实验与基线说明

面向读者:不了解项目史、需要执行或审计实验的人。回答"每个臂是什么、每个参照数字从哪来、结果怎么判读"。

## 1. 任务、模型、裁判

- **基准**:BFCL v4 Multiple 类 function calling。全部数据已内嵌(`data/round1/base_failures.json` 含 episode、函数池、ground truth、base 模型的成败记录),运行时零网络依赖。
- **基座** `M0`:Qwen/Qwen2.5-7B-Instruct,冻结。一切能力变化都装在可加载/卸载的 LoRA adapter 里。
- **裁判** `V`:BFCL AST 匹配器(`src/edr/verifier/`)——函数选择正确 ∧ 必需参数齐全 ∧ 参数值落在可接受答案集内。它是对照人工标注的**任务级 oracle**,不是语法/格式检查。数据过滤、验证、评估、分区中不存在任何 LLM judge。
- **Scaffold** `H1`:第一轮自进化 loop 产出的 39 条自然语言 patch(冻结于 `data/round1/evolution_h1.json`)。注入方式:按相关性匹配后以 `[PATCH i] ...` 拼进 prompt,平均成本 ~12.5k tokens/episode——这正是"内化后拆除"的动机。

## 2. 数据划分(冻结;泄漏由代码 assert 强制)

哈希确定性划分,seed=20260630,全项目唯一:

| 划分 | 规模 | 用途 | 训练权限 |
|---|---|---|---|
| train share | 313 失败 | 失败采集、全部训练数据 | 可训练 |
| D_val | 156 失败 | patch 接受、选点 | **绝不训练** |
| D_heldout | 158 失败 | **唯一主考场**,纯前向 | **绝不训练** |
| sibling-300 | 300 个 base 成功 | 回归竞技场(forget = 新错比例) | **绝不训练** |
| old-400 | 400 个 base 成功 | 历史回归延续面 | **绝不训练** |

heldout 失败的 pass@16 分区(base 模型 T=0.8 采 16 条):**scaffold-only 47**(采样永不成功、scaffold 能修)/ sampling-rescuable 10 / neither 101。scaffold-only 层是机制主张的裁决场。

## 3. 第一轮冻结结果(参照数字与出处)

| 量 | 值 | 出处 |
|---|---|---|
| teacher(scaffold-on)heldout 修复 | 50/158 = 0.3165 | `data/round1/teacher_heldout_reference.json` |
| 第一轮主臂 heldout 修复(5 seeds) | [54, 51, 46, 58, 43],均值 50.4/158 | `data/round1/reference_evals/seed*/heldout.json` |
| 第一轮主臂 sibling forget 均值 | 0.0453(> 0.02 硬闸,**未过**) | 同上 `sibling.json` |
| 机制对比(scaffold-only 层,主臂 − 自采样对照) | +0.153,95% CI [+0.064, +0.251] | 第一轮 5-seed paired bootstrap |
| 指定 M1 | seed 20260704(heldout 51/158,sibling 292/300) | `data/round1/MANIFEST.json`;规则=五 seed heldout 修复取中位数,并列取小 seed |
| 主臂训练集 | 444 行 = 148 core(74 episodes × T=0/T=0.8)+ 296 replay | `data/round1/distill_train.jsonl` |

## 4. 消融/对照臂(本仓库执行)

**统一规则(预注册)**:全部训练臂锁定第一轮配方(rank 16 / lr 5e-5 / 3 epochs / replay 2:1 / KL λ2 / 非有限值防护),零逐臂调参——配方是控制变量,不是实验变量;seeds = {20260704, 20260705, 20260706}(与参照重叠以最大化配对功效),3-seed 数字一律带 `PROBE` 标注,进 headline 才补至 5 seeds;臂间差异只认 episode 级 paired bootstrap CI(10k 重采),点估计不是结论。

### A5 unverified —— verifier 必要吗?

同一 teacher、同一管线,**关闭 AST 过滤**:对全部 313 个 train 失败重采 teacher(T=0 ×1 + T=0.8 ×8,保留全部输出),episode 级均匀下采样到 148 行 core(每 episode 先取 T=0 行),replay 用冻结的 296 行原块。行数与主臂配平;unique episode 数(~148 vs 74)、每 episode 深度、错误输出混入是"无 verifier 世界"的合成代价,**不拆开归因**。"unverified"指未经正确性验证——解析仍是机械管线步,不可解析的输出计数报告、不训练。
**判读**:裁决场 = heldout 全集 158 paired CI(裁决场随假设作用域:验证信号作用于全部训练数据)。显著劣于参照 → verifier 必要性成立;不劣 → 如实报告、措辞降级;预注册的不对称分支:全集打平但 scaffold-only 层显著更差 → "验证必要性**层限定**成立"。

### A6 retrieval-patch —— 为什么内化而不是 RAG?

不训练:对每个 heldout 失败检索 top-k patch 注入。双检索器(BM25 + `all-MiniLM-L6-v2`,revision 在冻结索引工件里钉死)× k∈{1,3} 四格;query = 用户指令 + 候选函数**名**列表(无 schema),doc = patch 全文;检索路径零 LLM、零 rerank;索引与逐 episode 检索结果冻结落盘。
**判读**:四格全报,最优格进 context-成本 × 性能前沿图(与 teacher 全量注入 ~12.5k tokens、主臂零 context 两点同图)。"对基线慷慨"是有意设计。预注册观察项:任何一格修复数**超过** teacher 全量注入 → 记为 patch 间干扰的直接证据,单列。

### A7 replay-ablation —— replay 保护了什么?

148 行 core、**零 replay**。KL 锚打在 replay 行上,所以 A7 按定义 = "配方 − replay − KL",报告如实标注。预期 sibling 面显著恶化。

### A8 data-scale —— 数据量缩放

core 按 episode 级哈希取 25% / 50% 子集,replay 按 **2:1 行数比**随动(比例是配方一部分,绝对量不是);100% 点 = 第一轮 5-seed 结果原样入曲线,**不砍不重训**,图上每点标注 seed 数,误差棒如实。

### A11 placebo-patch —— 只是 prompt 扰动吗?

每条 patch 做 token 级乱序(长度与词表分布保持、信息摧毁),**三个存档乱序 seed** {20260707, 20260717, 20260727},注入路径与 scaffold-on 完全一致,三个面(train 失败 / val 失败 / heldout)纯前向。
**判读**:主判定面 = heldout;episode 在**任一** seed 下过 V 即计入(并集——对 placebo 慷慨 = 对我们保守);并集率 Wilson 95% 下界 > 0 → "显著大于零",教学净收益按点估计扣除(= teacher 修复 − placebo,CI 随附);下界 ≤ 0 → "≈0",排除扰动解释。

## 5. 第二轮迭代(round 2)

在指定 M1 上重走整个循环:

1. **F2 收集**:M1 在 train share 上 T=0 前向,记录残余失败(与 F1 的构成对比 = 哪些失败类型被内化消灭)。
2. **等价性冒烟**:loop 的模型加载路径必须与 F2 收集器逐字复现(10 episodes),不一致即停。
3. **H2 进化**:自进化 loop 跑在 **merged M1** 上(裸 M0 的 patch 搜索无效,入口拒绝),NL 族,接受判据 = 验证集修复数上升且零回归。
4. **T2 构建**:teacher-2 = M1+H2,AST fail-closed 过滤,四面泄漏 assert;|T2| < 30 标 `MATERIAL_EXHAUSTION`(材料枯竭本身是预注册发现,不中止)。
5. **A2 训练**:锁定配方,**KL 锚 = 冻结 M1 而非裸 M0**,seeds {20260708…20260712}。
6. **M2 评估**:M2 = M0+A1+A2 无 patch,heldout + sibling 双面;retention_2 = repair(M2)/repair(M1+H2)(teacher-2 前向只跑一次)。
7. **Gate-2 机械分类**(`edr.analysis.gate2`):**compound**(M2>M1,CI 分离)/ **collapse**(M2<M1 CI 分离,或 forget 破闸)/ **converge**(CI 重叠且 |F2|<0.5|F1|)/ flat(其余,如实报 undetermined)。三种主类都是合法结果——分类是机械的,不许叙事性改判。

## 6. 判读纪律(执行者与审计者共同遵守)

1. 配方、seeds、判据、阈值不可改;想改 = 先改预注册、再跑数据,顺序不可倒。
2. 泄漏 assert 触发 = 结果,不是要绕过的 bug。
3. Gate 报告是机械对号;解释、取舍、措辞属于人。
4. 负结果(A5 不劣、placebo>0、collapse)与正结果同权交付,不软化。
5. 每个对外数字先查:干净来源?标注了 PROBE?报了 forget?配平证据在?
