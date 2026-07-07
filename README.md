# EDR: Evolve → Distill → Retire

**把外部脚手架教会的修复,内化进小模型权重——然后拆掉脚手架。**

- [1. 研究问题:为什么做这件事](#1-研究问题为什么做这件事)
- [2. 系统与方法:为什么这样设计](#2-系统与方法为什么这样设计)
- [3. 第一轮实验(已完成,结果冻结)](#3-第一轮实验已完成结果冻结)
- [4. 消融与对照实验:为什么需要、怎么设计、怎么跑](#4-消融与对照实验为什么需要怎么设计怎么跑)
- [5. 第二轮迭代(round 2):动力学问题](#5-第二轮迭代round-2动力学问题)
- [6. 快速开始](#6-快速开始)
- [7. 代码地图](#7-代码地图)
- [8. 输出、溯源与可信性](#8-输出溯源与可信性)

---

## 1. 研究问题:为什么做这件事

小模型在 function calling 上有一类失败,靠 prompting 永远修不好,但靠外部结构可以临时补上。我们在前期实验中实测:frozen 的 Qwen2.5-7B-Instruct 在 BFCL Multiple 的自然失败上,无干预修复率为 **0.0000**;让一个自进化 loop 为这些失败生成并验证 prompt patch(外部脚手架),held-out 修复率升到 **0.2405**(5 seeds)。scaffold 有效,这不是问题。

问题是**代价和终点**。被接受的 patch 会累积:每个 episode 平均要注入 **12.5k–15.3k tokens** 的 patch 上下文,而小模型恰恰对 context 增长最敏感。scaffold 越教越多、负担越背越重——**外挂式自进化没有终点**。脚手架在建筑学里本来就是施工期临时结构:建筑落成之日,即脚手架拆除之时。

于是核心问题变成:

> **scaffold 发现并被验证过的修复,能不能蒸馏成模型权重里的能力,在 scaffold 完全卸载后保留,并且不打碎模型原本会做的题?**

这拆成三个可证伪的主张:

- **C1(存在性)**:修复可以高保留率蒸馏进权重,卸载 scaffold 后仍在(Wipe Test)。
- **C2(机制)**:scaffold 教会的数据与模型自己采样碰对的数据在**类别上不同**——在采样根本够不到的失败上,scaffold 教学显著优于自采样蒸馏。一句话:**蒸馏教会的,不是蒸馏碰对的。**
- **C3(动力学)**:evolve→distill 循环多轮后是复利、收敛还是崩塌?三种结果都是有效刻画,只要测量干净。

第一轮实验(§3)回答了 C1 和 C2;本仓库自动化执行剩余的消融(§4,给 C1/C2 的每句话配上对照)和第二轮迭代(§5,回答 C3 的前半)。

## 2. 系统与方法:为什么这样设计

### 2.1 EDR 循环

```
M_0(frozen base)
  └─ Evolve:  在训练份失败上跑自进化 loop → 得到 harness H_1(accepted patches)
  └─ Distill: teacher = M_0+H_1 生成修复轨迹,过 verifier 后训练 LoRA A_1
  └─ Retire:  H_1 归档卸载;一切主评估在 M_1 = M_0+A_1 无任何 patch 状态下进行
  └─ 循环:   M_1 产生新的(更小更难的)失败集,进入第二轮
```

关键设计选择及理由:

**权重侧全部用可卸载的 LoRA adapter,不做不可逆 merge**(`src/edr/training/lora.py`、`src/edr/modeling.py`)。理由:任意轮可回滚到上一状态,adapter 是权重侧的"patch",延续 scaffold 侧的 rollback 纪律。

**唯一裁判是确定性 AST 匹配器,全程零 LLM judge**(`src/edr/verifier/ast_matcher.py`)。它对照人工标注 ground truth 判定:函数选择正确 ∧ 必需参数齐全 ∧ 参数值落在可接受答案集——是**任务级 oracle,不是格式检查**。为什么这么执拗:前期项目实证过"无外部验证信号的内部闭环不可靠"——模型自评是镜子,一致性选择会稳定地选中错误。因此**每一条**蒸馏样本必须过确定性 verifier,这条负结论是设计原则(§4 的 A5 臂就是把这条原则变成实验结果)。

**蒸馏样本的 input 必须是原始无-patch prompt**(`src/edr/data/distill.py` 的 `make_distill_row`)。patch 只出现在 teacher **生成** y 的过程中,绝不出现在训练输入里——否则学生学到的是"看着 patch 答题",Wipe Test 就没有意义。

**数据划分冻结 + 泄漏 assert 化**(`src/edr/data/splits.py`、`src/edr/data/leakage.py`)。哈希确定性划分(seed=20260630):train 313 / D_val 156 / D_heldout 158;外加两个回归面(sibling-300、old-400)。训练集与全部四个评估面的不相交由**代码 assert + pytest 强制**(`tests/test_frozen_protocol.py`),不靠自觉——第一轮曾发生过 D_val 泄漏进训练集的事故,根因是文档自相矛盾;此后"不许碰"一律 assert 化,assert 触发即停。

**为什么有两个回归面**:修复失败不能以打碎原本会做的题为代价。`forget = 新错比例`,硬闸 **forget ≤ 0.02**(部署级标准:2% 的既有能力损失是能向用户交代的上限)。sibling-300 是仲裁面,old-400 是历史延续面。

### 2.2 pass@16 分区:C2 的战场定义

C2 说"scaffold 教的和自己碰对的不一样",要让这句话可证伪,必须先定义"自己碰得到"的边界:对每个失败,base 模型 T=0.8 采样 16 次、AST 判定(`src/edr/scaffold/teacher.py` 的 `sample_base_episode`):

- **sampling-rescuable**:≥1 次通过——自采样够得到;
- **scaffold-only**:16 次全灭但 scaffold 修复了——只有教学够得到;
- **neither**:两者都没救。

heldout 158 个失败的分区:**scaffold-only 47 / sampling-rescuable 10 / neither 101**(冻结于 `data/round1/heldout_pass16_partition.json`)。teacher 修复的 50 个 heldout 失败里 **47 个(94%)是 scaffold-only**——大部分教学材料,base 自采样根本产不出来。**C2 的裁决场就是这 47 个 episode 上"scaffold 教学蒸馏 vs 自采样蒸馏"的配对差异。**

## 3. 第一轮实验(已完成,结果冻结)

以下每一步的产物都冻结在 `data/round1/`,复现管线在 `src/edr/data/round1_build.py`(五个子命令:prepare / run-shard / sample-teacher-shard / merge-shards / build-datasets),逐步复现协议见 [docs/reproduce.md](docs/reproduce.md) §3。

### 3.1 Evolve:自进化 loop 怎么工作

代码:`src/edr/scaffold/loop.py`(loop 本体)、`src/edr/scaffold/patches.py`(patch 结构与注入)、`src/edr/scaffold/patch_generation.py`(从失败物化 patch 文本)。

对训练份的每个失败,物化一个候选 patch(自然语言的修复指引),然后用**验证集上的机械判据**决定接受与否:`fixed > 0 且 regressed == 0`(`loop.py` 的 `update_validation_state`)——修好至少一个且一个不退步才收。不用任何"看起来更好"的判断。最终 **39 条 accepted patches** 构成 H₁(冻结于 `data/round1/evolution_h1.json`)。注入方式:按元数据相关性匹配(函数名/参数交集,`patches.py` 的 `patch_relevant_to_episode`——匹配只看元数据,不看文本,这个性质 A11 臂要用到),命中的 patch 以 `[PATCH i] ...` 拼进 prompt。

### 3.2 Distill:蒸馏数据与训练配方

代码:数据构建 `src/edr/data/round1_build.py build-datasets` + `src/edr/data/distill.py`;训练器 `src/edr/training/lora.py`。

- **teacher 生成**:对 train 份被修复的 74 个 episode,teacher(M₀+H₁)T=0 一条 + T=0.8 增广至 ×8。实测漏斗:592 采样 → 553 过 AST → **按 episode 去重后只剩 74 个唯一输出**——正确答案是窄点不是宽集("点质量分布"),继续加采不会带来多样性。core = 148 行(74 唯一输出 ×2 重复权重)。
- **replay 混入**:296 行 base 原成功轨迹(2:1)。为什么:蒸馏修复会把"参数要改"之类的规则错误泛化到原本做对的题上(遗忘/干扰),replay 钉住原能力。
- **KL 锚(λ=2,只打在 replay 行上)**:`lora.py` 的 `kl_anchor_loss`。为什么:第一轮网格监控发现 repair 和 forget 是**同一场分布移动的两岸**——剂量越大,越多低先验失败被推过正确边界,同时越多低 margin 原成功被推向最近的错误邻居。replay 只是重复正确答案,KL 锚直接约束输出分布不要离 base 太远,是对"margin 侵蚀"病因最贴的配方。训练含非有限值防护(nan loss/grad 跳批),因为第一轮真的出过 nan。
- **最终配方(锁定)**:rank 16 / lr 5e-5 / 3 epochs / replay 2:1 / KL λ2。这不是网格搜出来的最优点——是遗忘闸约束下走完预注册升级路径后的代表点,这个"如何选出"的历史正是消融臂必须锁配方的原因(§4)。

### 3.3 Retire 与评估

代码:`src/edr/evaluation/heldout.py`(主考场,含 pass@16 分层、2×2 泛化格、teacher 一致率)、`src/edr/evaluation/sibling.py` + `evaluation/parallel_arena.py`(回归面,含多调用回溯匹配判定)。

一切主评估在 **M₁ 无任何 patch** 状态下进行(Wipe Test)。对照臂 STaR:同样的训练器、同样的行数(148,配平方向永远向下)、但数据换成 base 自己 pass@16 碰对的轨迹——它只覆盖 31 个 unique episode,这不是不公平,**自采样的内生覆盖限制正是 C2 要测的现象**(行数配平的是训练预算,不是修改被测变量)。

### 3.4 第一轮结果(全部冻结,`data/round1/reference_evals/`)

| 量 | 结果 | 判读 |
|---|---|---|
| heldout 修复(5 seeds) | [54, 51, 46, 58, 43],均值 **50.4/158** | — |
| **retention(C1 主指标)** | 50.4 / teacher 50 = **≈1.008** | 卸载 scaffold 后学生≈teacher,内化成立,远超 0.60 的"强信号"预注册线 |
| **C2(scaffold-only 47 层)** | 教学蒸馏 − 自采样蒸馏 = **+0.153**,95% CI **[+0.064, +0.251]** | CI 不含 0,优势恰好集中在采样不可达层——C2 成立 |
| **forget 硬闸(≤0.02)** | sibling forget = **0.045** | **未通过**。诚实负结果:本配方族能创造修复能力,但尚未把既有能力损伤压到部署级 |
| 指定 M1 | seed 20260704(heldout 51/158,sibling 292/300) | 规则:五 seed heldout 修复取中位数,并列取小 seed(`data/round1/MANIFEST.json`) |

所以论文叙事是三件事同时成立:**高保留内化 + scaffold-only 机制优势 + 无回归点尚未找到**——第三件事催生了 §4 的 A7/A8,前两件事需要 §4 的 A5/A6/A11 来封死替代解释。

## 4. 消融与对照实验:为什么需要、怎么设计、怎么跑

### 4.0 统一规则(先读这个,再看各臂)

**为什么全部锁配方、不许逐臂调参**:主臂自己就没有享受过"网格选点"待遇(§3.2),给消融臂开网格反而破坏臂间对等("任何一臂不得获得额外调参轮次");更根本地,消融的语义是**单变量隔离**——A5 只动验证过滤、A7 只动 replay、A8 只动数据量,配方是控制变量,动了它效应就无法归因。全部臂:rank 16 / lr 5e-5 / ep3 / replay 2:1 / KL λ2(A7 例外见下),seeds {20260704, 20260705, 20260706}(与主臂 seed 块重叠,最大化配对检验功效),3-seed 数字一律标 `PROBE`,进 headline 才补 5 seeds。臂间差异只认 **episode 级 paired bootstrap CI(10k 重采)**(`src/edr/analysis/stats.py`),点估计和"看起来在涨"不是结论。

**一键全跑**:`python3 scripts/run.py ablations`(65 步 DAG 里的消融部分:数据构建 → 12 次训练 → 全部前向 → Gate-1 机械对号,步骤定义在 `src/edr/runner/steps.py` 的 `ablation_steps()`)。以下每臂也给出单独运行的入口,便于只跑一个臂或调试。

---

### A5 · unverified:verifier 到底必不必要?

**挡什么**:"你们那个 AST 过滤会不会根本无关紧要?teacher 输出直接训不就行了?"——这是把 §2.1 那条"无验证内部闭环不可靠"的设计原则实验化。A5 若显著劣于主臂 → 验证信号必要性成立;若不劣 → 如实报告并把原则措辞降级(预注册的诚实分支,不许删臂不报)。

**为什么这样设计**(每个选择都有理由):
- **episode 池 = 全部 313 个 train 失败,而非主臂的 74 个被修复 episode**。因为"哪些 episode 被修复了"这个信息**本身就是 verifier 的产物**——没有 verifier 的世界里你连这个都不知道。只关样本级过滤、保留 episode 级过滤,是个半吊子反事实。
- **配平 = 148 行 core,episode 级均匀下采样(每 episode 先取 T=0 行,不够再下潜高温行)**(`data/ablations.py` 的 `select_core_episode_uniform`)。行数与主臂配平是训练预算公平;不许从主臂的 74-episode 池里采(那又把 verifier 的选择偷渡回来)。预注册的后果声明:A5 会有 ~148 个 unique episode(主臂 74)、更浅的每题深度、混入错误输出——**这三者的合成效应就是"无 verifier 世界"的真实代价,不拆开归因**。
- **解析失败的输出计数报告但不训**。解析是机械管线步,不是 verifier;"unverified"指未经**正确性**验证,不是未经格式解析——这句话必须写进报告,防止被抠字眼。
- **replay 用冻结的 296 行原块**,一个字节不动——replay 不是被测变量。

**裁决场 = heldout 全集 158 的 paired CI**(不是 scaffold-only 层)。原则:**裁决场跟着假设的作用域走**——验证信号作用于全部训练数据,所以在全集裁决;对比 C2 的假设本身是层限定的,所以在 47 层裁决。预注册的不对称分支:全集打平但 scaffold-only 层显著更差 → 判"验证必要性**层限定**成立"。

**怎么跑 / 代码在哪**:

```bash
# ① teacher 重采,保留全部样本含 V=0(第一轮只存了成功样本,所以必须重采;
#    采样 seed 派生与第一轮逐字节相同,见 src/edr/io_utils.py 的 stable_int)
python3 -m edr.data.ablations resample-teacher --shard-index 0 --shard-count 2   # GPU,两片可双卡并行
python3 -m edr.data.ablations resample-teacher --shard-index 1 --shard-count 2
# ② 构建数据集(CPU;四面泄漏 assert 在这里强制)
python3 -m edr.data.ablations build-a5
# ③ 训练×3 seeds + heldout/sibling 评估(runner 自动做;手动则用 edr.training.lora + edr.evaluation.*)
```

采样逻辑 `src/edr/scaffold/teacher.py::sample_episode_keep_all`;数据构建 `src/edr/data/ablations.py`(resample-teacher / build-a5);输出 `outputs/ablations/a5/`(`a5_dataset_summary.json` 里能看到 AST 通过/失败行的构成)。

---

### A6 · retrieval-patch:为什么内化而不是 RAG?

**挡什么**:"何必训练?把 patch 建个索引按需检索注入不就行了?"——这是审稿必问,也是 context-成本 × 性能前沿上的合法竞争者。A6 输了是路线的事;A6 在某格赢了是前沿的真实形状,照报。

**为什么这样设计**:
- **双检索器(BM25 + 钉定 revision 的 all-MiniLM-L6-v2)× k∈{1,3} 四格,取最优格进前沿图、四格全报**。基线军备必须给足——只配一个弱检索器,一句"你的 RAG 基线是稻草人"就废了整张图。"对基线慷慨"是有意的:输给装备精良的基线才有说服力。
- **query = 用户指令 + 候选函数名列表(不含 schema)**。schema 几千 token 会撑爆 dense 模型 512 上限,而函数名已携带判别信号。doc = patch 全文。无 query 扩展、无 reranker、检索路径全程零 LLM(判定永远只有 AST)。
- **索引与逐 episode 检索结果冻结落盘 + 哈希**(`a6_frozen_retrieval.json`),评估只读冻结检索——离线可复现,dense 模型的 resolved revision 也钉在工件里。
- **预注册观察项**:任何一格修复数**超过** teacher 全量注入(50/158)→ 记为 **patch 间干扰**的直接证据单列(少给 patch 反而修得多,反过来支持"全量 scaffold 不可持续"的动机)。

**怎么跑 / 代码在哪**:

```bash
python3 -m edr.retrieval.patch_retrieval build-index   # 冻结检索(dense 编码可用 GPU,BM25 纯 CPU)
python3 -m edr.retrieval.patch_retrieval eval          # GPU:4 格 × heldout 158 前向,报 repair + 实测注入 tokens
```

代码 `src/edr/retrieval/patch_retrieval.py`(纯 python BM25 实现在同文件,零额外依赖);输出 `outputs/ablations/a6/a6_summary.json`(四格 + best_cell)。**不训练,纯前向。**

---

### A7 · replay-ablation:replay 买到了什么?

**挡什么/回答什么**:forget 闸没过(§3.4),replay+KL 是压遗忘的主要手段——那它们到底贡献了多少?去掉会怎样?

**为什么这样设计**:core 148 行、**零 replay**。物理连带:KL 锚的实现是打在 replay 行上的(`lora.py::kl_anchor_loss` 只对 `is_replay` 行计算),replay 归零后 KL 无处可打——所以 **A7 按定义 = "配方 − replay − KL"**,runner 显式传 `kl_anchor_lambda=0` 并在报告里如实标注。为什么不做"replay=0 但 KL 改打教学行"的臂:那是另一个变量组合,scope 已冻结,消融不许现场发明新杠杆。预期:sibling 面显著恶化(这本身就是 replay 价值的量化)。

**怎么跑 / 代码在哪**:`python3 -m edr.data.ablations build-a7a8` 构建(`outputs/ablations/a7/distill_a7_noreplay.jsonl`),训练评估由 runner 排队。构建逻辑 `src/edr/data/ablations.py::build_a7a8`。

---

### A8 · data-scale:效应随数据量怎么走?

**回答什么**:148 行 core 是不是刚好够?更少的数据能保住多少 repair、多付多少 forget?给"数据量"这个自然追问一条小曲线。

**为什么这样设计**:
- **episode 级子采样(25% / 50%),不是行级**。行级采样会造成 episode 覆盖参差(有的题只剩 T=0 行没有增广行),混进"覆盖形态"这个额外变量;episode 级哈希取前缀保证每个入选题的行结构完整。
- **replay 按 2:1 行数比随动缩放**(25% → 36 core + 72 replay)。比例是配方的一部分,绝对量不是——固定 296 行 replay 会让小数据点的 replay:core 比暴涨,又混进一个变量。
- **100% 点 = 主臂五 seed 结果原样入曲线,不砍不重训**。砍成 3 seeds 换表面整齐是扔信息且引入"选哪 3 个"的自由度;图上每点标注 seed 数,误差棒自己说话。

**怎么跑 / 代码在哪**:同 A7 的 `build-a7a8` 一并构建(`outputs/ablations/a8/distill_a8_{25,50}pct.jsonl`);子采样逻辑 `ablations.py::subsample_episodes`。

---

### A11 · placebo-patch:scaffold 效应只是 prompt 扰动吗?

**挡什么**:"你注入了 12.5k tokens,会不会随便塞点什么都能把分布扰动出 24% 修复?"——这是 C2 证据链的最后一块:零信息扰动隔离。

**为什么这样设计**:
- **token 级乱序**:每条 patch 用模型 tokenizer 切 token、打乱、拼回——长度与词表分布逐 token 保持,信息被摧毁。这比"换成随机文本"干净:排除了长度和词元统计这两个混淆。
- **注入路径与 scaffold-on 逐字节一致**:相关性匹配只看元数据不看文本(§3.1 的设计在这里兑现),所以乱序后每个 episode 命中的 patch 集合与真 scaffold 完全相同,唯一差异就是 patch 文本的信息含量。
- **三个存档乱序 seed {20260707, 20260717, 20260727},episode 在任一 seed 下过 V 即计入(并集)**。对 placebo 慷慨 = 对我们保守——placebo 只有一次机会太容易判"≈0"了。
- **"≈0"的操作化(先于数据写死)**:主判定面 = heldout(与所有臂同考场;train/val 面只作参照);并集率的 **Wilson 95% 下界 > 0** → "显著大于零",教学净收益按点估计扣除(= teacher 修复 − placebo 修复,CI 随附);下界 ≤ 0 → 排除扰动解释。不设 kill,只设诚实扣除。

**怎么跑 / 代码在哪**:

```bash
python3 -m edr.evaluation.placebo scramble   # CPU:冻结三套乱序 patch(带原文/乱序双哈希)
python3 -m edr.evaluation.placebo eval       # GPU:3 面 × 3 seed ≈ 1900 次前向
```

代码 `src/edr/evaluation/placebo.py`;输出 `outputs/ablations/a11/a11_summary.json`(per-seed + union)。**不训练,纯前向。**

---

### Gate-1 机械对号

全部臂跑完后 `python3 -m edr.analysis.gate1`(runner 自动触发)输出 `outputs/ablations/gate1_report.md`:A5 全集/分层 paired CI 与预注册分支判定、A6 四格前沿表 + patch-干扰旗标、A7/A8 描述表(每点标 seed 数)、A11 Wilson 判定 + 净收益。报告只做机械对号,横幅写明**解释需人签字**;判读细则(每个分支的预注册措辞)见 [docs/experiments.md](docs/experiments.md) §4。

## 5. 第二轮迭代(round 2):动力学问题

**回答什么**:C3 前半——把 M1 当新起点重走一遍 Evolve→Distill→Retire,循环是**复利**(M2 > M1)、**收敛**(打平但失败集腰斩——失败被消灭导致没得教,健康)、还是**崩塌**(M2 < M1 或遗忘破闸)?三类都是合法发现,分类由脚本机械判定,不许叙事性改判。

一键:`python3 scripts/run.py round2`(步骤 DAG 在 `src/edr/runner/steps.py::round2_steps()`)。逐步拆解——每步的"为什么"都标出来:

| 步 | 做什么 | 为什么这样设计 | 代码 |
|---|---|---|---|
| ① 验收 | 在本机重评 M1,heldout 51±1 / sibling 292±2 才打印 `ACCEPTED` | **不能复现 M1 的环境上跑出的 round 2 无效**——第一轮曾发生 adapter 被静默覆写导致评估结果不可解释的 provenance 事故,此后"复用 checkpoint 前必须对账"成为铁律 | `scripts/reconcile.py` |
| ② F2 收集 | M1 在 train share 313 题上 T=0 前向,残余失败 = F2;同时报 F1→F2 构成对比 | "哪些失败类型被内化消灭了、哪些残留"本身是动力学数据;被消灭的题(eliminated)还要喂给 ⑤ 的 replay 池 | `src/edr/round2/collect_failures.py` |
| ③ 等价性冒烟 | loop 的模型加载路径在 10 题上必须与 ② 逐字复现,不一致即停 | loop 和收集器是两条代码路径,都声称跑在"merged M1"上——这个**模型加载不变量**必须实测,不能信声明 | `src/edr/round2/equivalence_smoke.py` |
| ④ H2 进化 | 自进化 loop 跑在 **merged M1** 上,输入 = **F2**,NL 族,同款接受判据 | patch 要治的是 **M1 现在的病**,不是 M0 一年前的病——裸 M0 上搜出的 patch 对 M1 无效也无意义,所以入口直接拒绝无 adapter 运行;输入必须是 F2 而非第一轮失败集(已修好的题不需要再教) | `src/edr/scaffold/loop.py`(`--base-adapter-dir` 必填) |
| ⑤ T2 + replay 构建 | teacher-2 = M1+H2 生成,**AST fail-closed**(缺判定字段的行直接 assert,不静默入集);replay 池 = M1 T=0 成功 @ {第一轮 replay 的 120 题 ∪ ② 的 eliminated},只留 V=1,2:1 配比 | fail-closed 是因为第一轮发生过"字段缺失的行滑进训练集"类事故;replay 池必须用 **M1 自己的**成功轨迹(保护的是 M1 的当前能力面,不是 M0 的);\|T2\|<30 标 `MATERIAL_EXHAUSTION` 但**不中止**——材料枯竭本身是预注册的动力学发现 | `src/edr/round2/build_t2.py`、`build_m1_success.py`、`build_replay2.py` |
| ⑥ A2 训练 ×5 seeds | 锁定配方,**KL 锚 = 冻结的 M1,不是裸 M0** | KL 锚的职责是"别把现在会的忘了"——现在会的是 M1 的能力面;锚在 M0 上会把第一轮学到的东西也拉回去。实现:base 载入 → A1 merge 进权重 → 挂新 LoRA,此时 `disable_adapter()` 评估的正是冻结 M1(`training/lora.py::merge_base_adapter` 注释有说明);seeds {20260708…20260712} 预注册,传别的 seed 直接拒绝 | `src/edr/round2/train.py` |
| ⑦ M2 评估 ×5 | M2 = M0+A1+A2 无 patch,heldout+sibling 双面;retention₂ = repair(M2)/repair(M1+H2);2×2 泛化格的"见过"以 **T1∪T2 累积**定义 | teacher-2 前向只跑一次(seed 无关,5 次是浪费);"见过"必须累积否则第二轮把第一轮教过的题误标为泛化 | `src/edr/round2/evaluate.py`、`teacher2_forward.py` |
| ⑧ Gate-2 分类 | 5-seed paired bootstrap:compound / collapse / converge / flat,全部输入(F2 规模、实测 forget、M1 参照)从运行产物**机械读取**,零手填 | 手填数字是造假温床;forget 破闸优先判 collapse(能力涨但遗忘爆,循环也不可持续) | `src/edr/analysis/gate2.py`(classify + report 两个子命令) |

## 6. 快速开始

```bash
git clone -b edr-main --single-branch git@github.com:Yihao-DD/EDR-scaffold.git EDR && cd EDR
pip install -e '.[retrieval,dev]'

python3 tools/fetch_adapters.py          # 5 个 LoRA adapter(~800MB),SHA256 校验
# 基座模型 Qwen2.5-7B-Instruct 走 HF 自动下载;离线:configs/local.json 写 {"model_id": "/本地路径"}

python3 scripts/run.py preflight         # 必须 PASS(无 GPU 也能跑)
python3 scripts/run.py ablations         # §4 全部 + Gate-1
python3 scripts/run.py round2            # §5 全部 + Gate-2
python3 scripts/run.py report            # 汇总 REPORT.md
```

失败/重启后**重跑同一条命令**即断点续跑(状态 done 且产物存在才跳过,产物丢失自动重跑)。监控:`run.py status` + `tail -f outputs/logs/<step>.log`。多卡自动并行、每步独占一卡;显存要求(消融 ≥24GB,round2 需 ≥48GB 卡)与调参见 [docs/gpu_scheduling.md](docs/gpu_scheduling.md)。配置只允许改 `model_id`/`model_cache_dir`/`gpu.*`(写 `configs/local.json`),**其余全是预注册常量,改动即作废**。

## 7. 代码地图

| 功能 | 模块 | 关键入口 | 对应测试 |
|---|---|---|---|
| AST 判定(唯一裁判) | `src/edr/verifier/ast_matcher.py` | `call_success` / `parse_model_output` | `test_frozen_protocol.py::test_verifier_semantics` |
| patch 结构/相关性/注入 | `src/edr/scaffold/patches.py` | `build_prompt` / `patch_relevant_to_episode` | — |
| patch 文本生成 | `src/edr/scaffold/patch_generation.py` | `build_pair` | — |
| 自进化 loop | `src/edr/scaffold/loop.py` | `run_family_loop` / CLI | `test_pipeline_contracts.py::test_round2_loop_*` |
| teacher 采样(含 A5 keep-all) | `src/edr/scaffold/teacher.py` | `sample_episode_keep_all` | — |
| 数据划分(冻结) | `src/edr/data/splits.py` | `deterministic_split` / `load_round1_inputs` | `test_split_seed_20260630_*` |
| 泄漏 assert(fail-closed) | `src/edr/data/leakage.py` | `assert_training_disjoint` | `test_leakage_assert_*` |
| 蒸馏行/确定性采样 | `src/edr/data/distill.py` | `make_distill_row` / `deterministic_sample` | `test_distill_train_shape_*` |
| 第一轮复现管线 | `src/edr/data/round1_build.py` | 4 个子命令 | — |
| 消融数据构建 | `src/edr/data/ablations.py` | resample-teacher / build-a5 / build-a7a8 | `test_a5_*` / `test_a7_a8_*` |
| LoRA 训练(SFT+KL+防护) | `src/edr/training/lora.py` | `train` / `kl_anchor_loss` | 配方锁:`test_training_steps_use_locked_recipe` |
| 模型加载/生成 | `src/edr/modeling.py` | `load_model_for_eval` / `merge_base_adapter` | — |
| heldout 主考场评估 | `src/edr/evaluation/heldout.py` | CLI | — |
| sibling 回归面评估 | `src/edr/evaluation/sibling.py` + `parallel_arena.py` | CLI / `parallel_call_success` | — |
| A6 检索基线 | `src/edr/retrieval/patch_retrieval.py` | build-index / eval,`BM25` | `test_bm25_*` / `test_a6_query_*` |
| A11 placebo | `src/edr/evaluation/placebo.py` | scramble / eval | `test_a11_*` |
| round2 管线(8 步) | `src/edr/round2/*.py` | 见 §5 表 | `test_round2_dag_*` |
| 统计(bootstrap/Wilson) | `src/edr/analysis/stats.py` | `paired_bootstrap` / `wilson_ci` | — |
| Gate 机械对号 | `src/edr/analysis/gate1.py` / `gate2.py` | CLI | `test_gate2_classifier_categories` |
| 调度器(车道/续跑/DAG) | `src/edr/runner/` | `scheduler.py` / `steps.py` | `test_scheduler.py`(3 项实测) |

## 8. 输出、溯源与可信性

**落盘约定**:一切运行产物只落 `outputs/`,按阶段分层(`ablations/<臂>/`、`round2/`、`reconcile/`);每步日志 `outputs/logs/<step>.log`,调度账本 `outputs/state.json`。

**溯源链**:`REPORT.md` → 臂级 summary JSON → 逐 episode 评估 JSON → adapter `train_metadata.json`(含全部超参与损失曲线)→ 训练集 jsonl → `data/round1/` 冻结输入 → `MANIFEST.json`(逐文件 SHA256)。每一跳都是仓库里的文件。

**可信性装置**:泄漏由 assert+pytest 强制且 fail-closed;全部随机性显式 seed,采样流的 seed 派生字符串与第一轮逐字节一致(重采即复现);冻结协议(split 精确 id 集、参照数字逐 seed、判定行为)进测试钉死(`tests/`,25 项含调度器实测);报告全部机械生成,解释权留给人;负结果与正结果同权交付。

## License

MIT
