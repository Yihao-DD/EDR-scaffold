# EDR: Evolve → Distill → Retire

**把外部脚手架教会的修复,内化进小模型权重——然后拆掉脚手架。**

EDR 研究一个自包含的小模型自我改进循环:外部 scaffold(自进化出的 prompt patch)修复 frozen 7B 模型在 function calling 上的失败(**Evolve**),被确定性 AST verifier 认证的修复轨迹蒸馏进 LoRA 权重(**Distill**),然后 scaffold 完全卸载、只用模型本身评估(**Retire**)。Scaffold 是临时教师,不是永久假肢。

任务与基准:BFCL v4(Berkeley Function Calling Leaderboard)Multiple 类;基座模型:Qwen2.5-7B-Instruct(冻结);唯一裁判:BFCL AST 匹配器(任务级 oracle,对照人工标注的 ground truth 判定函数选择、必需参数、参数值——不是格式检查,全程无任何 LLM judge)。

## 核心问题与第一轮结果

> scaffold 在场时当然有帮助。真正的问题是:**它教会的东西能不能变成权重里的能力,并且不打碎模型原本会做的题?**

第一轮(已冻结,复现材料齐备)的回答:

| 量 | 结果 | 说明 |
|---|---|---|
| 内化保留率 retention | **≈ 1.0**(50.4/158 vs teacher 50/158,5 seeds) | 卸载 scaffold 后,学生达到 teacher 的 heldout 修复量 |
| 机制差异(scaffold-only 层) | 蒸馏 scaffold 教的 − 蒸馏自采样碰对的 = **+0.153**,95% CI [+0.064, +0.251] | 在 base 采样 16 次全灭的失败上,scaffold 教学显著优于 STaR 式自采样——"蒸馏教会的,不是蒸馏碰对的" |
| 回归硬闸 forget ≤ 0.02 | **未通过**(sibling forget = 0.045) | 诚实的负结果:本配方族能创造修复能力,但尚未把对原有能力的损伤压到部署级 |

本仓库交付两部分尚未完成的实验的**全自动执行**:五个消融/对照臂(A5–A11)与第二轮迭代(round 2),外加第一轮的完整复现管线。

## 快速开始

```bash
git clone <this-repo> && cd EDR
pip install -e '.[retrieval,dev]'        # 或 pip install -r requirements.txt

# 大文件(5 个 LoRA adapter,~800MB)不入 git,从 HF Hub 拉取并校验 SHA256:
python3 tools/fetch_adapters.py

# 基座模型走 HF 自动下载;离线环境把 configs/local.json 写入 {"model_id": "/本地路径"}

python3 scripts/run.py preflight         # 环境体检,必须 PASS(无 GPU 也能跑)
python3 scripts/run.py ablations         # 五个消融臂 + Gate-1 机械对号
python3 scripts/run.py round2            # 第二轮迭代 + Gate-2 动力学分类
python3 scripts/run.py report            # 汇总 REPORT.md
```

运行中:`python3 scripts/run.py status` 看进度,`tail -f outputs/logs/<step>.log` 看单步日志。**任何一步失败或机器重启,重跑同一条命令即可断点续跑**(完成的步骤自动跳过)。多卡自动并行、每步独占一卡;显存要求与调参见 `docs/gpu_scheduling.md`。

## 仓库结构

```
EDR/
├── src/edr/                 核心源代码(唯一的代码目录)
│   ├── verifier/            BFCL AST 判定(全项目唯一裁判)
│   ├── scaffold/            patch 结构 · 注入 · 自进化 loop · teacher 采样
│   ├── data/                数据划分 · 蒸馏集构建 · 消融变体 · 泄漏 assert
│   ├── training/            LoRA SFT + KL-anchor 训练器(含非有限值防护)
│   ├── evaluation/          heldout / sibling 回归 / validation / placebo 评估
│   ├── retrieval/           A6 检索基线(BM25 + 钉定版本的 dense embedder)
│   ├── analysis/            paired bootstrap · Wilson CI · Gate 机械对号
│   ├── round2/              第二轮管线(F2 收集 → H2 loop → T2 → A2 训练 → M2 评估)
│   └── runner/              调度器(GPU 车道 · 断点续跑 · 步骤 DAG)
├── scripts/                 薄入口:run.py(统一命令)· reconcile.py(验收)
├── tools/                   fetch_adapters.py · build_manifest.py
├── configs/                 base / ablations / round2(预注册常量;本机覆盖写 local.json)
├── data/round1/             冻结输入:失败集 · H1 patches · 蒸馏集 · 分区标签 ·
│                            episode 表 · teacher 参照 · 五 seed 参考评估 · MANIFEST(SHA256)
├── data/adapters/           (gitignore)fetch_adapters.py 拉取的 LoRA 权重
├── outputs/                 (gitignore)唯一落盘点,按阶段分层:
│                            outputs/ablations/<臂>/ · outputs/round2/ · outputs/reconcile/
│                            outputs/logs/ + outputs/state.json(调度账本)
├── tests/                   冻结协议校验 + 管线合同测试(pytest)
└── docs/                    experiments.md(实验与基线说明)· reproduce.md(复现协议)
```

落盘约定:代码永不写进源码树;一切运行产物落 `outputs/` 对应阶段目录,同一实验的 config 快照、指标 JSON、日志共存一处;每个数字可以沿 `REPORT.md → 臂级 summary → 逐条评估 JSON → adapter 训练元数据 → 数据集 jsonl → data/round1/ 冻结输入` 一路溯源到 SHA256。

## 实验设计一览

| 臂 | 回答的问题 | 形式 |
|---|---|---|
| A5 unverified | verifier 必要吗?(关掉 AST 过滤训练) | 3 seeds 训练 |
| A6 retrieval-patch | 为什么内化而不是检索注入?(BM25+dense × k∈{1,3}) | 纯前向,四格取优全报 |
| A7 replay-ablation | replay(连带 KL 锚)保护了什么? | 3 seeds 训练 |
| A8 data-scale | 效应随数据量如何缩放?(25%/50%/100%) | 3 seeds × 2 档 |
| A11 placebo-patch | scaffold 效应只是 prompt 扰动吗?(token 乱序 patch) | 纯前向 × 3 乱序 seed |
| round 2 | 循环第二轮是复利、收敛还是崩塌? | F2→H2→T2→A2×5 seeds→M2 |

所有配方(rank 16 / lr 5e-5 / 3 epochs / replay 2:1 / KL λ2)、seeds、判据、边界规则都是**预注册常量**,写死在 `configs/` 与 `docs/experiments.md`,执行零调参。Gate 报告只做机械对号(bootstrap CI / Wilson CI / 预注册分支),不含叙事判断;负结果与正结果同权交付。完整定义、每个参照数字的出处、判读规则见 [docs/experiments.md](docs/experiments.md);逐步复现协议见 [docs/reproduce.md](docs/reproduce.md)。

## 验收与可信性

- `scripts/reconcile.py`:在你的机器上重评估指定的第一轮模型 M1,heldout 51/158(±1)、sibling 292/300(±2)对上才打印 `ACCEPTED`——round 2 只有在能复现 M1 的环境上跑才有效。
- 训练数据与四个评估面(D_val / D_heldout / old-400 / sibling-300)的不相交由**代码 assert + pytest** 强制,不靠约定;assert 触发即停。
- 全部随机性可复现:数据划分、采样、乱序、bootstrap 的 seed 都是显式常量,采样流的 seed 派生字符串与第一轮逐字节一致(重采即复现)。
- `data/round1/MANIFEST.json` 记录每个冻结文件与 adapter 的 SHA256。

## License

MIT
