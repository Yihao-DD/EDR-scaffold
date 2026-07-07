# 复现协议

三个层次,按需选择。全部命令在仓库根目录执行。

## 0. 环境

- Python ≥ 3.10;`pip install -e '.[retrieval,dev]'`(依赖全部钉版本)。
- 基座模型 Qwen/Qwen2.5-7B-Instruct:联网自动下载,或离线预下载后在 `configs/local.json` 写 `{"model_id": "/绝对路径"}`(`local.json` 不入 git,只允许覆盖机器相关键)。
- Adapter 大文件:`python3 tools/fetch_adapters.py`(从 MANIFEST 记录的 HF 仓库拉取并校验 SHA256)。
- 显存:消融臂任意 ≥24GB 卡(QLoRA);**round 2 的 A2 训练与 M2 评估需要 ≥48GB 卡**(M1 bf16 merge 路径)。多卡自动并行、每步独占一卡,细节与调参见 `gpu_scheduling.md`。
- 开跑前:`python3 scripts/run.py preflight` 必须 PASS。

## 1. 验收复现(~1 小时 GPU):这台机器能复现第一轮吗?

```bash
python3 scripts/reconcile.py
```

重评估指定 M1(seed 20260704)于 heldout 158 与 sibling 300。heldout 修复 51±1、sibling 成功 292±2 之内 → 打印 `ACCEPTED`。**round 2 只在 ACCEPTED 的环境上跑才有效。**

## 2. 剩余实验执行(每阶段约 1–2 天 GPU)

```bash
python3 scripts/run.py ablations   # A5/A6/A7/A8/A11 → outputs/ablations/gate1_report.md
python3 scripts/run.py round2     # F2→H2→T2→A2×5→M2 → outputs/round2/gate2_report.md
python3 scripts/run.py report     # 汇总 REPORT.md
```

- 断点续跑:失败/重启后重跑同一命令;已完成步骤按"状态=done 且产物存在"跳过。
- 监控:`scripts/run.py status`;单步日志 `outputs/logs/<step>.log`。
- 完成判据:`status` 中对应前缀(`ab.*` / `r2.*`)全部 `done`,且 gate 报告存在。Gate-2 的 compound/converge/collapse 三种分类都是合法结果。
- 常见故障:CUDA OOM → `configs/local.json` 写 `{"gpu": {"max_lanes": 1}}` 或用 `CUDA_VISIBLE_DEVICES` 圈定大卡;HF 下载失败 → 预下载改本地路径;`AssertionError {..._disjoint_...}` → 泄漏 assert,**停、原样上报,不许绕过**;A6 dense 缺依赖 → `pip install -e '.[retrieval]'`。

## 3. 第一轮全量复现(可选,~数天 GPU)

从零重建全部冻结数据(结果应与 `data/round1/` 中的文件一致;采样 seed 派生与第一轮逐字节相同):

```bash
# 1) teacher T=0 + base pass@16 全扫(可分片并行)
python3 -m edr.data.round1_build run-shard --shard-index 0 --shard-count 2 --output outputs/round1_build/shards/shard_0.json
python3 -m edr.data.round1_build run-shard --shard-index 1 --shard-count 2 --output outputs/round1_build/shards/shard_1.json
python3 -m edr.data.round1_build merge-shards

# 2) teacher T=0.8 增广(train 份修复 episode)
python3 -m edr.data.round1_build sample-teacher-shard --shard-index 0 --shard-count 1 \
    --output outputs/round1_build/teacher_samples/teacher_shard_0.json

# 3) 蒸馏集(train-only,泄漏 assert 全程)
python3 -m edr.data.round1_build build-datasets

# 4) 任一 seed 重训第一轮主臂并复评(配方即 configs/base.json 的 recipe_locked)
python3 -m edr.training.lora --model-id Qwen/Qwen2.5-7B-Instruct \
    --dataset data/round1/distill_train.jsonl \
    --output-dir outputs/round1_build/adapters/seed20260704 \
    --rank 16 --lr 5e-5 --epochs 3 --seed 20260704 --kl-anchor-lambda 2.0
python3 -m edr.evaluation.heldout --model-id Qwen/Qwen2.5-7B-Instruct \
    --adapter-dir outputs/round1_build/adapters/seed20260704 \
    --arm round1_main --config repro --seed 20260704 --run-id repro_seed20260704 \
    --train-dataset data/round1/distill_train.jsonl \
    --output outputs/round1_build/heldout_seed20260704.json
```

```bash
# 5) STaR 对照臂(A4)同法重训——同一训练器、同一评估器,只换数据。
#    注意 lr=1e-4:预注册对称规则是各臂用自己网格 repair 最高点的 base lr
#    (main 5e-5 / STaR 1e-4),两臂 adapter 的 train_metadata.json 均可核对:
python3 -m edr.training.lora --model-id Qwen/Qwen2.5-7B-Instruct \
    --dataset data/round1/distill_star_train.jsonl \
    --output-dir outputs/round1_build/adapters/star_seed20260704 \
    --rank 16 --lr 1e-4 --epochs 3 --seed 20260704 --kl-anchor-lambda 2.0

# 6) teacher 分母前向(复现 50/158):
python3 -m edr.evaluation.teacher_forward

# 7) C2 头号数字重算(纯 CPU,直接对冻结参考评估做配对 bootstrap):
python3 -m edr.analysis.c2
```

对账:与 `data/round1/reference_evals/main/seed20260704/heldout.json`(STaR 对 `reference_evals/star/`)的 summary 比对(GPU 硬件差异可能带来 ±1 episode 级波动;`scripts/reconcile.py` 的容差即为此设);C2 重算的点估计应与 §3.5 表逐位一致。

## 4. 溯源链

`REPORT.md` → 臂级 summary JSON → 逐 episode 评估 JSON → adapter `train_metadata.json` → 训练集 jsonl → `data/round1/` 冻结输入 → `data/round1/MANIFEST.json`(SHA256)。每一跳都是仓库里的一个文件。
