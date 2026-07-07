# GPU 调度设计

## 模型

1. 启动时解析 `nvidia-smi --query-gpu=index,memory.free`,空闲显存 ≥ `gpu.min_free_mem_gb`(默认 20GB)的卡各成一条**车道**;`gpu.max_lanes` 可封顶。
2. 每个 GPU 步骤是一个子进程,`CUDA_VISIBLE_DEVICES=<车道>` 独占一整卡,永不共卡、无跨卡并行。仓库内所有模型代码用 `device_map="auto"`,只会看到自己的车道。
3. 并发数 = 车道数:单卡即严格串行队列,N 卡最多 N 步并行。CPU 步骤(数据构建、报告)在依赖就绪时于调度进程内联执行。
4. 调度按依赖驱动而非阶段顺序:A5 的 teacher 重采占着 0 号卡时,A7 的训练可以在 1 号卡开跑。
5. 计划含 GPU 步骤而检测到零车道 → 拒绝启动(不做静默 CPU 回退——7B 前向在 CPU 上等于假死)。

## 各类步骤的显存预期

| 步骤类型 | 显存 | 说明 |
|---|---|---|
| 消融臂 LoRA 训练(QLoRA,M0 基) | ~18–24 GB | 4-bit 基座 + rank-16 LoRA,batch 1 |
| 消融臂前向评估(heldout/sibling/A6/A11) | ~12–18 GB | 仅生成 |
| **round 2 的 A2 训练 / M2 评估** | **~40–48 GB** | M1 merge 走 bf16 路径(冻结代码性质),用 ≥48GB 卡 |
| A6 dense 索引编码 | < 4 GB | MiniLM 级 embedder,CPU 也能跑 |

## 可调项(configs/local.json → "gpu")

- `min_free_mem_gb`:调高可排除忙卡/小卡。
- `max_lanes`:硬上限(设 1 = 完全串行)。
- 指定用卡:在 `scripts/run.py` 前设 `CUDA_VISIBLE_DEVICES`,检测到的编号相对该掩码。
- 混卡机器跑 round 2:用 `CUDA_VISIBLE_DEVICES` 圈定大卡,或把 `min_free_mem_gb` 调到小卡容量之上。

## 失败语义

- 某步 OOM 只判该步失败,其他车道继续,只有它的下游被阻塞;修复后重跑同一条 `scripts/run.py` 命令续跑(完成步骤按"状态 done + 产物存在"跳过)。
- 同时只允许一个调度进程(单一状态文件,未加锁)。
