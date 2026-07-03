# v1.22 指令:rep2 复现包 + Phase 2 迭代代码 + code review + 推送 handoff 分支

目标:公司拿到分支即可冷启动跑第二轮迭代。rep1 修复线照旧并行,不阻塞本任务。

## 0. 分支与范围
- 新分支:`handoff/phase2-round2`(从 main 切出)。
- 范围:只含 rep2 线(干净线)。rep1 任何 artifact 不进本分支。
- Ian 随本指令提供的 HANDOFF 文档原文存为 `docs/PHASE2_HANDOFF.md`。

## 1. rep2 复现包(目录 `repro_rep2/`)
1.1 **代码**:`scripts/lora_phase0.py`(含 KL-anchor 与 non-finite guard)、评估 wrapper(heldout/val/old400/sibling 全套)、judge(单调用 + parallel 族扩展版,注明冻结 git hash)、全部 assert 与 pytest。
1.2 **数据(全部冻结文件 + SHA256 清单)**:distill_main core(148 行)、capped replay 池、D_val/D_heldout/old-400/capped-106/sibling-300 的 episode ID 列表、heldout pass@16 分区标签、per-arm 2×2 标签生成脚本。
1.3 **Artifacts**:main rep2 五个 seed 的 adapter + 训练 JSON + 日志;`MANIFEST.md` 逐文件记 SHA256 / 大小 / 训练配置 / seed / dry-run 对账行。
1.4 **M1 指定(按规则执行并记录)**:main rep2 五 seed 按 heldout repair 取中位数者(并列取 seed 号小者)。在 MANIFEST 顶部标注 `M1_DESIGNATED`,给出该 adapter 路径 + SHA256 + heldout/sibling 参考数字(公司对账用)。
1.5 **环境**:`requirements.txt` 钉版本(torch/transformers/peft/bitsandbytes 等)、CUDA 版本说明、模型获取说明(HF Qwen2.5-7B-Instruct,不进 repo)。
1.6 **一键复现脚本**:`repro_rep2/run_repro.sh`——从零训练任一 seed 的 rep2 并复现评估数字;README 给出逐条命令。

## 2. Phase 2 迭代代码(目录 `round2/`,全部新写或改造,要求可冷启动)
2.1 `collect_failures.py`:加载 M0+A1(M1),T=0 在 D_train share 上前向,输出 F2 清单 + 与 F1 的构成对比(哪些一轮失败已被内化消灭、新失败类型分布)。
2.2 `loop/`:从 EDG-EXP2-struct vendor 最小 NL-evo loop 代码(注明来源 commit),改造模型加载支持 base+adapter 栈;patch 接受判定仅在 D_val;judge 仅 AST。loop 配置锁一轮 NL-evo 同款。
    loop 全链路运行在 M1(merged)上;任何在裸 M0 上搜索/验证 patch 的运行无效。
2.3 `build_t2.py`:同 Step 0.2 规程(train 份 only;T=0 x1 + T=0.8 x4->8;AST 过滤;去重;逐条标注);泄漏 assert 四连(∩ D_heldout / D_val / old-400 / sibling-300 = ∅)写死在构建路径上。|T2|<30 -> 输出 `MATERIAL_EXHAUSTION` 标记但不中止(材料枯竭本身是发现,PROBE)。
2.4 `build_replay2.py`:M1 T=0 在 train share 成功(V=1)轨迹为池,排除全部 eval 集(assert),2:1 配比。
2.5 `train_round2.py`:**配方锁死** = r16 / lr5e-5 / ep3 / replay2:1 / KLλ2 / guards on;**KL 参考模型 = M1(冻结)**,不是 M0——代码里显式加载 M1 作 anchor,注释写明;A2 训练在 M0+A1 之上(实现:load base -> merge A1 in-memory -> attach 新 LoRA;A1 文件永不改动);seeds = {20260708..20260712}。
2.6 `eval_round2.py`:M2 = M0+A1+A2(顺序 merge 加载)无 patch 全套评估——heldout 158 全分层(47/10/101;2×2 的"seen"以 T1∪T2 累积定义)、retention_2 = repair(M2)/repair(M1+H2)(需 teacher-2 = M1+H2 的 heldout 前向,同脚本支持)、sibling 300(仲裁)/ old-400(continuity)/ val@156(monitor)三面齐报、Gate 2 分类器(复利/收敛/崩塌,按 5-seed CI 机械判)。
2.7 `reconcile.py`:公司验收脚本——pytest 全绿 + 用 M1 复现 heldout(±1 episode)与 sibling(±2)参考数字,全过才打印 `ACCEPTED`。

## 3. Code review(推送前,逐项打勾进 `docs/REVIEW.md`)
- pytest 双机全绿(含全部泄漏/arena assert);
- 每个新脚本最小规模端到端干跑(F2 收集 20 条、loop 1 轮 smoke、T2 mock、train 10 步、eval 单 checkpoint),贴输出;
- 无硬编码本地路径(统一 config/相对路径);无秘钥;requirements 钉死;
- README 命令逐条可复制执行;MANIFEST SHA256 全核;
- adapter 二进制确认 <100MB 直接入库,否则 LFS;
- KL 参考 = M1 这一行代码人工复核并在 REVIEW.md 单独确认。

## 4. 推送与登记
- push `handoff/phase2-round2`,记录分支 HEAD hash;
- CHANGELOG v1.22:M1 指定规则与结果、第二轮三项锁定决策(配方锁死 / KL 锚=M1 / 新 seed 块)、分支 hash、REVIEW 结论。

## v1.22 增补:EXP2 loop 的 adapter-aware 改造
1. vendor EXP2 loop 时,模型加载入口改为:base -> PeftModel(A1) -> merge_and_unload();merge 只在内存,A1 文件只读;adapter 路径走 config(M1_DESIGNATED),禁止硬编码。
2. loop 内所有 generate 调用(含 patch 在场重跑、验证、采样)共用 `round2/loop/evolution_loop_m1.py` 的 `ModelRunner`;grep 确认没有第二处独立加载模型的代码路径。
3. 等价性 smoke:改造后 loop 入口在 10 个 F2 episode 上无 patch 前向,输出应与 `collect_failures.py` 逐 episode 逐字一致;不一致即停并贴 diff。
4. T2 生成走同一入口:teacher-2 = merged(M1) + H2 patch-in-context。
5. `eval_round2.py` 的 teacher-2 前向走同一入口。
6. `REVIEW.md` 单列人工复核项:loop 无任何 M0 裸模型残留路径。
