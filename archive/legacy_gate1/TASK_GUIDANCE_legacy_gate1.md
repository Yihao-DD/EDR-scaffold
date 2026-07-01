任务:从零搭建多组件 edge scaffold governance 验证器(Gate-1)
你是一个全新的 agent,对此项目没有任何先验上下文。本指令包含你需要的全部信息。严格按顺序执行,前一步不通过不写后一步。不要自行扩大范围、不要跳过任何 gate、不要"优化"掉任何反捷径约束。

0. 这个项目在做什么(一句话)
研究:一个 frozen 小模型 agent 在任务失败后,当多个外部组件(工具路由器 / 参数校验器)都能被修补、但本地验证预算有限时,该把预算优先投给哪个组件。 我们要证明:自适应预算仲裁,在紧预算下优于"均分 / 随机 / 单组件 / 全修"。
关键约束:模型权重永不更新。 所有"更新"都是对模型外部的、可版本化、可回滚的配置 artifact(路由规则 / 参数规范化规则)做小补丁(atomic patch),用有限本地验证预算回放验证,过则接受、不过则回滚。
你不是在优化 benchmark 的官方分数。你是在本地可判的修复子任务上,验证"预算仲裁"这个机制是否真实存在。

1. 背景:为什么有一堆"反捷径"约束(必读,否则你会重蹈覆辙)
这个项目的前身用一个自建环境跑出过漂亮结果(某机制省了 24× 预算),后来用一个检验证明:那个机制根本没在做它声称的事,它的全部"效果"来自一个隐藏的字符串匹配捷径(组件名字和失败标签字面撞上就得分),一旦移除捷径,效果归零。
教训:一个能通过常规指标(预算节省、泛化)的机制,可能实际在测一个预埋的捷径,而不是你以为的能力。 本 gate 的所有反捷径约束,都是为了在第一天就堵死这类捷径。你每一条都必须实现,不能因为"看起来多余"就省略。 如果你发现某个结果"好得可疑",先怀疑是捷径,做下面规定的 ablation 验证,而不是接受它。

2. 环境:STEP 0 是 go/no-go 数据探针(第一行代码,不是形式)
不要假设环境就绪。第一个 commit 只做一件事:验证数据。
我们计划用 StableToolBench / ToolBench 数据。它的官方任务级评测依赖 GPT-4 judge——那个我们不用(理由见 §4)。我们只需要它的一个本地字段做工具路由的 gold 信号:每条 ToolBench instruction 应有 relevant APIs(真正该用的工具,router 的 gold)和 api_list(候选工具池,含干扰项,不是 gold)。
但 StableToolBench 是从 ToolBench 过滤重打包的,relevant APIs 字段可能在你下载的文件里丢了、改名了、或只在另一个需 query_id 回连的文件里。这是文档说有、但你硬盘上未必有的东西。必须实测。
STEP 0 数据探针(~30 行,先跑):
在你真实下载的 StableToolBench/ToolBench 数据文件上,逐 episode 检查能否读到三个字段并报告覆盖率:
python# 对每条评测 episode,断言可读:
#   - query_id
#   - api_list          (候选工具池,可能含 distractor,不是 gold)
#   - relevant APIs     (router 的 gold;可能需要用 query_id 回连到 ToolBench instruction/answer 数据)
# 报告:总 episode 数,三字段齐全的比例(coverage)
assert "relevant APIs" in sample or can_recover_relevant_apis(sample["query_id"])
go/no-go 判据:

coverage 高(relevant APIs 真在/可回连) → 解法 A 成立,用 StableToolBench,继续 STEP 1。
coverage 低 / 字段缺失且无法回连 → 当场切换到 BFCL(Berkeley Function-Calling Leaderboard)。理由:BFCL 用 AST evaluation 本地判定"函数选对没有"+"参数对没有",天然同时覆盖路由和参数两端,正好补上 StableToolBench 缺 router gold 的短板,且不需要 GPT-4 judge。

先把 STEP 0 的 coverage 结果输出出来。high → StableToolBench;low → BFCL。两条路后续信号定义见 §3。不要在没跑 STEP 0 的情况下开始写验证器。

3. 两个组件 + 对称的本地修复信号(STEP 0 过了才写)
只做两个组件,且两者的修复成功必须都能本地判、且对称(一个能测、一个测不到 = 验证器半盲 = gate 作废)。
组件与 patch 形态:
组件一次 atomic patch 的形态(可版本化/可回滚)Tool router加一条路由规则 / 加 tool alias / 调工具优先级 / 加一条 routing 示例Argument validator/parser加 field alias / 加类型转换 / 加 required-field 检查 / 加 parser fallback
本地修复成功信号(StableToolBench 路):

router_success = selected (tool_name, api_name) ∈ relevant APIs

必须用 (tool_name, api_name) 对,不能只比 tool_name(同一 tool 可能多 api)。
不能用 api_list 当 gold(它含干扰项)。不能用"API 返回非错误"当 router 成功(选错工具也可能返回非错误)。


validator_success = json_parse_ok ∧ schema_valid ∧ required_fields_present ∧ type_ok ∧ no_local_parsing_or_parameter_error
call_success = router_success ∧ arguments_locally_valid ∧ no_local_call_error(联合信号,仍不等于官方 task success)

本地修复成功信号(BFCL 路):

router 与 validator 都用 BFCL 的 AST evaluation 拆解:函数名匹配 = router_success,参数 AST 匹配 = validator_success。两端天然对称、本地 deterministic。

写完信号,各跑几条 episode 自检:确认 router 和 validator 两个信号都能产出非平凡的 success/fail(不是一个永远 1、一个永远 0)。这是补半盲的全部意义。 若有一端恒定,停下排查,不要继续。

4. 四条反捷径硬规则(写进验证器/allocator,逐条实现,不可省)
规则 1 — allocator 只读历史实测,不读失败语义。
第一版 allocator 是 pure history-based(non-contextual):
score(component) = 历史平均 net gain(该组件) / 历史平均 cost(该组件)
紧预算下,优先验证 score 高的组件。
它不读 failure type、不读 tool name、不读 error name、不读任何失败描述文本。 只读各组件历史实测的 net gain / cost / regression / latency。
(理由:若 r̂ 来自"wrong_tool → router 收益高"这种失败类型推断,就是上次字符串捷径的换皮。)
规则 2 — 每个失败为两个组件都生成候选 patch,不按失败类型预筛。
一个失败 episode,同时生成 router patch 和 validator patch,哪怕其中一个大概率没用。
不允许"看起来是 wrong_tool 就只生成 router patch"。
(理由:若候选生成按失败类型派生,"该修哪个组件"在生成阶段就被预设,allocator 只是走过场。)
第一版 patch 可以是模板化/规则化生成,不做复杂 LLM 生成——但模板不能按失败类型筛选组件。
规则 3 — oracle 是穷举实测上界,不是人工标注。
oracle allocator 定义为:穷举两个组件的所有候选 patch,在本地 validation suite 上实测每个的 net gain,选实测净收益最高者。
不允许人工标"这个失败应该修 router"当 oracle。
(理由:失败常多解,人工标注会把标注者偏见变成上界,且重蹈"唯一正确 target"陷阱。empirical oracle 自动处理多解。)
规则 4 — 在线验证只用本地信号,绝不调云端 judge。
online validation loop 内禁止 GPT-4 / 任何 LLM-as-judge / SoPR / SoWR。
只用 §3 的本地 deterministic 信号。
官方 GPT-4/human task-level 评测只能作为 offline sanity check 另外报告,绝不进入在线验证预算闭环。
(理由:每次 patch 验证调 GPT-4 会让预算大头变成 judge 钱,且端侧系统依赖云端 judge,edge 叙事破。)

5. 实验组织(STEP 1–5)
STEP 1 — frozen baseline 采集失败。
用一个 frozen 小模型(Qwen2.5-7B-Instruct,服务器单卡 48GB 够推理)在 benchmark 子集上跑 agent,收集失败 episode:episode_id / task input / trajectory / tool calls / API responses / final answer / 本地判定的失败类型 / 是否失败。
建议先用较简单子集(如单工具或短任务),避免多工具顺序/多解路径一开始把信号搞复杂。
STEP 2 — 候选生成(遵守规则 2)。 每个失败给两个组件都生成模板化候选 patch。
STEP 3 — allocator 选择(遵守规则 1)。 在预算约束下,history-based allocator 决定先验证哪个组件的 patch。
STEP 4 — 本地验证(遵守规则 4)。
不在 test set 上验证。在小 validation suite 上回放:apply patch → 重跑受影响的 validation episodes → 测 success 提升 + regression + latency/context/memory cost → 算 net utility → 过则接受、不过则回滚。
预算单位明确:1 validation call = 1 个 patch 在 validation episode 上回放一次。
STEP 5 — held-out 评测。 最终只在 held-out benchmark episodes 上报本地组件级修复指标。
accepted update 必须记日志(供后续审计,证明不是黑箱调 prompt):
json{"update_id":"...", "component":"...", "trigger_failure_ids":[...],
 "patch_type":"...", "patch":{...},
 "validation_cost":{"calls":N,"latency_ms":N,"context_tokens":N},
 "validation_result":{"fixed":N,"regressed":N,"net_gain":N},
 "accepted":true}

6. 预算档位与对照组
预算:3 档 —— loose(基本够 update-both)/ medium(只能验证部分候选)/ tight(只够修一个组件或少数候选)。
预算维度:validation call cap(主)+ latency cap + context-token cap。预算必须是真 binding 的(超了就截断/失败),不是表格里多一列 cost。
对照组(8 个,都跑,× 3 档预算 × 多 seed):
no-update / router-only / validator-only / uniform-split / random-allocator / update-both / oracle-allocator(规则3 的 empirical oracle) / ours(规则1 的 history-based allocator)。

7. 两个内置 ablation(Gate-1 第一版就要有,这是不重蹈 CARE 的保险)
Ablation A — 收益估计 ablation: 把 ours 的 r̂ 换成随机值和常数值,重跑。
判据:若 ours 相对随机/常数 r̂ 没有显著优势 → 停。 说明 allocator 没在做真正的预算仲裁(真正起作用的是别的东西)。
Ablation B — 候选顺序 ablation: 打乱候选 patch 的枚举顺序,重跑。
判据:若效果依赖固定枚举顺序 → 停。 这是上次"hit@N 正好是枚举顺序"那类假象的探针。

8. Gate 主图与通过判据(跑前写死,事后不改)
主图: x 轴 = budget tightness(loose→tight),y 轴 = net success under budget(本地组件级),8 条对照曲线。
Gate 通过(全满足):

tight 档下 ours 显著优于 uniform / random / single-component(CI 不重叠)。
update-both 在 loose 可用,但 tight 档因超预算或低净收益输掉。
最优分配随预算从 loose→tight 发生切换(证明"治理"是真变量,不是装饰)。
两个 ablation 都通过(换随机/常数 r̂ 效果显著下降;不依赖枚举顺序)。
ours 与 oracle 的 gap 合理(oracle 是实测上界)。

Gate 不过(任一成立即停,不扩规模):

tight 档下 ours 相对 uniform 无显著优势。
Ablation A:换随机/常数 r̂ 效果不掉(allocator 没在真仲裁)。
Ablation B:效果依赖枚举顺序。
两个组件信号有一端恒定(验证器半盲)。


9. 纪律(全程强制)

每个数字标 PROBE signal (n=X) 或 PUBLICATION conclusion;n<30 或单 seed 一律 PROBE;趋势结论需 ≥3 档预算点 + ≥多 seed,单点不当趋势。
所有 patch 可版本化、可回滚、有成本、能在 validation suite 上测、绝不接触 held-out test labels、必须能影响未来一类 episode(禁止 instance-level leakage:不允许"对当前这一题追加一句提示"这种只修当前样本的假 patch)。
不重写整个 prompt、不重写 agent policy、不塞大段反思总结、不重设计整套 workflow(那是别人的战场且无法归因)。一次 update 必须小到可归因(atomic patch)。
不中途改预算定义、不中途改 gate 判据、不中途改预注册阈值。
遇到"好得可疑"的结果(某信号恒定、某档完美),先做对应 ablation 自查,确认后再写进结论。


10. 执行顺序总结(照这个走)

STEP 0 数据探针 → 输出 relevant APIs coverage → high 用 StableToolBench / low 切 BFCL。先贴 coverage 结果。
写两组件对称本地信号(§3)→ 各跑几条自检两端非平凡。
实现四条反捷径硬规则(§4)。
STEP 1–5 实验管线(§5)。
3 档预算 × 8 对照 × 多 seed(§6)。
两个内置 ablation(§7)。
按预注册判据出 Gate 主图与裁决(§8)。

先做 STEP 0,把 relevant APIs 的 coverage 数字贴出来,我据此确认走 StableToolBench 还是 BFCL,再继续后面。不要一口气跑到底,STEP 0 是 go/no-go 闸。