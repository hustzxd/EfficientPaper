# Harness^2: Recursive Agent Harnessing for an Open World

> Ruiyao Xu, Yanfei Chen, Zhongying CuiZhu, Bhavana Dalvi Mishra, Yifei Ming, Han Yu, Rujun Han, Chen-Yu Lee, Tomas Pfister

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> 来源：[arXiv:2610.14224v1](https://arxiv.org/abs/2610.14224v1)，2026-10-09 提交，37 页；[官方代码](https://github.com/google-research/harness2) / [项目页](https://harness2.github.io)。另核对了官方仓库的实验说明。

## Abstract

LLM agents' capabilities depend not only on their underlying models but also on their execution harnesses. Recursive self-improvement (RSI) offers a promising approach to automatically evolve these harnesses. However, deploying RSI in open-world settings still faces two critical challenges. First, open-world tasks are typically unseen during the harness design phase and novel at test time, leaving no ground-truth verifiers to reliably guide online harness edits. Second, the RSI process remains highly task-specific and fails to generalize to novel scenarios. We introduce Harness^2, a plug-and-play, recursive agent harnessing framework that addresses these challenges through two connected levels of recursion. Task-level recursion refines the current task's harness using a contrastive Propose-Probe-Compose step that reliably guides edits without ground-truth verifiers, and can be scaled further via parallel or sequential recursion. Harnessing-level recursion then extracts experience from these edits to update a continual improvement harness, enabling the agent to better adapt to future tasks. Across three professional benchmarks and four model-harness configurations, a single step consistently improves GLM-5 and Gemini 3.5 Flash over their base harnesses, yielding gains of up to 11.5 points on WorkBuddy-Bench, 4.5 points on JobBench, and 11.8 percentage points on LAB, with further gains as the task-level recursion budget increases.

## 一句话总结

Harness² 在冻结模型权重的前提下，先用同任务执行对照完成 Propose–Probe–Compose 的 harness 修订，再将被采用的可执行组件及修订经验存入库，改善未来任务的修订过程，把“优化当前执行配置”和“积累如何优化配置的经验”连接起来。

## 创新点

1. **区分 execution harness 与 improvement harness。** 前者有八类可编辑文件：system prompt、guardrails、memory、tool guidance、subagents、skills、scripts、plugin。后者通过索引库保存已采用的完整 harness 和 `EXPERIENCE.md`，记录适用情境、修订原因和证据。每个新任务从公共初始配置 $H_0$ 开始，选择性复用历史组件；不直接继承上一任务的全部专用配置。模型权重、阶段指令和决策规则固定，跨任务改变的是改进库。
2. **无标准答案的 Propose–Probe–Compose。** 先运行参考配置；两个 editor 分别定位跨领域的执行问题和当前领域问题，提出 invariant/domain 候选；在相同任务、模型及环境配置下 probe；composer 根据轨迹、交付物和文件差异合并、修订或拒绝更改。执行对照用于判断行为变化，不能充当正确性 verifier；没有足够证据时可保留原配置或不新增经验。
3. **同时扩展宽度与深度。** Parallel 在预算 $k$ 下使用 $k$ 次参考执行、$k$ 个 invariant 候选、$k$ 个 domain 候选和 $k$ 个 composition；实现是一次 population update，$k$ 控制宽度。Sequential 连续执行 $k$ 个修订步骤并携带历史。$k>1$ 时由看不到配置来源的 selector 比较匿名交付物，选择已有输出；selector 不生成新内容，也不接触评分。$k=1$ 不走最终选择。
4. **保留“可执行组件 + 为什么有用”。** 经验库不是只检索一段反思文本，还可复制和适配 skill/script 等组件；但旧任务事实必须在新任务重新验证。论文以去掉经验库、去掉编辑分工、去掉 composition 的消融及跨 benchmark 迁移，检验跨任务积累的作用。

## 带来什么提升

1. **单步收益跨配置成立。** 在 Gemini 3.5 Flash、GLM-5 × OpenCode、Codex 四种组合及三个 benchmark 上，$k=1$ 均优于各自初始配置。WorkBuddy 三领域平均分最高提升 **11.5 分**（GLM-5/OpenCode：64.2→75.7）；JobBench 最高 **4.5 分**（Gemini/OpenCode：31.5→36.0）；LAB criterion-level Partial pass rate 最高 **11.8 个百分点**（GLM-5/OpenCode：59.6%→71.4%）。这些是绝对分差，不是相对百分比，也不是所有任务全部通过的比例。
2. **增加递归预算继续提高质量。** 从单步到 parallel $k=5$，WorkBuddy 平均分进一步增加 **2.9-3.9 分**，JobBench 增加 **3.0-4.2 分**，LAB Partial 增加 **2.6-6.7 个百分点**。Gemini/OpenCode 的 WorkBuddy 平均分为 72.4→77.5→80.8。Figure 6 统计最终交付前执行和改进的总 tokens，在所比较配置内处于经验 Pareto frontier；它不证明墙钟时间、美元费用或任意更大预算都更优。
3. **经验库、分工和合成各有独立贡献。** Gemini/OpenCode、$k=1$ 下，去掉库使 WorkBuddy/JobBench 分别下降 **3.1/1.8 分**，去掉 scope 分工下降 **2.9/3.7 分**，去掉 composition 下降 **1.6/1.9 分**。JobBench 经验库迁移到 WorkBuddy 后，Code/Web/Office 进一步增加 **0.6/0.5/2.9 分**，支持有限范围的跨任务迁移。
4. **收益不只依赖更强 editor。** 主实验 editor/selector 使用 Claude Opus 4.8；另在 OpenCode、$k=2$ 下让同一模型兼任 executor/editor，五个模型均在三个 benchmark 上改善。工具错误率也下降，例如 JobBench **14.4%→9.1%**，但工具调用数 **27.4→32.8**；更可靠不意味着调用更少。

## 备注

- **评测范围：** WorkBuddy 测试任务为 Code/Web/Office 各 40/35/25 个，JobBench 34 个，LAB 100 个；主表是三次随机任务顺序运行的均值和标准差。库在每个 benchmark stream 开始为空，按领域维护；同一任务可多次执行，但在线流程不重跑已完成任务。`#Steps` 只统计 executor session 的模型轮数，不能代表全部 probe/editor/selector 成本。
- **无标签不等于正确性保证。** 论文披露 selector 可能选中 55 分输出而舍弃 95 分输出，也可能让错误前提在多个候选间传播。Leave-one-out 重跑能读到后续任务经验，是离线复用分析；其 WorkBuddy +1.9、JobBench +1.7 分不能当作在线收益。双层循环是有限预算实验，不证明无限递归自我改进或持续增长。
- **官方代码的评测隔离风险：** [EXPERIMENTS.md 的 Web access 节](https://github.com/google-research/harness2/blob/main/docs/EXPERIMENTS.md#web-access) 说明 release 默认不限制联网，并记录一次验证运行中 Codex 自行下载了 JobBench 的 `RUBRICS.json`；建议在 setup 后于网络层阻断 benchmark 数据/评分来源。论文描述了工作目录评分隔离，但这与网络隔离不同。该记录是公开实现的复现风险，尚不能据此判定原论文实验已经发生泄漏。
- **后续研究空间：** 经验库当前是简单索引，缺少系统化合并、过时经验修订和检索治理；递归形式及预算预先指定。可研究基于证据不确定性与迁移价值的停止策略，在总 token、时延和费用预算下分配 probe，并用独立评估检验经验入库后的真实增益。ReasoningBank、Meta-Harness、Harness Scaling 三个实际比较基线目前未在本库收录，暂不建立不存在的图引用。
