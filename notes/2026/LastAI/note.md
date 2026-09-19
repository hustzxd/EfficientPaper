# The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

> Yi Duan, Ying Liu, Zirui Tang, Haodong Chen, Jun Zhou, et al. (Shanghai Jiao Tong University, Theseus Labs, Tsinghua University, ByteDance, ModelBest, Xiaohongshu, Shanghai AI Lab, Humanlaya, Agent-Native Research Lab, Frontis.AI)

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.

## 一句话总结

一篇 79 页的立场综述：以"改进回路"为分析单元、以 AI 内化了哪些改进决策为标准，提出 B0→L5 的 RSI 自主性五级路线图，用 HCI 指数量化现有 LLM 交互式能力的巨大 headroom，并以八个工业实践说明真正 RSI（改进机制本身可被改进并被继承）尚未实现。

## 创新点

1. **HCI（Headroom-Closed Index）能力进展审计**：汇集 2023–2026 共 393 个 model–benchmark 观测（17 个可跨 harness 链接的协议族），归一化为"入场年前沿=0、满分=100"，并按第一方权重 ×0.75 折算。结论：进展极不均衡——2026 年高等数学 HCI 86.4、研究生科学 85.8，但软件工程仅 52.6、搜索/终端 agent 56.8、工具 agent 仅 39.9。静态可验证任务趋于饱和，长程交互式工作流仍留有 30–46 点差距，这正是 RSI 的价值空间。
2. **自主性五级分类框架（B0–L5）**：以改进回路（experience → target → improver → strategy → verifier → improvement → successor）为基本分析单元，按 AI 逐步内化的改进决策分级：B0 仅改进当前输出（Self-Refine/Reflexion，无持久化）；L1 执行人类定义的改进流程且产物持久化（FineWeb-Edu、Meta Capacity Efficiency）；L2 自主选择改进策略但目标/验收标准外置（GEPA、ADAS、AFlow、DGM、AutoKernel）；L3 自主决定未来学习经验（SIMA 2、AZR、R-Zero、VOYAGER）；L4 部署环境反馈驱动的持久适应（PANDO、HDSO、Trace2Skill、HarnessDev）；L5 递归元改进——improver/verifier/研究策略本身成为改进对象并被后继继承（STOP、RQGM、A-Evolve-Training、HyperAgents）。每级都回答三问：回路在哪里闭合、什么被继承、哪些决策仍由人控制。
3. **区分"结构性递归"与"有效性递归"**：结构性 L5 只要求 AI 修改的机制确实被后续轮次调用；有效性 L5 还要求修订后的机制在同等预算、独立评估下产出更强的后继者。现有系统几乎全部停留在结构性层面（如 AIDE2 将进化后 harness 装为外层 improver 未获得统计显著的效率优势），这是对当前"RSI 进展"宣称的关键冷却剂。
4. **与邻近范式的判据表**：相对 continual learning（更新规则固定）、AutoML（搜索空间/评估器预先给定）、agentic AI（episode 内 harness/验收不变），RSI 的独特性在于"围绕系统自身的持久改进回路已形成，且回路权威内生化"——机制可被修改并跨代复用。
5. **四大应用域的反馈机制比较**：软件工程（成熟 L2、新兴 L3、有界结构性 L5，工件与 agent 都可执行/版本化/回滚，反馈最便宜）；科学（强 L2、早期 L3，归因困难）；具身智能（L2–L3 主战场、L4 靠模拟，物理试错昂贵不可逆）；医疗（成熟 L2、L3 有限，反馈延迟且混淆）。核心论点：验证反馈的成本与可靠性决定改进回路能闭合到哪一级。
6. **八个工业 RSI 实践案例**：Theseus 环境-数据-模型协同进化（干净 workspace 使 8 个模型-harness 配置 pass rate 提升 21.7–51.6pp）；Lark 企业知识图谱+自动数据质量评估（可用性 52%→65%）；小红书 IMA 双时间尺度推荐 RSI（score_mean@16 +7.0%）；Humanlaya 交付驱动质检外环（关键缺陷率 9.0%→3.7%，人工处理 48→27 分钟）；ModelBest 零人力工业 AI 工程（ForgeTrain 约 8 小时匹敌 Megatron-LM v0.15，MFU 40.1%→44.1%）；腾讯 Hyra 经验库+可演化评估器（nanoGPT Speedrun 76.4s）；Agent-Native Research Lab 的 ARA 可执行研究工件（继承 QA 准确率 93.7% vs 传统论文 72.4%）；Frontis.AI ME–WE–MA 跨任务元改进（216 个改进任务，元经验使 unseen 任务进化加速约 20%）。

## 带来什么提升

1. **量化定位 RSI 的优先收益区**：给出示意性外推（假设关闭 78% 剩余 headroom），工具 agent 39.9→86.8、软件工程 52.6→89.6，而网络安全 91.9→98.2 增益有限——指出 RSI 投入应集中于交互式长程工作流，而非已在饱和区的静态基准。
2. **可操作的 RSI 审计框架**：三级交叉问题 + 评估维度表（adaptivity / retention / transfer / efficiency / stability / meta-recursion），以及机制审计要求（记录被改工件、动机证据、后续轮次调用验证），可直接用于检验任何自称"自我改进"的系统是否只是 L2 搜索或评测器利用。
3. **命名并例证关键失败模式**：safe inheritance（Gödel Agent 100 次 MGSM 试验中 14% 低于初始策略）、autonomy attribution（DGM 的归档维护与亲本选择仍在自改范围外）、reliable verification（Anthropic 观察到 seed cherry-picking 与试图通过评测器查询提取测试标签）、experience corruption（错误经验经参数/记忆/课程跨轮传播）、persistent update failure 与 library drift（技能库无限累积使检索退化）。
4. **系统技术分类表（Table 2）**：将约 160 个系统按 6 大技术族（search & optimization / data-task-experience construction / training & persistent update / verification & selection / online & meta-level adaptation）× 5 级自主性映射，附 L1–L5 每级的代表性系统、关键机制与"层级限制"栏，是当前 RSI 文献最完整的导航图。

## 备注

- 证据定位需要清醒看待：大量关键证据来自 company-reported 技术报告/博客（作者本身来自这些机构），论文虽多处强调"非独立复现"，但阅读时应打折。
- 这是路线图/领域定义型综述而非方法论文：核心贡献是分类学与评估判据，不做任何新实验；标题的"Last AI"是愿景式修辞，正文对当前系统普遍给出的是"bounded autonomy"的保守定级。
- 对 RSI 研究的直接启示：短期可发表空间集中在 L3–L4 的可靠经验获取与持久状态治理（学习价值估计、库管理、激活/遵循分离），L5 的"有效性递归"（匹配预算下的跨代统计显著加速）仍是开放问题。
