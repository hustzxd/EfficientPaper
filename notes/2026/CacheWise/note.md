# CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents

> Shubham Tiwari, Tapan Chugh, Nash Rickert, Simon Peter, Ratul Mahajan, Haiying Shen

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> 版本与会议信息：本文依据 [arXiv:2606.16824v1](https://arxiv.org/abs/2606.16824v1)（2026-06-15，14 页）。[MLSys 2026 官方条目](https://mlsys.org/virtual/2026/poster/10200) 使用 **ForeCache** 名称，副标题和六位作者均相同；官方数据将其列入 **YPS Poster Session & YPS Reception**，不是已核实的主会 proceedings 论文。

## Abstract

Coding agents are a fast-growing LLM application, executing as long-running closed-loop sessions in which LLM generations alternate with external tool calls. Yet, unlike chat workloads, their serving behavior has not been studied extensively. We address this gap by collecting a dataset of real-world coding assistant traces. Our analysis shows that coding agent sessions repeatedly reuse large prefixes and create sustained KVCache pressure that conventional LLM serving policies handle poorly.
  Based on our analysis, we present CacheWise, a KVCache management layer that improves KVCache reuse for coding agent workloads. CacheWise combines prefix-aware scheduling with reuse-aware eviction guided by lightweight predictions from tool call metadata. Implemented in vLLM and evaluated on the collected traces, CacheWise reduces KVCache evictions by up to 2-2.6x and improves total agent session completion time by up to 3.5x.

## 一句话总结

CacheWise 在 vLLM 中联合使用“优先调度新增 KV 块最少的请求”和“根据工具名称、参数及已执行时间预测最晚复用的会话并淘汰其闲置 KV”，减少真实 coding-agent 长会话的缓存颠簸，目标是会话推进效率而不只是单次请求延迟。

## 创新点

1. **用真实 trace 明确 coding-agent serving 的差异。** CATraces 来自实验室内知情同意用户的 Claude Code 使用记录，约 10M tokens。相对 chatbot，prefill/decode token 比约高 21 倍，工具触发请求与用户触发请求之比的中位数约为 20；会话时长中位数约 36 分钟。长前缀不断增长且在工具执行后复用，使 FCFS + LRU 容易反复驱逐、重算同一会话状态。
2. **Prefix-aware admission 抑制 working-set 扩张。** 对请求 $i$，令所需 KV 块为 $d_i$、已驻留块为 $k_i(t)$，优先调度新增块数 $a_i(t)=d_i-k_i(t)$ 最小者。准确规则是最小化新增块数，不只是最大化命中比例。该思路并非首次提出；本文的价值是量化它在 coding-agent 闭环会话中的作用，并与预测淘汰联用。
3. **预测复用排序，而非追求精确工具耗时。** 从服务端可解析的 `tool_name`、`tool_args`、工具启动时间提取信号。工具参数用 TF-IDF（最多 5000 维词表）与 KMeans 聚类，维护组内历史耗时分布；已执行 $e$ 时间后，以 $\mathbb{E}[D-e\mid D>e,m]$ 表示给定元数据 $m$ 的条件剩余时间，优先淘汰预计最晚复用的会话。只看工具名或会话内移动均值，难以区分 `bash ls` 与 `bash pytest`。
4. **不修改 agent 的运行时集成。** 在 vLLM 中增加约 2500 行 Python；仅把引用计数为零的块放入淘汰堆，同会话块共享预测。每 3 个 engine iterations 更新预测并重建堆，平衡估计陈旧和 CPU 开销。本文主要管理单个 inference node 的跨轮次 KV，而非提出新的跨节点路由或多层存储系统。

## 带来什么提升

1. **重算场景的整体收益。** 在 2×H200、TP=2、Qwen2.5-Coder-32B-Instruct、512-token chunked prefill 的确定性 replay 中，80% 会话用于训练、其余用于测试。并发会话数 $N>10$ 时，相对 FCFS 的 vLLM/InferCept，token goodput 约提升 **1.64-2×**，请求吞吐约提升 **1.5-2×**，KV 块淘汰次数降至约 **1/2-1/2.6**，会话指标耗时降至约 **1/2.7-1/3.5**；低负载下差异较小。
2. **预测淘汰在相同调度下仍有增益。** 与已加入 prefix-aware scheduling 的 vLLM/InferCept 比较，goodput 进一步提升约 **1.2-1.6×**，会话指标耗时降至约 **1/1.7-1/2**。这比仅与 FCFS + LRU 比较更能隔离预测器的价值；论文未直接运行完整 SGLang 基线。
3. **真实 offload 收益明显更小。** Figure 17 的 TB 数据量由淘汰次数乘块大小折算，并非全部为实测传输。Figure 18 才实际执行 GPU↔CPU DRAM swap；在 $N=30$ 时，累计 offload latency 改善约 **1.40×**，会话指标耗时改善约 **1.19×**。不能把最高 3.5× 重算收益直接套到 LMCache/DRAM/SSD 层次化缓存。
4. **收益伴随公平性和控制开销代价。** P50 请求耗时约降低 13-14 倍，但新请求缺少驻留前缀，会被推迟，**P99 请求延迟反而变差**。CPU 调度开销约为 vLLM 的 2.4-3 倍；论文在 $N=40$ 下报告 0.33→0.99 秒，而 GPU 执行时间从 16.6→11.2 秒，净收益来自减少重算，并非预测决策没有开销。

## 备注

- **指标口径：** Section 6.1 的 session completion time 是同一会话内 LLM 请求延迟之和，**不含工具执行耗时**；replay 还把人类空闲期当作会话边界。它不是包含工具与用户等待的真实任务墙钟完成时间，不能由此推导真实开发任务也快 3.5 倍。
- **泛化边界：** 单一实验室真实用户 trace、一个 32B 模型和单节点双 GPU；按会话随机划分 80/20，不等于跨用户、跨项目或跨时间泛化已验证。分布漂移、冷启动预测、混合交互式请求和饥饿保护仍需额外实验。ground-truth 工具时长版本也不是联合调度、淘汰和分层选择的全局最优 oracle。
- **公式阅读提醒：** Section 4 的 $W(t)$ 先按完整需求 $\sum_i d_i$ 定义，Figure 10 则按实际驻留量 $\sum_i k_i(t)$ 表示，符号不一致。实现容量检查应使用实际已用容量与空闲块数，驱逐量为新增需求超出可用空间的部分，不能直接照抄“容量不足就驱逐 $a_i$ 块”。
- **对 cost-aware KV lifecycle 的启发：** 工具条件剩余时间可以作为 reuse signal，但策略还需比较恢复/重算代价、tier 带宽和排队状态，并约束新会话等待与 session tail latency。最小消融应固定 prefix-aware 调度，只替换淘汰策略；再分别测试 recompute 与真实 DRAM/SSD offload。
- **公开材料：** [作者仓库](https://github.com/cachewise-project/cachewise-coding-traces) 的 README 提供脱敏 trace、事件提取、workload 分析和工具时长预测命令；本次未确认其中包含完整 vLLM serving 改动。实际比较的 InferCept 尚未在本库收录，因此未写入 baseline 图引用。
