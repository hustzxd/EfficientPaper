# Together Compiler: Continual Compilation for AI Inference

> Reyna Abhyankar, Dylan Lim, Drew Wadsworth, Venkata Hasith Vattikuti, YuTong Julia Zhang, Jung Hwan Heo, Taeksang Kim, Kesavan Ramakrishnan, Evangelene Stanley, Xinyu Fang, Daniel Y. Fu, Simran Arora

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

> Together AI Research，2026-11-04。[完整技术报告](https://www.together.ai/together-compiler-paper) · [官方发布文章](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference)。已阅读 38 页正文与附录；当前未检索到对应 arXiv 条目或独立公开实现，未复现性能实验。

## Abstract

Today's AI workloads are rapidly changing: new models are released more often, model architectures are increasingly diverse, and traffic distributions change frequently. Together Compiler (TKCC) continually optimizes inference forward passes with parallel coding agents. A Tiered Persistent Engine separates model loading from candidate execution and balances fast experimentation with high-fidelity measurements; a parallelism cost model selects promising initial configurations before agent optimization. The report also studies agent communication, expert-curated context, and source-level optimization. It reports up to 1.7x higher TPS than hand-written megakernels on Llama 1B, up to 2.6x higher TPS than selected InferenceX baselines, and acceleration beyond LLM inference. These results are author-reported under the protocols described below.

## 一句话总结

TKCC 用可热加载 forward pass 的两档持久推理引擎和并行 Agent，提高“提出优化—测量—校验”的试验速度，再通过结合实际流量的并行配置成本模型，为随模型、硬件和请求分布变化的推理部署持续寻找优化。

## 创新点

1. **分层持久引擎：用真实权重/流量快速试候选。**引擎只初始化一次，Agent 提交新 forward 源码后，各 rank 替换函数指针、重新 capture CUDA/HIP graph，测完恢复 baseline 并卸载候选模块。快档只加载少数层，过滤慢、错误或挂起候选；慢档加载更多层检查真实数据分布下收益。报告表 1 示例分别为 12/60 层，job 中位耗时 28/64 秒，热加载 17/45 秒。固定请求的 prefix cache 和 kernel binary 也可复用。拆开 prefill/decode 测量使连续运行变异从 6.6% 降到 2.7%。
2. **并行 Agent 共享测量服务，缩短 GPU 空闲。**多个 Agent 思考和写代码的同时，其他 Agent 可占用 GPU 测候选；Agent 看到测量队列深度和完成时间，并通过 message board 交流，减少重复工作。对比反复重启整个服务的 RSI-Inference Bench，作者报告实验吞吐 252x、每 Agent 空闲时间降低 11x；两档测量相对单慢档测试 1,354 vs. 168 个候选，并找到快 7.2% 的方案。这些是优化过程的效率，不能当作推理 TPS 提升。
3. **先选并行布局，再优化实现。**成本模型枚举 attention TP/DP/CP、MoE TP/EP 等布局，以权重容量、维度整除和 TTFT/TPS SLA 剪枝；逐层计算 compute/memory 较大者加 collective 时间，再由 HBM 与速度约束得到可行并发和 chip-seconds/request。硬件规格、实测 microbench 和完整 forward 作为不同精度的信息源；实际流量中的专家不均衡、MTP 接受长度和请求长度分布决定 straggler，避免把尚未优化的软件布局误判为硬件无潜力。附录独立建模 sparse KV 的存储容量与每步读取量，二者不能混为一谈。
4. **直接改源代码，少限定搜索策略。**允许 Agent 修改 forward、kernel 和 host-side metadata；相比自定义 IR、专家知识图和固定策略分工，较简单提示在部分实验中得到更高收益。有效改动包括 collective 与 residual/norm/quantization 融合、去除多余 copy/memset、消除 attention padding、调整供应商 kernel 参数；完整 megakernel 在 GPU 上常无收益，在 Trainium2 上却有效，说明实现策略随硬件变化。

## 带来什么提升

1. **小模型单请求：**Llama-3.2-1B-Instruct、batch 1、ISL 32、H100/B200、OSL 256–65,536 下，相对手工 megakernel 等基线最高 **1.7x TPS**；B200 短输出点达到约 **2,118 output tokens/s**。PDL、CUDA graph、L2 prefetch 等普通 driver/runtime 原语可恢复部分原先依靠 persistent kernel 获取的收益（图 6）。
2. **前沿模型/公开引擎：**每设置优化不超过 24 小时、使用两档引擎（优化资源为 serving replica 的 2 倍），对比截至 2026-10-27 的最佳 InferenceX 配置，报告最高 **2.6x TPS**。NVIDIA/AMD 使用 AgentX 393 条 coding-agent trace；图 7 固定 baseline 并行布局以隔离代码优化。图 8 则允许成本模型改变布局，batch 1 的 MiniMax M3.1-Flash/B200 达 **1,111**、DeepSeek v4.1-Flash/GB300 达 **1,205**，这里是 $1/\mathrm{P90\ intertoken\ latency}$ 的 per-user TPS，不能与总吞吐混用。
3. **超出 LLM 的 forward 优化：**8 个天气、生物、时间序列、PDE workload 相对各自指定 baseline 为 **1.5–6.4x**；Aurora 3.7x、Chronos-2 2.6x、Poseidon-T 6.4x。表 5 在列出的输入范围内验证 bit-exact 输出，随机模型还对齐 RNG 状态。相同墙钟预算下增加天气 ensemble samples 可改善 RMSE/CRPS，属于更快 forward 支持更多采样的间接收益（表 2、3、5）。

## 备注

- **数据分布会改变 kernel 排名。**随机数据使活跃 experts 多 13%–38%；一个随机输入下约 20µs 的 histogram top-k 在真实 ReLU score 大量零值时触发平方复杂度，70K context 达 62ms，并使引擎端到端慢 1.3x。候选必须在真实 activation、routing、KV cache 和端到端流程中复测，不能只信独立随机 microbench。
- **比较口径：**AgentX 的 throughput/GPU 把 cached input tokens 全额计入，与 output TPS 不同。Trainium2 用独立 synthetic multi-turn 流量（65,792-token context、约 95% cache hit、OSL 700），不是同一 AgentX workload。跨领域 Claude Code 基线虽给了评测要求，却没有 TKCC 的持久引擎、adapter 和 correctness tests，比较包含基础设施准备成本。静态/共享请求的测量不能直接证明任何流量漂移下都有持续收益；Kimi K3 的一项优化在 concurrency 22–78 好 2.8%–5%，在 concurrency 8 却慢 6.6%。
- **消融与正确性边界：**专家知识库/固定策略有时降低搜索效果，是特定模型、预算与知识库组织下的结果，不能推广为知识或结构化记录普遍无用。报告也保留 agent 自己写 Markdown 记录及频繁交流；反对的是限制优化表达和高开销的人工规定图结构。Bit-exact 验证只覆盖指定测试，不能等同形式化证明；作者记录过非法访问导致 GPU reset、两台机器移出集群。生产 trillions-of-tokens 是作者部署陈述，未做本地独立验证。
- **系统启发：**“持续”主要是固定预算内反复搜索并随流量变化重新编译，可按 batch 条件选用不同优化。仍有超过 50% Agent 时间在等测量 gate/排队。成本模型假设层内 compute 与 memory 完全重叠、collective 顺序相加，需按真实 engine 校准。对 KV/serving 研究，可借鉴持久服务与分档测量减少 policy 试验成本，但必须隔离候选的 KV/RNG/graph 状态，冻结独立评估器，并将编译时间、额外 GPU、上下文命中和真实请求分布纳入收益核算。报告的 Mirage 基线实际引用 2026 MPK，库中 `2025/MPK-Mirage` 的标题却是另一篇 Mirage superoptimizer；本次不将二者强行关联。
