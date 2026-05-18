# Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference

> Sanjeev Rao Ganjihal

![111](../../blank.jpg)

## Abstract

Key-value (KV) cache memory management is the primary bottleneck limiting throughput and cost-efficiency in large-scale GPU inference serving. Current systems suffer from three compounding inefficiencies: (1) the absence of unified KV cache sizing across all attention architectures--particularly multi-head latent attention (MLA), which is unsupported in general-purpose frameworks, resulting in up to 57x memory over-provisioning; (2) confinement of KV cache to a single memory tier (GPU HBM) despite the availability of a rich hierarchy spanning CPU DRAM, CXL-attached memory, NVMe via GPUDirect Storage, RDMA fabric, and parallel filesystems; and (3) reactive eviction policies that discard reusable state, forcing redundant recomputation. We present a unified system that addresses all three problems. Our architecture-variant-aware sizing engine computes exact memory requirements per attention type, enabling up to 7.4x higher batch sizes. A six-tier memory hierarchy extends effective KV cache capacity from 40 GB to over 38 TB per node while maintaining sub-millisecond time-to-first-token (TTFT) for hot entries. A Bayesian reuse predictor with Beta conjugate priors over 16 (block-type, transition-type) pairs achieves 70-84% cache hit rates, combined with EMA-scored head-granular eviction and RoPE-aware prefetching. Component-level validation on trace replay using ShareGPT, LMSYS-Chat-1M, and agentic workloads demonstrates 70-84% cache hit rates. Analytical projections combining validated component behavior with published hardware specifications indicate 1.4-2.1x projected TTFT reduction, 1.7-2.9x throughput improvement, and 47% cost reduction compared to state-of-the-art baselines.


---

*以下总结由 MiMo 生成：*

这篇论文针对大规模GPU推理中KV缓存内存管理的瓶颈问题，提出了一种统一的多层内存管理系统。该系统通过架构感知的内存需求计算、六层内存层次扩展以及基于贝叶斯的重用预测器，解决了KV缓存尺寸不统一、内存层级利用不足和被动淘汰策略低效三大问题。实验表明，该系统能实现高达7.4倍的批量大小提升、70-84%的缓存命中率，并在投影中实现1.4-2.1倍的首词生成时间减少、1.7-2.9倍的吞吐量提升和47%的成本降低。

---

## 论文详细总结

> 由 GPT 自动生成，请人工核验。

### 1. 研究背景与动机

大规模 GPU LLM 推理中，KV cache 已成为吞吐和成本效率的核心瓶颈。论文指出，70B 模型在 128K context 下，单请求 KV cache 可能超过 40GB HBM，显著限制 batch size。作者总结当前系统的三个问题：

- **跨 attention 架构 sizing 不统一**：MHA/GQA/MQA/MLA 的 KV cache 形态不同，但通用 serving 框架缺少统一 sizing engine；MLA 若按 MHA 等价方式估算，会产生最高 **57×** 内存高估。
- **单层 HBM confinement**：vLLM/SGLang/TensorRT-LLM 等主要系统通常将 KV cache 限制在 GPU HBM，未系统利用 CPU DRAM、CXL、NVMe、RDMA、parallel filesystem 等数据中心内存层级。
- **Reactive eviction**：GPU 内存满后用 LRU/random 等被动策略丢弃 cache，忽略 system prompt、tool context、agent workflow、RoPE sequential locality 等可预测复用模式。

### 2. 核心思想

论文提出一个 **predictive multi-tier KV cache management system**：先用 architecture-aware sizing 精确计算不同 attention 架构的 KV memory budget，再把 KV cache 放入六级内存层级，并用 Bayesian reuse predictor 主动决定 promotion/demotion/eviction/prefetch。

一句话概括：**把 KV cache 从“HBM 内的被动 LRU 缓存”扩展为“跨 HBM/DRAM/CXL/NVMe/RDMA/FS 的预测式分层记忆系统”。**

### 3. 关键技术

| 技术 | 说明 |
|------|------|
| **Architecture-aware KV sizing** | 分别支持 MHA、GQA、MQA、MLA 的 KV cache 公式，避免在 MLA/GQA 模型上按 MHA 过度预留内存。 |
| **Six-tier memory hierarchy** | T0 GPU HBM、T1 CPU DRAM、T2 CXL 3.0 memory、T3 NVMe + GPUDirect Storage、T4 RDMA network pool、T5 parallel filesystem。 |
| **Bayesian reuse predictor** | 用 Beta conjugate priors 维护 16 个 `(block_type, transition_type)` 组合的复用概率，block 类型包括 system prompt、tool context、user context、intermediate reasoning 等。 |
| **Latency-aware placement** | 根据预测复用概率、重计算成本和各 tier 存储/访问成本计算 value score，异步执行 promotion/demotion，避免阻塞 decode path。 |
| **Head-granular eviction** | 使用 EMA 跟踪 attention head 级重要性，并结合 recency 与 positional distance decay，避免整块粗粒度 eviction。 |
| **RoPE-aware prefetching** | 利用 rotary position encoding 带来的顺序局部性，预测附近位置的 KV block 访问。 |

### 4. 实验与投影结果

| 指标 | 结果 |
|------|------|
| 论文状态 | arXiv preprint，文中注明 under review；部分实现细节因 patent examination 保留 |
| MLA sizing 示例 | DeepSeek-V3 每 token 每层实际 KV 约 **1,152 bytes**，MHA 等价估算 **65,536 bytes**，差 **57×** |
| Batch size 示例 | DeepSeek-V3 on 80GB H100，MLA-aware sizing 将 batch size 从 **15** 提升到 **104**，约 **7×** |
| 有效 KV 容量 | 单节点从 **40GB** HBM 扩展到超过 **38TB** multi-tier capacity |
| Cache hit rate | ShareGPT、LMSYS-Chat-1M、agentic workloads trace replay 上 **70–84%** |
| 投影 TTFT | **1.4–2.1×** projected TTFT reduction |
| 投影吞吐 | **1.7–2.9×** projected throughput improvement |
| 投影成本 | 约 **47%** projected cost reduction |
| 64-GPU H100 projection | 约 **4,150 tokens/s/GPU**，$0.43 / million tokens |

注意：论文多处强调结果是 **component-level validation + analytical projections**，不是完整生产系统的端到端实测。

### 5. 核心贡献

1. 将 KV cache management 统一到 MHA/GQA/MQA/MLA 多 attention 架构下，解决异构模型 fleet 的 sizing 问题。
2. 明确提出六级 memory hierarchy，把 CXL、NVMe GDS、RDMA、parallel filesystem 纳入 KV cache 管理。
3. 用轻量 Bayesian predictor 对跨请求 KV block 复用建模，重点覆盖 system prompt、tool context、agent workflow 等高复用模式。
4. 将 eviction 从 block-level reactive LRU 推进到 head-granular、RoPE-aware、prediction-driven policy。
5. 方向上与当前 long-context serving / KV offload / prefix sharing / agent cache 管理高度相关。

### 6. 局限性与风险

- **端到端验证不足**：核心收益主要来自 trace replay 与 analytical projection，完整 CXL/RDMA/NVMe 多层系统仍需 full-stack validation。
- **工程复杂度高**：跨 HBM、CPU、CXL、NVMe、RDMA、FS 的一致接口、异步调度、故障退化和 SLO 控制都很复杂。
- **预测粒度较粗**：16 个 `(block_type, transition_type)` 组合非常轻量，但可能无法捕获复杂 prompt/session 语义差异。
- **专利/实现细节保留**：论文说明部分 implementation details withheld pending patent examination，可复现性需要谨慎评估。

### 7. 对当前研究的启发

- 这篇可以作为 **Query/Agent-aware hierarchical KV cache** 方向的系统化参考：从 token eviction 扩展到跨请求、跨 tier、跨 workflow 的 KV memory system。
- 与 Cake/Tutti/CacheFlow 相比，它更强调 **architecture-aware sizing + Bayesian reuse prediction + CXL/RDMA tiering**，适合启发一个更完整的 KV cache scheduler 设计。
- 值得优先验证其中低成本部分：block type/transition type 复用预测、system prompt/tool context 跨请求共享、compute-vs-load value score，而不是一开始实现完整六级系统。
