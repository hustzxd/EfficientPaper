# MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

> Bowen Zhou, Jinrui Jia, Wenhao He, Yong Zhang, Fang Dong

![111](../../blank.jpg)

## 一句话总结

MixServe 是一个面向 MoE 模型的自动分布式推理服务系统，通过 TP-EP 混合并行策略和融合的 AR-A2A 通信算法，将节点内 AR 通信与节点间 A2A 通信重叠执行，从而在 DeepSeek-R1 和 Qwen3 模型上实现 TTFT 1.08×~3.80× 加速、ITL 1.03×~1.66× 加速以及 5.2%~50.3% 的吞吐量提升。

---

## 摘要翻译

混合专家模型（MoE）正成为大语言模型（LLM）的最新范式。然而，由于内存限制，拥有数十亿甚至万亿参数的 MoE 模型只能部署在多 GPU 甚至多节点多 GPU 的服务系统中。因此，通信已成为分布式服务系统中的主要瓶颈，尤其是节点间通信。当前的分布式 MoE 模型主要使用基于 all-reduce（AR）的张量并行（TP）和基于 all-to-all（A2A）的专家并行（EP）来实现。然而，TP 通常在节点间效率较低，因此局限于高速节点内带宽；而 EP 则容易出现负载不均衡问题，尤其是在并行度较高时。

本文提出 MixServe，一种面向 MoE 模型高效部署的新型自动分布式服务系统，基于融合的 AR-A2A 通信算法实现 TP-EP 混合并行。MixServe 首先评估各种并行策略的通信开销（考虑模型超参数和网络/硬件资源配置），然后自动选择最高效的并行策略。接着，提出基于融合 AR-A2A 通信算法的 TP-EP 混合并行，将节点内 AR 通信与节点间 A2A 通信重叠执行。在 DeepSeek-R1 和 Qwen3 模型上的大量实验表明，MixServe 实现了优异的推理性能：TTFT 加速 1.08×~3.80×，ITL 加速 1.03×~1.66×，吞吐量提升 5.2%~50.3%。

---

## 研究动机

1. **MoE 模型规模巨大**：如 DeepSeek-R1（671B 参数，256 路由专家）和 Qwen3（235B 参数，128 专家），只能在多 GPU/多节点环境中部署，通信成为性能瓶颈。
2. **现有并行策略的局限**：
   - **TP（张量并行）**：基于 AR 通信，节点内效率高但节点间效率低，受限于节点间带宽（如 InfiniBand/RoCE 带宽远低于 NVLink/HCCS）。
   - **EP（专家并行）**：基于 A2A 通信，节点间扩展性较好但负载不均衡，尤其在高并行度时。
3. **缺乏系统性理论分析**：现有并行策略主要基于经验直觉，缺乏对模型超参数、网络拓扑和硬件资源配置之间复杂交互的全面分析。
4. **未能有效利用带宽层次结构**：现有通信算子对所有通信一视同仁，未充分利用节点内/节点间带宽差异进行优化。

---

## 方法（技术细节）

### 系统概述

MixServe 采用两阶段架构：**离线阶段**和**在线阶段**。

- **离线阶段**：通过 Profiler 采集模型在不同 batch size 和 sequence length 下的 profiling 数据，结合网络和硬件资源配置（算力、节点内/节点间带宽和拓扑），通过自动分析器（Automatic Analyzer）计算理论值，输出最优并行策略。
- **在线阶段**：基于离线阶段的最优策略，通过权重加载器（Weight Loader）和分区器（Partitioner）加载和分割模型权重，并在模型前向方法中注入集合通信算子。

### 自动分析器（Automatic Analyzer）

1. **并行策略定义**：使用上下文无关文法（CFG）形式化定义单个 Decoder 层的并行策略，包括 TP、EP、DP、PP，每层的 Attention 块和 MoE 块可采用不同策略。

2. **通信算子分析**：
   - **AR**：分解为 Reduce Scatter (RS) 和 All Gather (AG)，通信量为 O(bs·h/d)，单轮完成。
   - **A2A**：通信量为 O((bs/d)·h·k)，需要 d-1 轮（Pairwise 算法）。

3. **DP 与 EP 权衡**：分析 dDP = dEP、dDP > dEP、dDP < dEP 三种情况的通信组结构和冗余。
   - dDP = dEP：所有设备参与 A2A，最平衡。
   - dDP > dEP：EP 组更小，产生专家权重冗余，增加内存开销但提升 DP 以提高吞吐量。
   - dDP < dEP：隐藏状态冗余，但可通过有效的丢弃策略降低通信开销。

4. **延迟建模**：
   - **计算延迟**：τ(dTP, dEP, dDP) ∝ Ψ/(dTP·dEP) · (b/dDP)·s·h
   - **通信延迟**：λ(dTP, dEP, dDP) = 2×AR(b/dDP·s·h, dTP) + 2×A2A(...)
   - **服务延迟**：Δtsvc = l[τ + λ] + (dPP-1)·P2P(b/dDP·s·h)
   - **排队延迟**：使用 M/M/1 队列模型，Wq = ρ/(μ(1-ρ))

5. **性能指标**：
   - **TTFT**：排队延迟 + 预填充服务延迟（s=Lin）
   - **ITL**：稳态解码服务延迟（s=1）
   - **吞吐量**：综合预填充和解码阶段

6. **约束条件**：NPU 内存限制，模型权重和 KV cache 总量 < M。

### 混合 TP-EP 分区器（Hybrid TP-EP Partitioner）

- Attention 块：节点内 TP + 节点间 DP
- MoE 块：节点内 TP + 节点间 EP
- 将 AR 拆分为 RS + AG，将 A2A 重组为 RS-A2A-AG 通信流程
- 分区器自动评估所有满足 nproc·nnode = dTP·dEP 的并行策略

### 融合 AR-A2A 通信算法（Fused AR-A2A Communication Algorithm）

核心创新：利用节点内/节点间通信的异步重叠机制。

1. **Fused RS-Combine 算法**（Algorithm 1）：
   - 步骤：(1) 节点内 RS，(2) 节点间 A2A，(3) 节点内 AG
   - RS 和 A2A 并发执行（异步），AG 在完成后执行
   - 时间复杂度 O(nnode)，空间复杂度 O(bsh·nproc)
   - 以空间换时间，通过额外临时存储实现通信重叠

2. **Fused AG-Dispatch 算法**（Algorithm 2）：
   - 类似地，节点内 AG 与节点间 Dispatch 通信重叠
   - 除第一轮 Pairwise 和最后一轮 AG 外，其余轮次均可重叠
   - 时间复杂度 O(nnode)，空间复杂度 O(1)

---

## 实验结果

### 实验设置

- **硬件**：
  - 2 台服务器，每台 8 块 Nvidia H20 GPU（96GB），节点内 NVLink 4.0（900 GB/s），节点间 InfiniBand（400 Gbps）
  - 4 台 Atlas 800T A2 服务器，每台 8 块 Ascend 910B NPU（64GB），节点内 HCCS（480 Gbps），节点间 RoCE（200 Gbps）
- **模型**：DeepSeek-R1（671B 参数，256 路由专家 + 1 共享专家）、Qwen3-235B-A22B（235B 参数，128 专家）
- **数据集**：ShareGPT-V3（1.2B tokens 人类对话）
- **基线**：vLLM（TP+PP / DP+EP）、Tutel（TP+EP）
- **请求速率**：2/4/8 req/s，最大 batch size 16，最大序列长度 4096

### 性能结果

| 指标 | 提升范围 | 详情 |
|------|---------|------|
| **TTFT 加速** | 1.08×~3.80× | Ascend 910B 上：DeepSeek-R1 2.67×（vs vLLM TP+PP）、1.70×（vs vLLM DP+EP）；Qwen3 3.80×（vs vLLM TP+PP）、1.32×~1.93×（vs vLLM DP+EP）；H20 上 1.08×~1.23× |
| **ITL 加速** | 1.03×~1.66× | Ascend 910B 上：DeepSeek-R1 1.42×（227.33ms→160.06ms）；Qwen3 1.66×（134.27ms→81.1ms）；H20 上 1.03×~1.16× |
| **吞吐量提升** | 5.2%~50.3% | Ascend 910B 上：DeepSeek-R1 22.0%（100.61→122.72 tokens/s）、Qwen3 32.2%（113.52→150.08 tokens/s）；H20 上 50.3%（DeepSeek-R1）、43.5%（Qwen3） |

### 消融实验

1. **DP 与 EP 权衡**：
   - Ascend 910B：dDP = dEP 最佳（如 Qwen3：383.14ms TTFT，150.08 tokens/s）
   - Nvidia H20：dDP < dEP 最佳（如 Qwen3：228.99ms TTFT，40.00 tokens/s）
   - 说明不同硬件环境下最优策略不同，MixServe 自动适应

2. **通信重叠的影响**：
   - 异步融合通信显著降低延迟，提升 TTFT、ITL 和吞吐量
   - 节点内/节点间通信重叠有效减少了总体通信开销

---

## 优势

1. **自动化策略选择**：基于理论分析自动选择最优并行策略，替代经验直觉。
2. **创新的通信重叠算法**：融合 AR-A2A 通信算法将节点内 AR 与节点间 A2A 通信异步重叠，有效利用带宽层次结构。
3. **通用性强**：在不同硬件平台（Nvidia H20、Ascend 910B）和不同模型（DeepSeek-R1、Qwen3）上均取得显著性能提升。
4. **理论驱动**：具有完整的理论分析框架（延迟建模、性能指标、约束条件），不仅有实验验证，还有理论预测。
5. **与现有系统兼容**：基于 vLLM 和 Tutel 实现，可与现有 LLM 服务系统的优化方法（请求调度、P/D 分离等）结合。
6. **空间换时间**：通过额外的临时存储实现通信重叠，空间复杂度可控。

---

## 局限

1. **仅评估两种硬件平台**：在 Nvidia H20 和 Ascend 910B 上测试，未涉及更多硬件配置（如 AMD GPU、其他 NPU）。
2. **仅评估两种模型**：仅在 DeepSeek-R1 和 Qwen3 上验证，未覆盖更多 MoE 模型架构（如 Mixtral 等）。
3. **仅关注推理服务**：未涉及训练场景，尽管其并行策略分析框架理论上可用于训练。
4. **额外内存开销**：融合 RS-Combine 算法需要额外的临时存储空间（O(bsh·nproc)），在内存受限场景下可能受限。
5. **未开源**：代码 URL 为空，无法直接复现和验证。
6. **排队模型简化**：使用 M/M/1 近似，实际在线服务中请求到达可能非 Poisson 分布。
7. **缺乏长序列或大规模集群验证**：最大序列长度为 4096，未评估更长序列；集群规模为 2-4 节点，未验证更大规模。

---

## 与 EfficientPaper 相关的研究方向

1. **MoE 模型高效推理**：MixServe 属于 MoE 模型分布式推理优化领域，与 DeepSeek-R1（baseline）和 Qwen3（baseline）直接相关，是这些模型的高效部署方案。
2. **混合并行策略**：TP-EP 混合并行是 MoE 模型部署的核心技术方向，与 MoE Parallel Folding（Megatron-Core）等高维并行方法互补。
3. **通信优化**：融合 AR-A2A 通信算法属于通信优化范畴，与 DistServe（P/D 分离）、MegaScale-Infer（专家并行分离）等系统层面优化相关。
4. **自动并行策略选择**：MixServe 的自动分析器与 Alpa/AlpaServe（自动化算子级并行）属于同一研究方向——自动化并行策略选择。
5. **分布式 LLM 服务系统**：与 vLLM、Orca、Llumnix、Sarathi-Serve 等分布式服务系统构成生态，MixServe 可作为其中的并行策略优化层。
6. **带宽层次利用**：利用节点内/节点间带宽差异进行通信优化，是分布式系统中的通用优化思路，可推广到其他并行策略。
7. **Overlap 通信与计算**：通信重叠技术是提升分布式系统效率的重要手段，与 Tiling、流水线等技术互补。

---

> **声明**：本 note 由 AI Agent（Hermes Agent）自动生成，基于 arXiv 论文 PDF 全文提取和分析。生成时间：2026 年 6 月。内容仅供参考，如有错误请以原文为准。
