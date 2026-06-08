# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

![](../../blank.jpg)

> **⚠️ 生成声明**：本 note 由 AI Agent 自动生成，基于论文全文的文本提取与分析。生成时间：2026-06-05。内容可能存在偏差，请以原论文为准。

---

## 一句话总结

FlashInfer 是一个高效且可定制的注意力计算引擎，通过统一的块稀疏格式、JIT 编译的可定制注意力模板和动态负载均衡调度算法，显著提升 LLM 推理服务的性能，已集成到 SGLang、vLLM 和 MLC-Engine 等主流框架中。

---

## 摘要翻译

Transformer 架构以注意力机制为核心，构成了大语言模型（LLM）的基础。随着模型规模扩大，高效的 GPU 注意力内核对高吞吐量和低延迟推理至关重要。多样化的 LLM 应用需要灵活且高性能的注意力解决方案。本文提出 **FlashInfer**：一个可定制的高效注意力引擎，用于 LLM 推理服务。FlashInfer 采用块稀疏格式和可组合格式来优化内存访问并减少冗余，解决 KV-Cache 存储异构性问题。同时提供可定制的注意力模板，通过即时编译（JIT）适配各种配置。此外，FlashInfer 的负载均衡调度算法能够适应用户请求的动态变化，同时兼容需要静态配置的 CUDAGraph。FlashInfer 已集成到 SGLang、vLLM 和 MLC-Engine 等主流 LLM 服务框架中。全面的内核级和端到端评估表明，FlashInfer 能够显著提升各种推理场景的内核性能：与最先进的 LLM 服务方案相比，在 LLM 服务基准测试中实现了 29-69% 的 token 间延迟降低，长上下文推理延迟降低 28-30%，并行生成加速 13-17%。

---

## 研究动机

LLM 推理服务面临两个核心挑战：

1. **工作负载多样性与输入动态性**：LLM 服务涉及多种注意力计算模式（prefill、decode、树解码等），请求的查询长度和 KV-Cache 大小在批次内和时间上不断变化。朴素实现可能导致负载不平衡，需要动态调度以优化性能。

2. **硬件定制化需求**：
   - **内存侧**：高效的存储格式（如 PagedAttention、RadixAttention）对于管理不断增长的 KV-Cache 和多样化存储模式至关重要。
   - **计算侧**：需要针对特定硬件（如 NVIDIA Turing 到 Hopper 架构）定制流水线和模板，以充分发挥 GPU 性能潜力。
   - **注意力变体**：现代 LLM 使用各种注意力变体（GQA、MQA、特殊掩码、自定义注意力分数计算等），需要灵活的实现策略。

当前各系统针对这些特性的子集实现专门的注意力方案，导致维护开销高且潜在效率低下。FlashInfer 旨在提供一个统一的、可定制的解决方案。

---

## 方法（技术细节）

FlashInfer 的核心设计包含三大组件：统一的 KV-Cache 存储格式、可定制的注意力计算模板、动态感知的运行时调度器。

### 1. 统一的块稀疏格式（Block-Sparse Format）

**核心思想**：将所有 KV-Cache 存储模式（PageTable、RadixTree、稀疏掩码等）统一为块压缩稀疏行（Block-Sparse Row, BSR）格式。

- **块稀疏矩阵表示**：KV-Cache 存储为 BSR 格式的稀疏矩阵，块大小由应用需求决定。Br 对应查询块大小，Bc 由 KV-Cache 管理算法指定。
- **查询/输出表示**：使用 ragged tensor（锯齿数组）高效存储，避免填充（padding），便于将不同请求的查询和输出紧凑打包。
- **任意块大小支持**：FlashInfer 的内核实现支持任意 (Br, Bc) 值。

### 2. 可组合格式（Composable Formats）

受 SparseTIR 启发，FlashInfer 利用多个块稀疏格式存储稀疏矩阵，提供更大的灵活性和内存效率：

- **共享前缀分解**：对于共享前缀的请求，KV-Cache 对应的行和列形成稠密子矩阵，使用较大 Br 的块稀疏矩阵存储。不同请求的唯一 KV-Cache 使用较小块大小的矩阵。
- **无需数据移动**：仅需计算稀疏子矩阵的索引和索引指针数组，无需移动 KV-Cache 数据。
- **内存效率提升**：较大块大小的注意力计算可通过快速共享内存和寄存器访问共享 KV-Cache 条目，显著提升内存效率。

### 3. 可定制的注意力模板（Customizable Attention Template）

**JIT 编译框架**：受 FlexAttention 启发，FlashInfer 设计了可定制的 CUDA 模板和 JIT 编译器，将注意力变体规范作为输入，生成优化的内核代码。

**变体规范包含以下函数对象（functors）**：
- `QueryTransform`、`KeyTransform`、`ValueTransform`：在注意力计算前对 query/key/value 张量进行变换
- `OutputTransform`：对注意力输出张量进行变换
- `LogitsTransform`、`LogitsMask`：在 softmax 计算前对 logits 进行变换和掩码

**支持的特性**：
- 可选的 softmax/sigmoid 等激活函数（如 FlashSigmoid）
- 融合 RoPE、归一化、投影（如 DeepSeek-V2 的 MLA）
- 自定义掩码、logits soft-cap、滑动窗口注意力
- 使用 PTX 指令或用户自定义库

**Tile 大小选择**：
- 为不同架构提供多种 tile 大小：`(1, 16, 32, 64, 128) × (32, 64, 128)`
- 基于硬件资源和工作负载强度的启发式选择
- 为 Ada 架构适配有限的共享内存

**硬件支持**：Turing 到 Hopper（sm75 到 sm90a），使用 FlashAttention2 和 FlashAttention3 算法。

### 4. 动态负载均衡调度（Load-balanced Scheduling）

**调度算法**（Algorithm 1）：
1. 计算每个 tile 的成本：`cost(lq, lkv) = α·lq + β·lkv`
2. 将每个查询块的 KV 分割为最大大小 Lkv 的块
3. 使用优先队列进行负载均衡分配，确保各 CTA（Cooperative Thread Array）工作量均衡

**运行时设计**：
- **Plan-Run 分离**：plan 函数在 CPU 上生成调度计划，run 函数在 GPU 上执行注意力计算，灵感来自 Inspector-Executor 模型
- **CUDAGraph 兼容**：attention 和 contraction 阶段使用持久化内核（persistent kernel），grid 大小固定，满足 CUDAGraph 静态配置要求
- **workspace buffer**：用户提供的 workspace buffer 用于存储部分输出和计划信息，固定偏移量确保指针一致
- **跨层复用**：计划信息可跨层复用，因为同一生成步骤中所有层使用相同的序列长度信息

### 5. 编程接口

FlashInfer 提供 PyTorch 编程接口，支持与 SGLang、vLLM、MLC-Engine 等框架集成：
- `AttentionWrapper`：初始化时 JIT 编译内核并缓存复用
- 支持可组合格式（多个注意力包装器，不同块大小）
- `plan()` 函数激活动态调度器，生成负载均衡计划
- `run()` 函数执行注意力计算，输出结果
- CUDAGraph 捕获 `run()` 调用，编译为单个图

---

## 实验结果

**实验环境**：NVIDIA A100 40GB SXM 和 H100 80GB SXM GPU，CUDA 12.4，PyTorch 2.4.0，f16 精度。

### 1. 端到端 LLM 服务性能

使用 SGLang v0.3.4 评估，与 SGLang + Triton v3.0 对比：

| 指标 | Llama 3.1 8B (1xH100) | Llama 3.1 70B (4xH100) |
|------|----------------------|------------------------|
| **ITL（ShareGPT）** | 21.7ms vs 29.6ms | 48.3ms vs 30.7ms |
| **ITL（Variable）** | 13.5ms vs 9.1ms | 24.0ms vs 21.8ms |
| **TTFT（ShareGPT）** | 49.2ms vs 61.8ms | 141.2ms vs 165.2ms |
| **TTFT（Variable）** | 38.8ms vs 53.2ms | 115.6ms vs 157.8ms |

- **ITL 降低 29-69%**（与 Triton 后端相比）
- **TTFT 显著降低**（在所有设置中均有提升）

### 2. 输入动态性下的内核性能

与 FlashAttention 在不同序列长度分布下的对比（batch size=16）：
- **解码（Decode）带宽利用率**：在 uniform 和 skewed 分布下，FlashInfer 显著优于 FlashAttention
  - H100：FlashInfer (GQA-4) 在 skewed 下达 44%，FlashAttention 仅 33%
  - A100：FlashInfer (GQA-4) 在 skewed 下达 46%，FlashAttention 仅 34%
- **预填充（Prefill）FLOPs 利用率**：FlashInfer 在 skewed 下显著领先（H100: 59% vs 44%）

**原因**：
- 动态负载均衡调度器（uniform/skewed 分布下优势明显）
- 多样化 tile 大小选择（解码场景下 FlashAttention 使用次优 tile 大小）

### 3. 长上下文推理（Streaming-LLM）

使用 Vicuna-13B 在 MT-Bench 上评估 Streaming-LLM：
- **融合 RoPE 的内核**：仅需 20 行额外代码即可实现
- **ITL 降低 28-30%**（不同最近窗口大小下）
- **带宽利用率提升 1.6-3.7×**（融合 RoPE vs 不融合）
- 证明了注意力内核定制化的重要性

### 4. 并行生成性能

在 MLC-Engine 中使用可组合格式，Llama 3.1 8B/70B，ShareGPT 数据集：
- **中等并行度（4≤n≤32）**：可组合格式持续提升
- **ITL 峰值提升**：n=4 时，8B 模型降低 13.73%，70B 模型降低 17.42%
- **TTFT 峰值提升**：n=4 时，8B 模型降低 16.41%，70B 模型降低 22.86%
- **较小/较大 n 无显著提升**：较小 n 块大小不足，较大 n 计算不再由注意力主导

---

## 优势

1. **统一的块稀疏格式**：将 PageTable、RadixTree、稀疏掩码等统一为 BSR 格式，简化 KV-Cache 管理，减少冗余。
2. **可组合格式**：利用多个块稀疏格式实现共享前缀分解，无需数据移动即可提升内存效率。
3. **JIT 编译的可定制模板**：支持灵活的注意力变体（GQA、自定义掩码、融合 RoPE 等），仅需少量代码即可适配新变体。
4. **动态负载均衡调度**：在序列长度变化时自动调整工作负载分配，兼容 CUDAGraph。
5. **广泛的架构支持**：Turing 到 Hopper（sm75 到 sm90a），支持 FlashAttention2 和 FlashAttention3。
6. **与主流框架集成**：已集成到 SGLang、vLLM、MLC-Engine 等生产级系统。
7. **显著性能提升**：ITL 降低 29-69%，长上下文延迟降低 28-30%，并行生成加速 13-17%。

---

## 局限

1. **仅支持前向传播**：当前 FlashInfer 仅支持注意力计算的前向传播，不支持反向传播（训练）。扩展到训练需要开发可定制的反向注意力内核模板。
2. **JIT 编译开销**：JIT 编译可能在首次运行时引入额外开销（尽管内核会被缓存复用）。
3. **硬件依赖**：主要针对 NVIDIA GPU 优化，对其他硬件（如 AMD GPU）的支持有限。
4. **Triton vs CUDA**：FlashInfer 生成 CUDA 代码而非 Triton 代码，因为 Triton 在许多场景下仍落后于 CUDA 和 CUTLASS，但这限制了 GPU 无关性。
5. **未包含所有最新优化**：例如 FlashDecoding++ 的异步全局注意力状态更新、Stream-K 的原子聚合等，留作未来工作。
6. **并行生成的局限**：在极小或极大并行度下（n=1 或 n=64），可组合格式的优势不明显。

---

## 与 EfficientPaper 相关的研究方向

- **sparse_pruning（稀疏剪枝）**：FlashInfer 的块稀疏格式与稀疏剪枝技术相关，可用于 KV-Cache 的动态稀疏化。
- **attention_sparsity（注意力稀疏性）**：FlashInfer 的块稀疏注意力直接利用注意力的稀疏性模式，与注意力稀疏性研究方向紧密相关。
- **kv_cache_management（KV-Cache 管理）**：FlashInfer 的统一块稀疏格式和可组合格式为 KV-Cache 管理提供了新的抽象，与 PagedAttention、RadixAttention 等技术互补。
- **tool（工具）**：FlashInfer 作为一个可定制的注意力引擎，可作为 LLM 推理服务的基础组件（工具），支持各种注意力变体和硬件架构。
- **推理加速**：与 EfficientPaper 关注的推理效率提升方向高度一致，可作为 baseline 或参考方法。
- **长上下文推理**：FlashInfer 对长上下文推理的优化（如 Streaming-LLM 的融合 RoPE）与长上下文高效推理的研究方向相关。
- **并行生成**：可组合格式对并行生成的优化与 LLM Agent、多轮对话等应用场景相关。

---

## 参考信息

- **论文**：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
- **arXiv**：http://arxiv.org/abs/2501.01005v2
- **会议**：MLSys 2025
- **代码**：https://github.com/flashinfer-ai/flashinfer
- **项目主页**：http://flashinfer.ai
- **作者**：Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, Luis Ceze
- **机构**：NVIDIA, University of Washington, Perplexity AI, Carnegie Mellon University
