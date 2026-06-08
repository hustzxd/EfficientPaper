# PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

> Jinjun Yi, Zhixin Zhao, Yitao Hu, Ke Yan, Weiwei Sun, Hao Wang, Laiping Zhao, Yuhao Zhang, Wenxin Li, Keqiu Li
> 
> Tianjin University, Stevens Institute of Technology
> 
> 出版：ASPLOS 2026
> 
> 代码：https://github.com/flashserve/PAT
> 
> **本文由 AI Agent 自动生成（生成时间：2026-06-04），所有内容用中文撰写。**

---

## 一句话总结

PAT 是一种前缀感知的注意力内核实现，通过 pack-forward-merge 执行范式，将共享前缀的查询打包到同一 CTA 中以减少冗余全局内存访问，并利用多 tile 内核和多流执行策略实现高效的硬件资源利用，最终将注意力延迟降低 53.5%，TPOT 降低 17.0%-93.1%。

---

## 摘要翻译

LLM 服务正日益被解码注意力（decode attention）所主导，由于需要从全局内存加载大量 KV 缓存，解码注意力是一种内存密集型操作。同时，真实工作负载在请求之间表现出大量层级化的共享前缀（如系统提示、工具/模板、RAG）。现有的注意力实现未能充分利用前缀共享：每个查询对应一个 CTA 的执行方式会重复加载共享前缀 KV 缓存，而一刀切的 tile 设计导致片上资源闲置，并加剧了 KV 长度不均匀时的执行气泡。这些选择放大了内存带宽压力，阻碍了内存密集型解码注意力的执行。

本文介绍了 PAT，一种用于 LLM 解码的前缀感知注意力内核实现，采用 pack-forward-merge 范式组织执行。PAT 通过共享前缀打包查询以减少重复内存访问，运行定制的多 tile 内核以实现高资源效率。它进一步采用实用的多流转发和 KV 分割策略来减少资源气泡。最终的合并阶段使用开销可忽略的在线 softmax。我们将 PAT 作为 vLLM 的即插即用插件实现。在真实和合成工作负载上的评估表明，PAT 相比最先进注意力内核，平均减少 53.5% 的注意力延迟，在相同配置下 TPOT 降低 17.0%-93.1%。

---

## 研究动机

### 背景与问题

1. **解码注意力成为瓶颈**：随着 LLM 上下文长度扩展至百万级别（如 Llama-4 支持 1000 万输入 token）以及 Chain-of-Thought 等技术导致输出长度增加，解码阶段的注意力操作反复从全局内存加载不断增长的 KV 缓存，成为内存瓶颈。实验表明，解码注意力可占总延迟的 68%。

2. **共享前缀的普遍性**：真实工作负载中，40%-62% 的 KV 缓存前缀在请求之间共享（系统提示、RAG 文档、Agent 模板等），形成多层级层次结构。虽然现有系统（如 vLLM、SGLang）支持 KV 缓存复用来减少内存占用，但无法减少全局内存访问——而这正是解码注意力的瓶颈。

3. **现有内核的两大缺陷**：
   - **冗余内存访问**：查询中心化（query-centric）的注意力内核（如 FlashAttention）采用 one-query-per-CTA 策略，每个查询独立处理其 KV 缓存，导致共享前缀被重复从全局内存加载。实测显示 FlashAttention 的 KV 缓存流量比理论最小值多 4.3-8.7×。
   - **资源效率低下**：无论查询中心化还是 KV 中心化内核，都采用一刀切的 tile 大小设计（如 m=64, n=32），忽略了动态工作负载特性。当查询数少于 tile 大小时需要填充（padding），浪费片上内存；KV 长度不均匀时导致 SM 利用率低和执行气泡。

### 研究目标

设计一个内存导向的前缀感知注意力内核，既能减少冗余全局内存访问，又能维持高硬件资源效率，从而加速 LLM 解码。

---

## 方法（技术细节）

PAT 采用 **pack-forward-merge** 执行范式，包含三个阶段：

### 1. Pack 阶段（打包调度器）

**前缀感知打包调度器**（Pack Scheduler）：

- **输入**：解码批次（batch of queries）+ block table（每个 query 的 KV block ID 列表）
- **树结构 block table**：将二维 block table 转换为树结构，每个内部节点表示共享的 KV block 前缀，属性包括共享前缀长度 l 和共享查询数 s。从根到叶的路径重建每个 query 的完整 KV cache block 序列。

- **利润-开销模型**（Profit Model）：
  - **节点内利润**：将 s 个共享前缀的 query 打包到一个 CTA，将 KV 缓存加载从 s 次减少到 1 次，节省 (s-1) × l × d 的全局内存访问（d 为 head 维度）。但会产生中间结果读写的开销（8 × s × d）。利润-开销比 r = (s-1) × l / (8s) ≥ l/16，由于 l ≥ 16（KV block 粒度），打包总是有正收益。
  - **节点间利润**：比较两种方案：
    - 方案 1（分割）：将父节点和子节点分成独立 CTA
    - 方案 2（合并）：将子节点 vᵢ 与父节点 u 合并到一个 CTA，消除中间结果
    - 方案 2 的增量利润为 4sᵢd - lᵤd，当 4sⱼ > lᵤ 时选择合并
  - 该算法的复杂度为 O(|V| + |E|)，线性时间。

- **懒更新机制**（Lazy Update）：在连续批次迭代之间复用调度结果，直到 block table 变化；将调度器移入服务系统异步运行，与预处理阶段（LayerNorm、QKV 投影）重叠，大幅降低调度延迟。

### 2. 多 tile 内核（Multi-tile Kernel）

**核心思想**：根据 CTA 的查询数量和 KV 长度动态选择合适的 (m, n) tile 大小，而非一刀切。

**约束条件推导**（离线）：
1. **寄存器和共享内存约束（m, n 的上界）**：
   - 共享内存：m × h × b + n × h × b + m × h × b' ≤ S_smem
   - 寄存器：每线程寄存器数 R_thr(m,n) ≤ S_reg_thr；并发 CTA 总寄存器 C × R_CTA(m,n) ≤ S_register
2. **高带宽利用率（n 的下界）**：
   - 在线数据量 D_flight = S × C × n × h × b ≥ L × B（L 为固有内存延迟，B 为可持续带宽）
3. **CUTLASS 约束（m, n 的下界）**：
   - m, n 必须为 2 的幂次且 ≥ 16

通过离线配置求解器为每个硬件目标计算可行的 tile 大小对。

**Tile 选择器**（Tile Selector）：在线决策树，常量时间查找：
- **Q tile m**：向上取整规则，选择最小的满足 m ≥ q 的可行 m 值
- **KV tile n**：根据 KV 长度自适应——长 KV 选择大 n（减少并发 CTA 数，缩小执行气泡）；短 KV 选择小 n（避免最后 tile 的计算气泡）
- **内核等效性**：所有可行配置在无前缀和气泡时性能等效（带宽利用率 83%-86%，延迟差异 <2%）

### 3. Forward 阶段（多流转发 + KV 分割）

**多流转发**（Multi-Stream Forward）：
- 为每个不同的 tile 大小配置 (m, n) 创建独立的 CUDA stream
- 相同配置的 CTA 在同一 stream 内顺序执行，不同 stream 并行运行
- 重叠后续内核的启动开销与前序内核的执行，减轻执行气泡

**长 KV 分割**（Long KV Split）：
- 将 KV 长度超过批次平均值的 CTA 分割成等长部分，使每部分 KV 长度不超过平均值
- 缩短最慢 CTA 的完成时间，提高整体 SM 利用率

### 4. Merge 阶段（输出合并）

- 使用在线 softmax（online softmax）的轻量级合并内核
- 每个 CTA 产生三个中间结果：最大分数、log-sum-exp 累加器、部分加权和
- 合并内核从全局内存加载中间结果，使用 online softmax 归约，拼接所有 head，写回最终输出
- 开销可忽略

### 实现细节

- 约 3k 行 Cutlass/CuTe 和 C++ 代码
- 使用 cp_async 原语和双缓冲实现数据传输与计算重叠
- 通过 pybind11 暴露 Python API
- 集成为 vLLM（v0.9.0）的即插即用插件（约 1.2k 行 Python 代码）
- 只需设置环境变量 `VLLM_ATTENTION_BACKEND=PAT` 即可启用

---

## 实验结果

### 实验设置
- **模型**：Llama-3-8B、Qwen3-8B
- **硬件**：NVIDIA A100-SXM4-80GB GPU
- **软件**：CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0
- **基线**：FlashAttention (v2.6.2)、FlashInfer (v0.2.5)、FastTree、RelayAttention、RelayAttention++

### 内核性能（合成工作负载）

在有共享前缀的配置中：
- 相比 FlashAttention：平均降低 67.8% 注意力延迟（最高 21.5× 加速）
- 相比 FlashInfer：平均降低 52.1% 注意力延迟（最高 11.7× 加速）
- 相比 FastTree：最高 3.2× 加速（FastTree 是最强基线）
- 相比 RelayAttention：最高 11.9× 加速
- 相比 RelayAttention++：最高 5.7× 加速
- 无共享前缀时，PAT 仍通过多 tile 内核和多流转发获得 1.6% 的改进

### 端到端性能（真实工作负载）

使用 vLLM 在两条真实 trace（toolagent、conversation）上评估：
- **TPOT 降低**：
  - 相比 FlashAttention：17.0%-89.5%
  - 相比 FlashInfer：32.2%-93.1%
  - 相比 RelayAttention++：17.2%-68.1%
- **TTFT 降低**：
  - 相比 FlashAttention：10.1%-99.6%
  - 相比 FlashInfer：22.5%-99.8%
  - 相比 RelayAttention++：9.3%-98.6%
- **P99 TPOT**：19.4%-93.4% 降低

### 消融实验

- **PAT-compute**（使用 FastTree 的计算导向成本模型）：延迟比 PAT 高 4.6%，内存读写高 10.9%
- **PAT-naive**（简单打包每个树节点为一个 CTA）：延迟比 PAT 高 10.4%，内存读写高 16.7%
- **PAT-fixed**（固定 tile 大小 64×128）：延迟比 PAT 高 39%
- **PAT-serial**（串行执行多 tile 内核）：延迟比 PAT 高 4.8%

### 开销分析

- 平均调度延迟比预处理延迟低 81.6%-88.8%
- 懒更新机制有效，异步 CPU 线程运行调度器不会引入额外端到端延迟

---

## 优势

1. **显著的性能提升**：注意力延迟平均降低 53.5%，TPOT 最高降低 93.1%，效果突出
2. **设计全面且深入**：从 pack 阶段的前缀感知打包，到 forward 阶段的多流执行，再到 merge 阶段的在线 softmax，每个阶段都有精心设计
3. **即插即用**：作为 vLLM 的插件，只需设置一个环境变量即可启用，无需修改模型或服务框架
4. **良好的可移植性**：多 tile 内核的约束推导过程适用于不同 GPU 架构（已在 A100 和 H100 上验证）
5. **开销可控**：懒更新机制和异步调度使得调度开销可忽略
6. **消融实验充分**：每个设计组件的贡献都经过量化验证

---

## 局限

1. **依赖共享前缀的存在**：当工作负载中没有共享前缀时（如小批次或无共享前缀的负载），PAT 的优势有限（仅 1.6% 改进）
2. **与特定模型架构相关**：对于使用 MLA、线性注意力、MLKV 等压缩或移除 KV 状态的架构，PAT 的收益可能缩小
3. **GPU 调度不可控**：GPU 调度的不可控性导致仍存在残余执行气泡，与理论最优仍有差距
4. **硬件适配需要重新推导**：虽然约束推导过程通用，但将 PAT 移植到新 GPU 需要重新推导可行的 tile 大小配置
5. **仅针对解码阶段**：PAT 仅优化解码注意力，不涉及预填充阶段

---

## 与 EfficientPaper 相关的研究方向

1. **LLM 推理优化**：PAT 属于 LLM 服务系统中的注意力内核优化，与 KV 缓存管理（如 vLLM 的 paged KV cache）、推理调度（continuous batching）等方向密切相关
2. **GPU 内核优化**：PAT 的多 tile 内核设计和多流执行策略是 GPU 编程优化的典型案例，涉及共享内存管理、寄存器分配、CUDA stream 调度等
3. **前缀缓存复用**：PAT 利用跨请求的前缀共享来减少内存访问，这一方向与 SGLang、vLLM 等系统的前缀缓存机制互补
4. **内存密集型计算优化**：PAT 的 pack-forward-merge 范式和内存导向的成本模型，对类似内存密集型的推理工作负载（如长上下文推理、RAG 等）有参考价值
5. **部署（deployment）**：PAT 的关键词为 deployment，其作为 vLLM 插件的即插即用特性使其成为 LLM 部署场景中重要的优化手段

---

*本 note 由 AI Agent 自动生成，基于论文全文阅读和结构化分析。*
*生成时间：2026-06-04*
*论文来源：arXiv:2511.22333v2, ASPLOS 2026*
