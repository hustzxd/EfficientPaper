# Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs

> Qijun Zhang, Chen Zhang, Zhuoshan Zhou, Haibo Wang, Zhe Zhou, Zhipeng Tu, Guangyu Sun, Zhiyao Xie, Yijia Diao, Zhigang Ji, Jingwen Leng, Guanghui He, Minyi Guo
>
> Hong Kong University of Science and Technology, Shanghai Jiao Tong University, Huawei Technologies, Peking University, Shanghai Qi Zhi Institute
>
> ISCA 2026

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Mixture-of-Experts (MoE) 已被众多领先的大模型采用以降低计算需求。然而，MoE 专家并行 (Expert Parallelism, EP) 中频繁的 GPU 间通信成为性能瓶颈。本文观察到 MoE 中存在大量可通过 in-switch computing 消除的冗余数据传输。现有的 NVLink SHARP (NVLS) 方案仅支持静态集合通信（规则通信模式），无法处理 MoE 中动态通信（不规则通信模式）。为弥补这一功能差距，作者提出 **DySHARP**，一种完整的动态 in-switch computing 加速方案，包含两部分：1）**动态多播寻址 (Dynamic Multimem Addressing)**，在 ISA、架构和运行时层面协同设计，作为 NVLS 的动态扩展，减少冗余流量；2）**以 Token 为中心的内核融合 (Token-Centric Kernel Fusion)**，深度融合 Dispatch-计算-Combine 流水线，解决流量减少的不对称性，将流量减少转化为实际加速。相比 SOTA 方案，DySHARP 实现了最高 **1.79×** 的加速。

## 一句话总结

DySHARP 通过在 NVSwitch 中引入动态 in-switch computing（支持动态多播寻址和 reduction），消除 MoE 专家并行中约 50% 的冗余通信流量，并通过 token 级流水线融合将流量减少转化为实际端到端加速，最高实现 1.79× speedup。

## 背景与问题

### MoE 通信瓶颈

MoE 架构将 FFN 层拆分为多个专家，每个 token 仅激活 topk 个专家进行计算。在多 GPU 系统中，专家并行 (EP) 将专家分布到不同 GPU 上，但 token 需要被路由到远程 GPU 上的专家，引入了 **Dispatch**（发送 token 到目标 GPU）和 **Combine**（聚合专家输出回源 GPU）两种通信操作。这些通信占据 MoE 层执行时间的 **50-80%**，是严重的性能瓶颈。

### 冗余通信问题

MoE 通信存在根本性的低效：**大量冗余数据传输**。

- **Dispatch 冗余**：一个 token 需要发送到多个 GPU 时，相同数据从源 GPU 到 switch 被传输多次。
- **Combine 冗余**：多个 GPU 上可聚合的专家输出被分别独立传输回源 GPU。

在模拟的 GH200 NVL32 上对 DeepSeek-V3 进行 profiling，发现近 **50%** 的总流量是冗余的。

### 现有 in-switch computing 的局限

NVLink SHARP (NVLS) 已集成在 NVLink/NVSwitch 中，通过 `multimem.st`（in-switch multicast）和 `multimem.ld_reduce`（in-switch reduction）支持静态集合通信。理论上可以用 multicast 消除 Dispatch 冗余、用 reduction 消除 Combine 冗余。

但 NVLS 是**静态**的，仅支持：
- **固定目标集**：所有 token 总是与相同的 GPU 组通信
- **对称寻址**：token 在各 GPU 上位于相同的内存偏移

而 MoE 的通信是**动态**的：
- **变化的目标集**：不同 token 路由到不同的专家子集
- **非对称寻址**：token 在各 GPU 上独立分配，内存偏移不同

将 MoE 通信强行转换为静态集合通信（用 AllGather 模拟 Dispatch、Reduce-Scatter 模拟 Combine）会引入 **340%** 的无用流量，完全抵消了 in-switch computing 的收益。

## 核心方法

DySHARP 提出两个协同工作的技术：

### 1. 动态多播寻址 (Dynamic Multimem Addressing)

**核心思想**：扩展 NVLS 的 multimem 寻址机制，支持动态不规则通信，同时保持高 payload 效率。

**关键洞察**：Dispatch 和 Combine 是 AllGather 和 Reduce-Scatter 的动态对应物。虽然目标集变化、内存布局不对称，但 **代数索引 (algebraic index)** 在各 GPU 间是相同的——token 在结果张量中的代数位置一致，只是存储布局不同。

**设计要点**：

- **自定义数据包格式**：数据包携带单一 multimem 地址（代数索引）+ 轻量级目标专家列表。相比显式寻址（嵌入所有目标地址），保持了紧凑的包头和高 payload 效率。
- **ISA 扩展**：引入 `dymultimem.st`（Dispatch multicast）和 `dymultimem.ld_reduce`（Combine reduction）指令，额外指定目标计数和目标列表基地址。
- **硬件内存管理器 (Memory Manager)**：在目标 GPU 的 Hub 中，通过 AL Table（代数-布局映射表）将 multimem 地址翻译为虚拟地址。Dispatch 时分配布局块，Combine 时复用映射。
- **AL TLB**：类似传统 TLB，加速 AL Table 查找，利用 token 向量的访问局部性实现高命中率。
- **Switch 增强**：Route 模块根据目标列表计算输出端口，复制并裁剪请求包；对 ld_reduce 请求，记录目标计数并在响应聚合完成后返回结果。
- **CUDA Runtime 扩展**：`CUDymulticastObjectProp` 和 `cuDyMulticastCreate` 等 API 支持动态 multicast 对象管理。

### 2. 以 Token 为中心的内核融合 (Token-Centric Kernel Fusion)

**核心思想**：将 MoE 层视为 token 级流水线而非四个孤立算子，在 token/tile 粒度确定就绪状态并调度。

**动机**：动态多播寻址减少了约 50% 的流量，但流量减少在两个方向上是**不对称**的——multicast 减少 GPU→switch 流量（Dispatch），reduction 减少 switch→GPU 流量（Combine）。如果 Dispatch 和 Combine 独立执行，未被优化的方向成为瓶颈，流量减少无法转化为加速。

**解决方案**：

- **Token Tracker**：追踪 token 级依赖链（Dispatch→GEMM-1→GEMM-2→Combine），包含三个轻量表：
  - **TS Table**（Tile Status）：追踪 Dispatch→GEMM-1 和 GEMM-1→GEMM-2 的就绪状态
  - **TID Table**（Token ID）：记录每个 tile 中的 token ID，用于 GEMM-2 完成后通知源 GPU
  - **OR Table**（Output Readiness）：在源 GPU 端追踪每个 token 的 topk 个专家输出是否全部就绪

- **Token-Centric Scheduler**：基于就绪检测的调度器，实现 token 级流水线
  - SM 分为四组：Dispatch、GEMM-1、GEMM-2、Combine
  - 就绪门控调度：GEMM TB 行仅在 tracker 标记就绪后才被调度；Combine 仅在 token 的所有 topk 输出就绪后执行
  - Dispatch 和 Combine 自然并发执行，合并互补的非对称通信模式，提高双向带宽利用率

**两个技术缺一不可**：单独的内核融合（无 in-switch computing）不比 SOTA 有优势；单独的动态多播寻址（无内核融合）因不对称性无法转化为加速。

## 技术细节

### 包格式扩展

在 NVLink 数据链路层包格式基础上扩展：
- flit0 中 64-bit 地址替换为：48-bit multimem 地址 + 1-bit stage + 15-bit target count
- 后续 flit 编码目标专家 ID（每个 16-bit，每 flit 8 个）
- payload flit 保持不变

### 加权 Combine 支持

`dymultimem.ld_reduce` 不支持带权 reduction（硬件复杂度高）。解决方案：在 GEMM-2 的 epilogue 中预先乘以 gate 权重 $w_i$，使得不带权 reduction $\sum_i(w_i \cdot o_i)$ 等效于加权求和。

### 硬件开销

- Switch 端：路由计算控制逻辑 < 0.01mm²（< 0.1% NVSwitch die 面积），仅增加 1 个 cycle 延迟
- GPU 端：全部额外架构支持仅需 0.198mm²（约 0.024% H100 GPU die 面积）
- AL Table：每 entry 4B，处理 1M token 时仅 4MB/layer
- AL TLB：512 entry 为最优配置，近理想命中率
- Reduction buffer：64KB 为最优配置，几乎无 eviction

## 实验设置

- **硬件**：模拟 NVIDIA GH200 NVL32（32 GPU，9 NVSwitch 全连接 fat-tree 拓扑），GPU 配置基于 H200 规格
- **仿真工具**：BookSim2 + 定制 Accel-Sim（cycle-accurate）
- **模型**：DeepSeek-V3 及其变体（Small/Medium/Large），topk = 8/16/32
- **对比方案**：DeepEP、NVLS、FasterMoE、Tutel、CCFuser、COMET、DualPipe
- **验证**：模拟器在 GEMM 和 DeepEP 通信算子上与 DGX-H100 实测误差在 6% 以内

## 主要结果

### 端到端性能

- 相比 DeepEP、NVLS、FasterMoE、Tutel、CCFuser、COMET、DualPipe，端到端训练加速分别为最高 **2.31×、5.12×、2.11×、1.98×、1.85×、1.79×、1.88×**
- 几何平均加速分别为 1.93×、3.38×、1.84×、1.72×、1.63×、1.59×、1.66×

### MoE 层性能

- MoE 层单独对比（排除 DualPipe），相比其他 6 个基线最高加速 **2.77×、6.93×、2.48×、2.32×、2.01×、1.94×**

### 流量减少

- DySHARP 将总通信流量减少约 **50%**（相比 DeepEP）
- 纯通信性能达到理论理想的 **90%** 以上
- NVLS 作为 workaround 反而增加了流量（因无用数据传输）

### 消融实验

- Dynamic multimem addressing alone（DySHARP-Basic）：减少流量但不直接加速
- Kernel fusion alone：不比 COMET 有优势
- 两者结合（完整 DySHARP）：将流量减少转化为加速，验证了缺一不可的分析

### 敏感性分析

- **GPU 数量**：4-64 GPU 范围内，DySHARP 一致性优于基线，且随 GPU 数增加差距扩大
- **序列长度**：1024-16384 范围内均保持最短执行时间，长序列优势更显著
- **Token 分布**：训练和推理分布下均保持显著加速

### 推理性能

- 在 prefill 和 decode 阶段均展示了优势

### 其他模型

- 在 GPT-OSS-120B 和 Qwen3-235B 上同样展示了优势

## 优点与局限

### 优点

1. **首次解决 MoE 动态通信的 in-switch computing 问题**：填补了 NVLS 仅支持静态集合通信的功能空白
2. **全栈协同设计**：从 ISA、数据包格式、微架构到 CUDA Runtime 的完整扩展
3. **两个技术互补**：动态多播寻址消除冗余流量，内核融合解决不对称性，缺一不可
4. **极低硬件开销**：GPU 端仅 0.024% die 面积，Switch 端 < 0.1% die 面积
5. **可扩展性强**：随 GPU 数和序列长度增加优势更明显
6. **向多节点扩展的潜力**：讨论了通过 InfiniBand Quantum Switch 扩展到多节点的方案

### 局限

1. **模拟验证而非实际硬件实现**：在 cycle-accurate 模拟器上验证，未实际流片
2. **依赖 NVSwitch 互连**：需要 NVLink/NVSwitch 环境，不适用于其他互连拓扑
3. **无开源代码**：论文未提供代码仓库
4. **加权 reduction 依赖软件 workaround**：权重在 GEMM epilogue 中预乘，非原生硬件支持
5. **持久线程块 (persistent TB) 实现的调度器**：可能在某些工作负载下存在 SM 利用率不均的问题

## 与 EfficientPaper 主题的关系

本文属于 **高效 AI 推理/训练** 和 **系统架构优化** 范畴，具体涉及：

- **MoE 加速**：针对 MoE 架构的通信瓶颈进行优化
- **多 GPU 通信优化**：通过 in-switch computing 减少 GPU 间冗余数据传输
- **硬件-软件协同设计**：ISA、微架构、运行时的全栈优化
- **网络计算 (In-Network Computing)**：将计算能力下沉到交换机层面

这是 EfficientPaper 关注的高效计算系统领域的前沿工作，展示了硬件-软件协同设计在解决 LLM 训练/推理通信瓶颈方面的潜力。

## 可复现/实现要点

- **仿真框架**：基于 BookSim2 + Accel-Sim，需扩展支持 Hopper FP8 kernel 和多 GPU 并发执行
- **NVLink 参数**：双向带宽 900 GB/s，单向延迟 250ns，flit 大小 16B
- **NVSwitch 配置**：每输入端口 16 个 256-depth VC（8 请求 + 8 响应），端口 reduction buffer 64KB
- **DySHARP 组件**：MultimemQ 32 entry，AL TLB 512 entry，TS Table 和 OR Table 各 1024 entry
- **同步 tile 大小**：128 tokens（匹配 GEMM tile 大小）
- **硬件综合**：TSMC 12nm，Synopsys Design Compiler

## 个人备注

- 这是 ISCA 2026 的论文，展示了 in-switch computing 从静态集合通信扩展到动态通信的完整方案
- 与同一团队的 CAIS (HPCA 2026) 工作互补——CAIS 关注 tensor parallelism 中的 in-switch computing，本文关注 MoE 专家并行
- NVLink SHARP 的局限性被清晰地分析，为后续研究提供了明确的改进方向
- 向多节点扩展的讨论（通过 InfiniBand）具有实用价值，但具体实现细节需要进一步研究
- 值得关注 NVIDIA 后续是否会将类似动态 in-switch computing 能力集成到硬件中
