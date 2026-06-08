# DistributedGEMM

![](fig1.jpg)

> **本文由 AI Agent 自动生成，内容基于博客文章全文、GitHub 源代码及公开资料分析，仅供参考。**
>
> 生成时间：2025-06-05 | 生成模型：Hermes Agent | 来源：[SHI Labs Blog](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b)

---

## 一句话总结

Distributed GEMM 是 NVIDIA 基于 CUTLASS 框架实现的分布式 GEMM 系统，通过将通信操作（All-Gather / Reduce-Scatter）与计算（GEMM）进行流水线化重叠，在 Hopper 架构的 NVLink 网络上实现了高效的张量并行，显著提升了多 GPU 张量并行的计算效率。

---

## 摘要翻译

DistGEMM 旨在利用快速的 CUTLASS 内核和计算-通信流水线化技术，在基于 NVLink 的 GPU 网络上大幅提升张量并行（Tensor Parallelism）的性能。该工作提出了一种在 CUTLASS 框架内原生实现的分布式 GEMM 方案，支持 Hopper（SM90）和 Blackwell（SM100）架构，要求任意对任意（any-to-any）的 NVLink 网络拓扑。

在典型的 All-Gather + GEMM 场景中，各设备上分布的激活切片先被汇聚到本地设备，然后 GEMM 生成输出切片。DistGEMM 通过将通信和计算分解为细粒度操作并进行流水线化，实现了通信与计算的高效重叠，从而减少 GPU 空闲时间，提升整体计算吞吐量。

该实现支持多种调度策略（schedules），包括：
- All Gather + GEMM（旋转 A/B 矩阵）
- GEMM + Reduce Scatter（旋转 C 矩阵）

---

## 研究动机

### 1. 张量并行的通信瓶颈

在大规模 LLM 训练和推理中，张量并行（Tensor Parallelism, TP）是常用的模型并行策略。TP 将每一层的计算分割到多个 GPU 上，每次前向传播都需要进行 all-reduce 或 all-gather 等通信操作。在单节点 NVLink/NVSwitch 网络中，通信开销成为性能瓶颈。

### 2. 现有实现的局限

传统的 TP 实现通常使用 NCCL 库进行通信，计算和通信是串行的，通信期间 GPU 处于空闲状态，造成资源浪费。现有方案难以在保持计算效率的同时实现通信与计算的重叠。

### 3. 为什么需要原生实现

DistGEMM 的核心动机是：在 CUTLASS 框架内原生实现分布式 GEMM，利用 CUTLASS 的高性能 GEMM 内核和灵活的调度机制，将通信操作（All-Gather / Reduce-Scatter）与 GEMM 计算进行深度集成，实现细粒度的计算-通信重叠，从而最大化 GPU 利用率。

---

## 方法（技术细节）

### 1. 整体架构

DistGEMM 是一个基于 CUTLASS 的分布式 GEMM 实现，利用 CUTLASS 的实验性 Distributed GEMM API（`cutlass/experimental/distributed/`）在多 GPU 之间实现张量并行。系统的核心组件包括：

- **分布式 GEMM 设备封装器**（`dist_gemm_universal_wrapper.hpp`）
- **分布式 GEMM 内核包装器**（`dist_gemm_kernel_wrapper.hpp`）
- **分布式 GEMM 调度策略**（`dist_gemm_1d_schedules.hpp`）

### 2. 调度策略（Schedules）

DistGEMM 支持两种主要的分布式调度策略，每种策略都有两个变体：

#### All Gather + GEMM（先汇聚后计算）

- **AllGather1D_TilingCD_RotatingA**：对 A 矩阵进行旋转分片，所有设备先执行 All-Gather 操作汇聚 A 矩阵的切片，然后本地执行 GEMM 计算。
- **AllGather1D_TilingCD_RotatingB**：类似地，对 B 矩阵进行旋转分片。

在这种模式下，激活切片分布在各设备上，先被汇聚到每个设备的本地内存，然后本地 GEMM 计算生成输出切片。

#### GEMM + Reduce Scatter（先计算后归约）

- **ReduceScatter1D_TilingA_RotatingC**：各设备先在本地执行 GEMM 计算（使用部分数据），然后通过 Reduce-Scatter 操作归约输出。
- **ReduceScatter1D_TilingB_RotatingC**：类似地，对 B 矩阵进行旋转分片。

### 3. 流水线化与重叠（Pipelining & Overlapping）

DistGEMM 的关键创新在于将通信和计算操作进行流水线化重叠。具体来说：

- **单阶段 vs 流水线化**：DistGEMM 对比了单阶段（single-stage）和流水线化（pipelined）两种模式。在流水线化模式中，GEMM 的不同 tile 的计算与通信操作交替执行，从而实现计算与通信的重叠。
- **Tile 级别的重叠**：通过将 GEMM 分解为多个 tile，每个 tile 的计算可以在本地设备上执行，而相邻 tile 的通信（All-Gather 或 Reduce-Scatter）可以在后台同时进行。
- **Grid Dependency Control（GDC）**：利用 CUDA 的 GDC 指令（Grid Dependency Control），在 Hopper 架构上实现细粒度的内核调度和依赖管理，使得通信和计算内核可以更好地重叠执行。

### 4. 硬件要求

- **GPU 架构**：Hopper（SM90）或 Blackwell（SM100）
- **网络拓扑**：任意对任意（any-to-any）的 NVLink 网络（如 8×NV18 拓扑）
- **CUDA 版本**：CUDA 12.6 或更新版本
- **编译选项**：需要启用 `CUTLASS_ENABLE_GDC_FOR_SM90`（Hopper）或 `CUTLASS_ENABLE_GDC_FOR_SM100`（Blackwell）

### 5. 配置参数

- **TP 并行度**：默认 8 个 GPU（TP=8）
- **数据类型**：FP16（`half_t`）
- **Tile 形状**：128×256×64（Threadblock 级别）
- **Cluster 形状**：1×2×1
- **调度策略**：`KernelTmaWarpSpecializedPingpong`（使用 TMA 和 Warp Specialized 调度）
- **Epilogue 调度**：`TmaWarpSpecialized`

### 6. TMA（Tensor Memory Access）

DistGEMM 使用 Hopper 架构的 TMA（Tensor Memory Access）机制进行高效的数据加载和存储。TMA 可以将数据从全局内存直接加载到共享内存，减少对共享内存的占用，并支持异步操作。

---

## 实验结果

### 1. 性能指标

DistGEMM 的示例代码提供了性能测试功能，可以测量不同问题规模下的运行时间和 TFLOPS：

- **示例命令**：`./65_distributed_gemm --m=16384 --n=106496 --k=16384 --warmup-iterations=10 --iterations=100`
- **输出格式**：报告平均运行时间（ms）和 TFLOPS
- **正确性验证**：默认与单设备 GEMM 进行比较，验证输出正确性

### 2. 问题规模

DistGEMM 支持大规模 GEMM 问题，通过 TP 分片后，每个设备的本地 GEMM 规模为：
- `local_M = M / TP`
- `local_N = N`
- `local_K = K`
- `local_L = L`（batch 维度）

### 3. 性能优势

- 通过计算-通信重叠，减少了 GPU 空闲时间
- 利用 CUTLASS 的高性能 GEMM 内核，最大化计算吞吐量
- 流水线化调度策略在大规模问题上表现优异

### 4. 与 NCCL 的对比

DistGEMM 与传统的 NCCL-based TP 实现相比：
- DistGEMM 将通信与计算深度融合，实现更细粒度的重叠
- DistGEMM 利用 CUTLASS 的内核调度机制，减少了通信开销
- 在 Hopper 架构上，DistGEMM 的 GDC 指令进一步优化了内核调度

---

## 优势

### 1. 原生集成

DistGEMM 在 CUTLASS 框架内原生实现，利用 CUTLASS 的高性能 GEMM 内核和灵活的调度机制，避免了 NCCL 的额外开销。

### 2. 计算-通信重叠

通过将通信和计算分解为细粒度操作并进行流水线化，实现了计算与通信的高效重叠，最大化 GPU 利用率。

### 3. 灵活的调度策略

支持多种调度策略（All-Gather + GEMM / GEMM + Reduce-Scatter），可以根据不同场景选择最优策略。

### 4. 硬件优化

利用 Hopper 架构的 TMA（Tensor Memory Access）和 GDC（Grid Dependency Control）指令，实现高效的内存访问和内核调度。

### 5. 多架构支持

支持 Hopper（SM90）和 Blackwell（SM100）架构，具有良好的可扩展性。

### 6. 简单易用

通过 CUTLASS 的模板化 API，用户可以轻松配置并行度（TP）、调度策略和问题规模，降低了使用门槛。

---

## 局限

### 1. 硬件限制

- 需要 Hopper（SM90）或 Blackwell（SM100）架构的 GPU
- 需要任意对任意（any-to-any）的 NVLink 网络拓扑
- 需要 CUDA 12.6 或更新版本

### 2. 实验性质

DistGEMM 目前是实验性功能（experimental API），可能不完全稳定，未来可能有接口变更。

### 3. 数据类型限制

当前示例仅支持 FP16（`half_t`），对其他数据类型（如 BF16、FP8、INT8）的支持可能需要额外适配。

### 4. 通信与计算的粒度

虽然 DistGEMM 实现了计算-通信重叠，但重叠的粒度受限于 tile 大小和网络带宽，在某些场景下可能无法完全隐藏通信延迟。

### 5. 缺乏全面的性能基准

目前公开的性能数据有限，缺乏与现有 NCCL-based TP 实现的系统对比。

### 6. 集成与部署

DistGEMM 目前主要作为 CUTLASS 的示例代码，与现有框架（如 PyTorch、Megatron-LM）的集成需要额外工作。

---

## 与 EfficientPaper 相关的研究方向

### 1. 计算-通信重叠

DistGEMM 的核心贡献是计算-通信重叠，这与 EfficientPaper 项目中的多个研究方向相关：
- **Async-TP**：异步张量并行，通过异步通信实现计算-通信重叠
- **DeepGEMM**：JIT 编译的 CUDA 张量核心库，也实现了 NVLink 通信与张量核心计算的重叠

### 2. 张量并行优化

DistGEMM 是张量并行优化的重要工作，与以下研究方向相关：
- **All-Gather / Reduce-Scatter 优化**：通过分割通信操作实现更细粒度的重叠
- **Tile 级别的分布式计算**：将 GEMM 分解为多个 tile，在多 GPU 之间分布计算

### 3. 硬件优化

DistGEMM 利用 Hopper 架构的 TMA 和 GDC 指令，与以下研究方向相关：
- **TMA（Tensor Memory Access）**：Hopper 架构的高效内存访问机制
- **GDC（Grid Dependency Control）**：Hopper 架构的内核调度机制

### 4. 大规模 LLM 训练与推理

DistGEMM 为大规模 LLM 训练和推理提供了高效的张量并行实现，与以下研究方向相关：
- **模型并行**：通过张量并行加速大规模模型的训练和推理
- **混合并行**：结合数据并行、模型并行和流水线并行的混合策略

### 5. 分布式计算系统

DistGEMM 作为分布式 GEMM 实现，与以下研究方向相关：
- **分布式线性代数**：通过分布式计算加速矩阵运算
- **GPU 集群优化**：通过高效通信和计算优化 GPU 集群的性能

---

## AI 生成声明

> **本文由 AI Agent 自动生成，内容基于以下来源：**
> - [SHI Labs Blog](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b)
> - [GitHub 代码](https://github.com/NVIDIA/cutlass/blob/main/examples/65_distributed_gemm/README.md)
> - [GitHub 示例代码](https://github.com/NVIDIA/cutlass/blob/main/examples/65_distributed_gemm/65_distributed_gemm.cu)
> - [LinkedIn 公告](https://www.linkedin.com/posts/thakkarv_distributed-gemm-activity-7269342255738458112-1bIT)
> - [NVIDIA CUTLASS 文档](https://docs.nvidia.com/cutlass/latest/)
>
> **生成模型**：Hermes Agent | **生成时间**：2025-06-05
>
> **免责声明**：本文内容仅供参考，可能存在错误或遗漏。建议读者阅读原始博客文章和代码以获取准确信息。
