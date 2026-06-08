# DeepEP

![111](low-latency.jpg)

> **一句话总结**：DeepEP 是 DeepSeek 开源的高性能专家并行通信库，通过统一的 ElasticBuffer 接口、JIT 编译和 NCCL Gin 后端，在 MoE（Mixture of Experts）场景下实现了接近硬件带宽上限的 all-to-all 通信，同时显著减少 SM 资源占用。

---

## 基本信息

- **论文/项目名称**：DeepEP: an efficient expert-parallel communication library
- **作者**：Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, Liang Zhao
- **机构**：DeepSeek
- **发布时间**：2025 年（GitHub 开源）
- **GitHub**：https://github.com/deepseek-ai/DeepEP
- **许可证**：MIT License
- **Stars**：~9700（截至 2025 年 6 月）
- **编程语言**：CUDA
- **注意**：本文为 GitHub 开源项目，无独立 arXiv 论文，本 note 基于官方 README 及技术文档撰写。

---

## 摘要翻译

DeepEP（DeepEveryParallel）是一个面向现代机器学习训练和推理的高性能通信库。该库目前聚焦于**专家并行（Expert Parallelism, EP）**——提供高吞吐、低延迟的 all-to-all GPU 内核（MoE dispatch 和 combine），支持包括 FP8 在内的低精度计算。同时，DeepEP 还提供实验性的流水线并行（PP）、上下文并行（CP）和远程内存访问（Engram）原语，所有内核均设计为零或最小 SM 占用。所有内核在运行时通过轻量级 JIT（Just-In-Time）模块编译，安装时无需 CUDA 编译。

尽管设计轻量，DeepEP 的性能在各种配置下均能达到或超过硬件带宽极限。

---

## 研究动机

### 1. MoE 模型通信瓶颈

现代大规模语言模型（如 DeepSeek-V3）广泛采用 MoE 架构，通过稀疏激活的专家网络提升模型容量。然而，MoE 的 **all-to-all 通信**（dispatch 和 combine）成为训练和推理的主要性能瓶颈：

- **吞吐量需求**：token 需要从所有 rank 路由到对应的专家，再将结果分发回去，通信量大。
- **延迟敏感**：推理场景（尤其是 decoding）对延迟极度敏感，需要低延迟的通信原语。
- **SM 资源竞争**：传统通信内核占用大量 SM（Streaming Multiprocessor），与计算内核竞争 GPU 资源。

### 2. 现有方案的局限

- **NCCL**：通用集合通信库，未针对 MoE 的 all-to-all 模式优化。
- **NVSHMEM**：提供对称内存访问，但 API 复杂，SM 占用高，配置复杂。
- **自定义内核**：虽然性能好，但缺乏通用性和易用性。

### 3. DeepSeek 的需求

作为大规模 MoE 模型（如 DeepSeek-V3、DeepSeek-R1）的开发者，DeepSeek 需要一个：
- 高性能、低延迟的 EP 通信库
- 轻量级、易集成的解决方案
- 支持 FP8 低精度通信
- 最小化 SM 占用，为计算留出空间

---

## 方法（技术细节）

### 1. ElasticBuffer 统一接口

V2 版本的核心创新是将高吞吐和低延迟的 EP 操作统一到 **`ElasticBuffer`** 接口中：

- **统一 API**：dispatch 和 combine 操作共用同一缓冲区接口
- **自动参数计算**：根据 MoE 设置（token 数、hidden 维度、top-k、专家数）自动计算最优 SM 和 QP 数量，无需手动调优
- **可扩展**：支持 up to EP2048 的大规模配置

```python
buffer = ElasticBuffer(
    group,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=use_fp8_dispatch,
)
num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)
```

### 2. NCCL Gin 后端

V2 从 NVSHMEM 后端切换到 **NCCL Gin 后端**：

- **轻量级**：Header-only 设计，依赖少
- **可复用**：能复用现有 NCCL communicator，避免额外初始化开销
- **兼容性好**：基于 NCCL 生态，易于集成

### 3. JIT 编译

所有内核通过 **Just-In-Time (JIT)** 编译：

- 运行时编译，无需预编译 CUDA 代码
- 安装时无需 CUDA 编译环境
- 编译结果缓存（`~/.deep_ep`），后续运行可复用

### 4. 通信-计算重叠（Overlap）

DeepEP 支持通信与计算的重叠：

- **异步操作**：`async_with_compute_stream=True` 启用异步通信
- **EventOverlap 接口**：管理通信流与计算流之间的依赖关系
- **推理场景**：支持 handle 缓存，避免重复 CPU 同步

```python
# 通信进行中时执行计算
recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch_forward(...)
# ... 独立计算 ...
event.current_stream_wait()  # 等待通信完成
```

### 5. 支持的并行模式

- **专家并行（EP）**：核心功能，支持高吞吐和低延迟两种模式
- **流水线并行（PP）**：实验性功能，支持 RDMA，0 SM 占用
- **上下文并行（CP）**：实验性功能，使用 Copy Engine，0 SM 占用
- **远程内存访问（Engram）**：实验性功能，支持 RDMA，0 SM 占用

### 6. 低精度支持

- **FP8 Dispatch**：支持 FP8 低精度 dispatch，减少通信量
- **BF16 Combine**：支持 BF16 combine 操作
- **NVFP4**：Hybrid-EP 实验分支支持 NVFP4 数据类型

### 7. SM 资源优化

V2 相比 V1 的关键改进：

- **SM 数量大幅减少**：从 V1 的 24 SM 降至 4-6 SM
- **性能提升**：峰值性能提升最高 1.3x
- **SM 节省**：最多节省 4x SM 资源
- **解析式计算**：无需自动调优，自动计算最优 SM 和 QP 数量

### 8. 网络配置

- **InfiniBand 支持**：完全测试，支持 RDMA
- **RoCE 兼容**：理论上兼容
- **流量隔离**：通过 Virtual Lane (VL) 支持
- **自适应路由**：推荐在所有网络负载条件下启用
- **拥塞控制**：默认禁用（影响最大带宽）

---

## 实验结果

### 基准测试配置

- 8K tokens/batch
- 7168 hidden 维度
- top 8 专家
- FP8 dispatch + BF16 combine

### 性能数据

| 架构 | NIC 类型 | 拓扑 | Dispatch 带宽 | Combine 带宽 | SM 数 |
|------|----------|------|--------------|-------------|-------|
| SM90 | CX7 | EP 8×2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8×4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8×2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (最大性能) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (最小 SM) |

**关键结论**：
- V2 相比 V1 峰值性能提升最高 1.3x
- V2 相比 V1 最多节省 4x SM 资源
- 在 SM90 架构上，RDMA 带宽可达 90 GB/s
- 在 SM100 架构上，NVLink 带宽可达 740 GB/s
- 性能达到或超过硬件带宽极限

---

## 优势

1. **极致性能**：接近或超过硬件带宽上限，在各种配置下表现出色
2. **SM 资源节约**：V2 将 SM 使用从 24 降至 4-6，为计算留出更多空间
3. **轻量级设计**：JIT 编译，无需预编译，安装简单
4. **统一接口**：高吞吐和低延迟操作统一到 ElasticBuffer，API 简洁
5. **自动调优**：解析式计算 SM 和 QP 数量，无需手动调优
6. **低精度支持**：FP8 dispatch + BF16 combine，减少通信量
7. **通信-计算重叠**：支持异步操作，最大化 GPU 利用率
8. **开源社区**：MIT 许可，活跃的社区贡献（多个实验分支和社区 fork）
9. **推理优化**：支持 handle 缓存，避免重复 CPU 同步，适合 decoding 场景
10. **可扩展**：支持 EP2048 大规模配置，适用于超大规模 MoE 模型

---

## 局限

1. **仅支持 Hopper 架构**：需要 SM90 或更高架构，不支持老 GPU
2. **V2 缓冲区消耗更大**：相比 V1，buffer size 消耗增加
3. **部分功能为实验性**：Engram、PP、CP 仍为实验性功能
4. **0 SM RDMA 低延迟 EP 不再支持**：V2 不再支持 RDMA 低延迟 EP 的 0 SM 模式
5. **网络依赖**：需要 InfiniBand 或 RDMA 网络支持，部署门槛较高
6. **无独立论文**：仅为 GitHub 开源项目，缺乏正式的学术论文和理论分析
7. **CUDA 锁定**：仅支持 NVIDIA GPU，不支持 AMD GPU（虽然有实验性 ROCm 支持）
8. **版本兼容性**：需要特定的 NCCL 版本（>=2.30.4）和 PyTorch 版本（>=2.10）

---

## 与 EfficientPaper 相关的研究方向

### 1. 通信-计算重叠（Overlap）
DeepEP 的核心优化之一是通信与计算的重叠，这与 EfficientPaper 中 "overlap" 关键词直接相关。该技术通过异步操作和 EventOverlap 接口，最大化 GPU 利用率。

### 2. 专家并行（Expert Parallelism）
DeepEP 专注于 MoE 模型的专家并行通信，是 MoE 训练和推理的关键基础设施。随着 MoE 架构在大规模模型中的普及，EP 通信优化变得越来越重要。

### 3. 低精度通信（FP8/FP4）
DeepEP 支持 FP8 dispatch 和 BF16 combine，与低精度计算趋势一致。低精度通信可以显著减少通信量，提高吞吐量。

### 4. GPU 内核优化
DeepEP 的 JIT 编译和 SM 资源优化展示了 GPU 内核级别的性能优化技术，对高效 AI 系统设计有重要参考价值。

### 5. 集合通信库
DeepEP 作为 NCCL 生态的扩展，展示了如何为特定场景（如 MoE）设计高效的通信库，与 NCCL、UCCL 等研究方向相关。

### 6. 分布式训练/推理系统
DeepEP 是分布式训练和推理系统的关键组件，与大规模 AI 系统的可扩展性和效率优化密切相关。

### 7. 0 SM 通信
DeepEP 的 0 SM 设计（PP、CP、Engram）展示了如何将通信开销降至最低，为计算留出全部 GPU 资源，是高效系统设计的重要方向。

---

## 相关论文与项目

- **DeepSeek-V3**：使用 DeepEP 的 MoE 模型
- **NCCL**：DeepEP V2 的 NCCL Gin 后端
- **NVSHMEM**：DeepEP V1 的通信后端
- **UCCL-EP**：社区 fork，支持异构 GPU
- **MORI**：AMD GPU 的通信库，支持 DeepEP

---

## 总结

DeepEP 是一个针对 MoE 模型专家并行通信的高性能库，通过 ElasticBuffer 统一接口、NCCL Gin 后端、JIT 编译和通信-计算重叠等技术，实现了接近硬件带宽上限的通信性能，同时显著减少 SM 资源占用。尽管缺乏正式的学术论文，但其开源实现和性能数据展示了高效的 GPU 通信内核设计，对大规模 MoE 模型的训练和推理具有重要价值。

---

> **声明**：本 note 由 AI Agent 自动生成，基于 DeepEP GitHub 仓库的官方 README 和技术文档。由于 DeepEP 为 GitHub 开源项目（无独立 arXiv 论文），本 note 内容主要来源于项目官方文档，未包含独立论文的实验分析。生成时间：2025 年 6 月。
