# Serving Large Language Models on Huawei CloudMatrix384

> Pengfei Zuo, Huimin Lin, Junbo Deng, Nan Zou, Xingkun Yang, Yingyu Diao, Weifeng Gao, Ke Xu, Zhangyu Chen, Shirui Lu, Zhao Qiu, Peiyang Li, Xianyu Chang, Zhengzhong Yu, Fangzheng Miao, Jia Zheng, Ying Li, Yuan Feng, Bei Wang, Zaijian Zong, Mosong Zhou, Wenli Zhou, Houjiang Chen, Xingyu Liao, Yipeng Li, Wenxiao Zhang, Ping Zhu, Yinggang Wang, Chuanjie Xiao, Depeng Liang, Dong Cao, Juncheng Liu, Yongqiang Yang, Xiaolong Bai, Yi Li, Huaguo Xie, Huatao Wu, Zhibin Yu, Lv Chen, Hu Liu, Yujun Ding, Haipei Zhu, Jing Xia, Yi Xiong, Zhou Yu, Heng Liao

![111](../../blank.jpg)

> **⚠️ 生成声明**：本 note 由 AI Agent 自动生成，基于对论文全文的阅读和理解。生成时间：2025年6月。所有内容为中文总结与翻译，仅供参考，建议结合原文阅读。

---

## 一句话总结

华为提出了 CloudMatrix384 超级节点架构及 CloudMatrix-Infer 推理方案，通过 384 个昇腾 910 NPU 全对等互联（UB 网络）+ PDC 分离式服务架构 + 大规模专家并行（EP320）+ 微批次流水线 + INT8 量化，在 DeepSeek-R1 上实现 prefill 6,688 tokens/s/NPU 和 decode 1,943 tokens/s/NPU 的 SOTA 效率，同时保持 16 个基准测试的精度不降。

---

## 摘要翻译

大语言模型（LLM）的快速发展——参数规模增长、MoE 架构采用、上下文长度扩展——对 AI 基础设施提出了前所未有的需求。传统 AI 集群在计算强度、内存带宽、芯片间通信和延迟方面面临瓶颈。本文介绍了华为 CloudMatrix，一种下一代 AI 数据中心架构，以生产级 CloudMatrix384 超级节点实现。它集成了 384 个昇腾 910 NPU 和 192 个鲲鹏 CPU，通过超高带宽统一总线（UB）网络互联，支持直接全对等通信和动态资源池化。为充分利用 CloudMatrix384，本文提出 CloudMatrix-Infer，包含三大核心创新：（1）将 prefill、decode 和缓存解耦为独立可扩展资源池的对等服务架构；（2）基于 UB 网络高效 token 调度的 EP320 大规模专家并行策略；（3）硬件感知优化，包括专用算子、微批次流水线和 INT8 量化。基于 DeepSeek-R1 的评估显示：prefill 吞吐量 6,688 tokens/s/NPU，decode 吞吐量 1,943 tokens/s/NPU（TPOT<50ms），在 15ms 延迟约束下仍可维持 538 tokens/s/NPU，INT8 量化在 16 个基准上保持精度。

---

## 研究动机

### LLM 发展趋势带来的挑战

1. **参数规模持续膨胀**：DeepSeek-R1 671B、Llama 4 Behemoth ~2T 等模型对算力和内存提出巨大需求。
2. **MoE 架构广泛采用**：如 DeepSeek-V3 256 个路由专家，每 token 仅激活 37B 参数。MoE 带来跨 NPU 的频繁通信需求（token dispatch 和 expert output combination）。
3. **上下文窗口大幅扩展**：从数万到百万 token，KV cache 存储和访问成为瓶颈。
4. **生产环境的动态性和异构性**：变长输入、突发请求、不均衡的专家激活，需要严格的 SLO 满足。

### 传统架构的四大挑战

1. **通信密集型并行的扩展瓶颈**：TP/EP 需要频繁细粒度低延迟通信，传统 RDMA 网络跨节点通信带宽不足。
2. **异构负载下的资源利用率低**：固定节点配置无法高效适应训练（计算密集）和推理（内存带宽受限）等不同工作负载。
3. **AI 与数据密集型工作负载的融合执行**：需要高吞吐低延迟通信和灵活资源编排。
4. **存储的内存级性能需求**：KV cache、RAG 模块等需要内存级带宽和延迟，传统存储成为瓶颈。

---

## 方法（技术细节）

### 1. CloudMatrix384 硬件架构

**核心设计原则**：全对等（peer-to-peer）高带宽互联 + 细粒度资源解耦。

**硬件规格**：
- 384 个昇腾 910 NPU（每个节点 8 个 NPU，48 个节点）
- 192 个鲲鹏 CPU（每个节点 4 个 CPU）
- 通过统一总线（UB）网络互联，实现全对等、非阻塞、all-to-all 拓扑
- 节点间带宽衰减 <3%，延迟增加 <1μs

**三大网络平面**：
- **UB 平面**（主要 scale-up）：超高速、非阻塞全对等互联，支持 TP/EP 跨节点扩展、快速对等内存访问
- **RDMA 平面**（scale-out）：跨超级节点通信，基于 RoCE
- **VPC 平面**（数据中心网络）：管理控制、持久化存储访问

**昇腾 910 芯片**：
- 双 die 封装，每个 die 24 个 AI Cube（AIC）核 + 48 个 AI Vector（AIV）核
- 支持 FP16/BF16 和 INT8 计算
- 8 个 on-package 内存栈
- 每个 die 有 7 个高速收发器连接 UB 平面 + 1 个 RDMA 接口

**UB Switch 系统**：
- 12 个计算机架 + 4 个通信机架
- 两级交换（L1 机板级 + L2 机架级），7 个独立子平面
- 非阻塞设计，L2 不过订阅

### 2. CloudMatrix-Infer 服务架构

**核心创新：PDC（Prefill-Decode-Caching）分离式对等服务架构**

将 LLM 推理分解为三个独立子系统：
- **Prefill 集群**：处理输入 prompt，生成首个输出 token 和初始 KV cache。每个实例 16 NPU（32 dies），EP32 并行。
- **Decode 集群**：自回归生成后续 token。每个实例 160 NPU（320 dies），EP320 并行。每个 NPU die 仅承载 1 个专家，降低 MoE 执行延迟。
- **Caching 集群**：基于解耦内存池的 UB 连接缓存层，提供上下文缓存（KV cache 复用）和模型缓存（加速模型加载）。

**与 KVCache-centric 架构（如 NVIDIA Dynamo、Mooncake）的对比**：
- KVCache-centric：请求调度与 KV cache 局域性紧密耦合，需要亲和性感知调度，复杂度高。
- 对等架构（本方案）：所有 NPU 可均匀访问共享解耦内存池，调度与数据局域性解耦，实现轻量无状态调度，简化调度逻辑，提升缓存效率和资源利用率。

### 3. 大规模专家并行（LEP）+ 融合通信算子

**关键设计**：
- **EP320**：decode 阶段支持 320 路专家并行，每个 NPU die 承载 1 个专家，包含 32 个共享专家 + 256 个路由专家 + 32 个冗余路由专家（用于 EPLB 负载均衡）。
- **FusedDispatch 和 FusedCombine**：
  - 用 send-receive 原语替代 all-to-all 通信
  - **AIV-Direct**：AIV 核直接写入远端 NPU 内存，绕过 SDMA，消除启动开销
  - **早期量化**：在 dispatch 阶段将 BF16 数据量化为 INT8（7.5KB/token），减少通信量
  - **静态预分配共享内存**：避免动态内存分配和 CPU-NPU 同步开销，双缓冲避免竞争
  - **数据发送流水线**：将 copy → 量化/计算偏移 → 远端写入 三阶段流水化

### 4. MLA 优化

- **MLAProlog**：融合 RMSNorm、Q/K/V 投影、RoPE 等多个小算子为单一算子，支持 AIC-AIV 并行流水
- **Fused Attention（FA）**：融合 FlashAttention 与相邻数据整形操作
- **NZ 格式 KV Cache**：原生以 NZ 格式存储 KV cache，避免格式转换开销
- **MTP 感知分块（BSND 布局）**：从 BNSD 切换到 BSND，动态沿 B 和 S 轴分块，恢复 MTP 下的负载均衡

### 5. 微批次流水线

**Decode 阶段**：
- 两个交错执行流（Stream 0: Attention 路径，16 AIC + 32 AIV；Stream 1: MoE 路径，8 AIC + 16 AIV）
- 不对称资源分配使两流延迟平衡（~600μs），实现微批次完美重叠
- 延迟增益 5.8%-9.4%（相比无微批次），因 UB 平面本身 MoE 通信开销低，增益上限受限

**Prefill 阶段**：
- 微批次流水线将轻量计算任务卸载到 AIV，SDMA 引擎处理批量数据传输
- 吞吐量提升 23%-31%，每层延迟降低约 24%

### 6. 多 Token 预测（MTP）支持

- DeepSeek-R1 的推测解码技术，每步预测 k 个 token 并验证
- 解决 MTP 导致的 pipeline break 问题（每个图调度引入 0.6-0.8ms 启动延迟）
- 启用 MTP 后 decode 吞吐量提升 6%-49%（在 70% 接受率下，每步平均 1.7 token）

### 7. UB 驱动的分布式缓存（EMS）

- **解耦内存池**：所有计算节点的 CPU DRAM 通过 UB 网络形成统一内存池
- **上下文缓存**：复用历史 KV cache 块，50% 复用率可使 prefill 吞吐量提升 1.42×，90% 提升 2.28×
- **模型缓存**：加速模型块加载，减少冷启动延迟
- UB 平面相比 VPC 平面缓存访问性能提升 1.52×，TTFT 在 50% 复用率下降低 861ms（34%）

### 8. INT8 量化

- 在昇腾 910 上实现 INT8 量化，通过 INT8 精度实现接近 FP8 的计算效率
- 量化模型权重和激活值，减少内存占用、计算开销和内存带宽需求

---

## 实验结果

### 实验环境

- 华为 CloudMatrix384 超级节点（ModelArts Lite 集群模式）
- 256 个昇腾 910 NPU + 鲲鹏 CPU
- DeepSeek-R1 671B 模型，INT8 量化

### 整体性能

#### Prefill 吞吐量

| 方法 | Batch | Input Length | Throughput (tokens/s) | Throughput/TFLOPS |
|------|-------|-------------|----------------------|-------------------|
| DeepSeek (H800 Blog) | N/A | N/A | 4,026 | 2.03 |
| SGLang (H100 Default) | 16,384 | 4,096 | 6,288 | 3.18 |
| **CloudMatrix-Infer (Default)** | 16,384 | 4,096 | 5,655 | **3.76** |
| DeepSeek (H800 Profile) | 16,384 | 4,096 | 7,839 | 3.96 |
| SGLang (H100 Perfect EPLB) | 16,384 | 4,096 | 7,417 | 3.75 |
| **CloudMatrix-Infer (Perfect EPLB)** | 16,384 | 4,096 | **6,688** | **4.45** |

#### Decode 吞吐量（TPOT <50ms）

| 方法 | Batch | KV Cache | TPOT (ms) | Throughput (tokens/s) | Throughput/TFLOPS |
|------|-------|----------|-----------|----------------------|-------------------|
| DeepSeek (H800 Blog) | N/A | 4,989 | ~50.0 | 1,850 | 0.93 |
| DeepSeek (H800 Profile) | 128 | 4,096 | ~50.2 | 2,325 | 1.17 |
| SGLang (H100 Simu. MTP) | 128 | 4,000 | ~55.6 | 2,172 | 1.10 |
| **CloudMatrix-Infer** | 96 | 4,096 | 49.4 | 1,943 | **1.29** |

#### 不同 TPOT SLO 下的 Decode 吞吐

| TPOT SLO (ms) | Prompt Length | Output Length | Batch | TPOT (ms) | Throughput (tokens/s/NPU) |
|---------------|--------------|--------------|-------|-----------|--------------------------|
| 50 | 1,024 | 1,024 | 128 | 46.8 | 2,733 |
| 50 | 2,048 | 256 | 112 | 47.4 | 2,360 |
| 50 | 4,096 | 256 | 96 | 49.4 | 1,943 |
| 30 | 4,096 | 256 | 24 | 24.6 | 974 |
| 15 | 4,096 | 256 | 8 | 14.9 | 538 |

### 精度评估（INT8 量化）

在 16 个基准测试上与 DeepSeek-R1 API 和官方报告对比，INT8 量化版本精度基本持平：

| 基准 | DeepSeek-R1 (INT8) | DeepSeek-R1 API | DeepSeek-R1 Report |
|------|--------------------|-----------------|--------------------|
| MMLU (Pass@1) | 90.82 | 91.05 | 90.8 |
| MMLU-Pro (EM) | 83.91 | 83.82 | 84.0 |
| DROP (3-shot F1) | 90.42 | 91.02 | 92.2 |
| GPQA Diamond (Pass@1) | 71.66 | 71.77 | 71.5 |
| AIME 2024 (Pass@1) | 78.96 | 78.12 | 79.8 |
| MATH-500 (Pass@1) | 94.46 | 94.62 | – |
| LiveCodeBench (Pass@1-COT) | 63.80 | 63.44 | 65.9 |
| HumanEval (Pass@1-COT) | 91.83 | 91.85 | – |
| CLUEWSC (Test) | 94.67 | 94.98 | – |
| C-Eval (EM) | 82.05 | 79.92 | – |

### 消融实验

#### 微批次流水线
- Decode：吞吐量提升 5.8%-9.4%，每层延迟降低约 10%
- Prefill：吞吐量提升 23%-31%，每层延迟降低约 24%

#### MTP
- 启用 MTP（1 个推测 token，70% 接受率）后 decode 吞吐量提升 6%-49%
- 每层执行延迟增加约 44%，但净吞吐量仍然提升

#### 上下文缓存
- EMS + UB 在 90% 复用率下 prefill 吞吐量提升 2.28×，TTFT 降低 59%
- UB 平面比 VPC 平面缓存访问性能提升 1.52×

### 通信算子性能（vs DeepSeek DeepEP on H800）

| 操作 | EP=8 | CM384 延迟 | H800 延迟 | CM384 带宽 | H800 带宽 |
|------|------|-----------|----------|-----------|----------|
| Dispatch | 8 | 116μs | 163μs | 71 GB/s | 46 GB/s |
| Dispatch | 256 | 152μs | 194μs | 54 GB/s | 39 GB/s |
| Combine | 8 | 118μs | 318μs | 131 GB/s | 46 GB/s |
| Combine | 256 | 149μs | 360μs | 103 GB/s | 40 GB/s |

### MLA 算子性能

| 指标 | CM384 | H800 |
|------|-------|------|
| 计算利用率（BF16） | 65.4% | 66.7% |
| 内存带宽利用率 | 84.1% | 89.6% |

### INT8 GEMM 性能

- 计算利用率 77.4%-82.7%，内存带宽 195-327 GB/s
- 主要为计算密集型操作，非内存带宽瓶颈

---

## 优势

1. **全对等互联架构**：UB 网络使 384 个 NPU 形成逻辑统一的紧耦合计算实体，消除传统层级架构的带宽不均匀问题。
2. **PDC 分离式服务**：将 prefill/decode/caching 解耦，各子系统独立可扩展，调度逻辑简化，资源利用率提升。
3. **大规模专家并行（EP320）**：每个 NPU die 仅承载 1 个专家，最小化串行执行，降低 MoE decode 延迟。
4. **融合通信算子**：AIV-Direct 通信 + 早期量化 + 静态预分配 + 数据发送流水线，显著降低 MoE 通信延迟。
5. **硬件感知优化**：针对昇腾 910 架构设计的 MLAProlog、FusedAttention、NZ 格式 KV Cache、MTP 感知分块等。
6. **INT8 量化无损精度**：在 16 个基准测试上与官方 API 精度相当。
7. **高效的吞吐-延迟权衡**：在严格 15ms TPOT 约束下仍可维持 538 tokens/s/NPU。
8. **计算效率领先**：prefill 4.45 tokens/s/TFLOPS 和 decode 1.29 tokens/s/TFLOPS 均超越 NVIDIA H100/H800 上的 SGLang 和 DeepSeek 方案。

---

## 局限

1. **平台特定**：方案深度绑定华为 CloudMatrix384 和昇腾 910 硬件，无法直接移植到其他平台。
2. **大规模 EP 的扩展瓶颈**：在大 EP 度数下，CANN EP 实现的有效带宽显著下降（大 EP 度数时的带宽退化）。
3. **微批次增益受限**：因 UB 平面本身 MoE 通信开销低，微批次流水线的延迟隐藏效果相比 NVIDIA 平台（35%）较为有限（5.8%-9.4%）。
4. **评测范围有限**：仅评测了 DeepSeek-R1 模型，未涵盖其他 MoE 模型或稠密模型。
5. **默认 EPLB 不完美**：默认配置下 prefill 吞吐量低于 Perfect EPLB，表明负载均衡算法仍有改进空间。
6. **评测基准未完全覆盖**：排除了 AlpacaEval 2.0（依赖 GPT-4）、Arena-Hard（依赖 GPT-4）和 CodeForces（缺少自动化评估脚本）。
7. **代码未开源**：无公开代码库，可复现性受限。
8. **未来方向的挑战**：VPC 和 RDMA 平面统一、更大规模超级节点、CPU 解耦等需要更多工程实践验证。

---

## 与 EfficientPaper 相关的研究方向

1. **LLM 推理部署优化（deployment）**：本文是该方向的典型代表，涉及推理系统的架构设计、资源调度、性能优化。
2. **MoE 模型的高效推理**：大规模专家并行、通信优化、负载均衡等，与 DeepSeek-R1 等 MoE 模型的部署紧密相关。
3. **Prefill-Decode 分离架构**：与 DistServe（OSDI'24）、Mooncake（FAST'25）等分离式架构研究相关。
4. **KV Cache 管理与上下文缓存**：与 CachedAttention（ATC'24）等缓存优化研究相关。
5. **低精度量化**：INT8 量化与高效推理，与量化技术研究相关。
6. **AI 基础设施/数据中心架构**：与 UB-Mesh 等下一代数据中心网络研究相关。
7. **推测解码（MTP）**：与投机解码和加速生成相关。
8. **硬件-软件协同设计**：昇腾 910 的架构特点与软件优化的协同设计。
