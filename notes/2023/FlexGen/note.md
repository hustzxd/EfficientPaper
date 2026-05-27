# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

> Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

LLM 推理的高计算和内存需求使其通常只能在多个高端加速器上运行。本文面向延迟不敏感的批处理任务（如信息提取、数据清洗、基准测试），研究如何在有限资源（如单个消费级 GPU）上实现高吞吐 LLM 推理。FlexGen 通过聚合 GPU、CPU 和磁盘的内存与计算资源，用线性规划搜索最优的张量存储和访问策略，并将权重和注意力缓存压缩到 4-bit，在 OPT-175B 上实现了单 T4 GPU 上首次达到 1 token/s 的生成吞吐。

## 一句话总结

通过线性规划搜索最优的 GPU/CPU/Disk 三级卸载策略（zig-zag block schedule + I/O overlapping）并结合 4-bit 量化，FlexGen 在单个消费级 GPU 上实现了比 DeepSpeed/Accelerate 高 100 倍的 OPT-175B 推理吞吐。

## 背景与问题

### 吞吐导向推理场景

除了交互式聊天场景，LLM 还广泛用于"后台"批处理任务：基准测试、信息提取、数据清洗、表单处理等。这些任务的关键特征是：
- 需要对大量 token 批量推理
- 对延迟不敏感
- 可以用延迟换取更高吞吐

### 现有系统的局限

1. **模型压缩方向**：假设模型能放入 GPU 显存，无法在单个消费级 GPU 上运行 175B 模型
2. **协作推理方向**：分布式网络延迟和带宽限制性能
3. **卸载方向**（DeepSpeed Zero-Inference、HuggingFace Accelerate）：
   - 继承自训练系统的卸载策略，未针对推理特性优化
   - I/O 调度效率低，batch size 极小（OPT-175B 仅 1-2）
   - 无法充分利用三级存储层次的带宽

## 核心方法

### 1. 卸载策略搜索空间

FlexGen 将推理建模为**计算图遍历问题**。计算图是一个二维网格：横轴是 batch 维度（无限数据集），纵轴是 layer 维度。目标是找到一条遍历所有节点的有效路径，最小化总执行时间（计算 + I/O）。

**三个关键决策维度：**

#### 计算调度（Compute Schedule）
- **逐行调度**（row-by-row）：现有系统默认方式，每次处理一个 batch 的所有层。问题：相邻行不共享权重，导致反复加载权重，I/O 开销巨大。
- **Zig-zag Block 调度**：先沿列方向遍历（同一层的多个 batch 共享权重），当激活/KV Cache 满内存时再切换方向。I/O 复杂度证明在最优解的 2x 以内。

#### 张量放置（Tensor Placement）
用 9 个变量定义权重(w)、激活(h)、KV Cache(c) 在 GPU(g)、CPU(c)、Disk(d) 上的分配比例：
- wg, wc, wd：权重放置比例
- hg, hc, hd：激活放置比例
- cg, cc, cd：KV Cache 放置比例

粒度从粗到细：模型级 → 层级 → 级别内的张量头

#### 计算委托（Computation Delegation）
指定在哪个设备上执行计算（GPU 或 CPU）。

### 2. 线性规划优化器

FlexGen 构建了一个**解析代价模型**，将搜索空间编码为线性规划问题：
- 决策变量：上述 9 个放置比例 + block size 等
- 目标函数：最大化吞吐（tokens/s）
- 约束：各设备内存容量限制

该优化器可根据不同硬件规格灵活配置，也可扩展延迟和吞吐约束。

### 3. I/O 重叠与流水线

Block Schedule 的内层循环中，6 个操作可以并行执行：
1. 加载下一层权重
2. 存储上一个 batch 的激活
3. 存储上一个 batch 的 KV Cache
4. 加载下一个 batch 的 KV Cache
5. 加载下一个 batch 的激活
6. 计算当前 batch

最后同步所有设备。依赖操作系统和 CUDA 驱动调度底层硬件资源。

### 4. 4-bit 量化

FlexGen 对权重和 KV Cache 均采用 **fine-grained group-wise 量化**到 4-bit：
- 无需重训练或校准
- 精度损失可忽略
- 显著减少 I/O 传输量和内存占用
- 使得权重可以完全放在 CPU 内存中，避免磁盘卸载

## 技术细节

### 理论保证

**定理 4.1**：Zig-zag block schedule 的 I/O 复杂度在最优解的 2x 以内。

### 内存分析（OPT-175B, FP16）
- 模型权重：325 GB（l=96, h1=12288, h2=49152）
- KV Cache 峰值（b=512, s=512, n=32）：1.2 TB（权重的 3.8 倍）
- 大 batch 下 KV Cache 成为新瓶颈

### 实现细节
- 基于 PyTorch 实现
- 支持 GPU 计算委托（CUDA kernel）和 CPU 计算委托
- Pipeline 并行支持多 GPU 场景

## 实验设置

### 硬件
- NVIDIA T4 (16GB GPU) — 消费级 GPU
- 208 GB CPU DRAM
- 1.5 TB SSD

### 模型
- OPT-30B, OPT-66B, OPT-175B

### Baseline
- DeepSpeed Zero-Inference
- HuggingFace Accelerate
- Petals（协作推理）

### 配置
- 输入序列长度：512
- 输出序列长度：32

## 主要结果

### OPT-175B 单 T4 GPU
- **DeepSpeed/Accelerate**：batch size 最多 2，因 OOM 无法扩大
- **FlexGen**：
  - 同延迟(5000s)下，吞吐比 DeepSpeed 高 **40x**（batch 64 vs 1）
  - 允许更高延迟(12000s)时，最大吞吐高 **69x**（batch 256）
  - 启用 4-bit 压缩后，最大吞吐高 **100x**（batch 144，4000s）

### OPT-30B
- FlexGen 在 Pareto 前沿上全面优于 DeepSpeed 和 Accelerate

### HELM Benchmark
- FlexGen 可在单 16GB GPU 上 21 小时内完成 30B 模型的 7 个代表性子场景基准测试

### 与 Petals 对比
- FlexGen 在单 GPU 吞吐上优于 Petals 集群
- 某些场景下延迟也更低

## 优点与局限

### 优点
1. **形式化建模**：将卸载策略搜索建模为线性规划问题，有理论保证（2x 最优性）
2. **灵活配置**：自动适配不同硬件规格（GPU/CPU/Disk 容量和带宽）
3. **实用性强**：首次在单消费级 GPU 上运行 OPT-175B 达到 1 token/s
4. **量化 + 卸载协同**：4-bit 量化使权重可完全放在 CPU，避免磁盘 I/O
5. **开源**：代码完整可用

### 局限
1. **吞吐导向**：牺牲延迟换取吞吐，不适合交互式场景
2. **模型范围**：主要在 OPT 系列上验证，未覆盖 GQA 架构（如 Llama/Qwen）
3. **4-bit 精度**：虽声称精度损失可忽略，但在复杂推理任务上的影响未充分评估
4. **单 GPU 限制**：多 GPU 场景仅简单扩展，未深入优化
5. **已过时**：后续工作（如 vLLM、SGLang）在延迟导向场景上大幅超越；FlexGen 的卸载思路被 ContiguousKV 等新工作继承和发展

## 与 EfficientPaper 主题的关系

FlexGen 属于 **部署/Serving**（deployment）领域，是 LLM 推理卸载（offloading）方向的奠基性工作。核心贡献：
- 首次系统性地研究 GPU/CPU/Disk 三级存储层次下的 LLM 推理卸载策略
- 线性规划驱动的自动策略搜索
- 证明了吞吐导向场景下通过大 batch + I/O overlapping 可以在有限资源上运行超大模型

对后续工作的影响：ContiguousKV 直接基于 FlexGen 框架构建，将其卸载策略扩展到共享前缀 KV Cache 的 Re-Prefill 场景。

## 可复现/实现要点

1. **框架**：PyTorch 实现，开源可用
2. **关键参数**：
   - GPU batch size
   - Block size（= GPU batch size × GPU batches per block）
   - 9 个张量放置比例（权重/激活/KV Cache × GPU/CPU/Disk）
3. **优化器**：线性规划求解最优配置
4. **量化**：group-wise 4-bit，无需校准

## 个人备注

- FlexGen 的 LP 搜索框架思路优雅，但实际部署中手动调参可能更实用
- 后续的 KV Cache 卸载工作（AttentionStore、IMPRESS、ContiguousKV）都继承了 FlexGen 的三级存储层次思想
- 对比 vLLM 的 paged attention，FlexGen 的 block schedule 更适合静态批处理场景
