# ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention

> Joe Sharratt

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Efficient attention algorithms are critical to mitigate the quadratic cost of attention in long-context workloads. Prior work utilises block-scaled quantisation techniques on Blackwell GPUs to move attention computation to 4-bit precision to accelerate inference. However, these techniques result in significant quality degradation in long-context settings. We show that the output impact of quantisation error is highly non-uniform and increases with the importance of each query-key interaction, concentrating functionally relevant error in a small number of attention blocks that contain the most important tokens. We propose ThriftAttention, a low-bit attention variant that delivers near-FP16 long-context quality at FP4 inference efficiency. This approach proceeds in two stages. First, a heuristic rapidly selects a small number of important query-key block pairs for FP16 precision. Second, the selected blocks are computed in FP16 and the remaining blocks in FP4, with both paths merged via online softmax into a single output. We demonstrate across long-context benchmarks and model families that by computing only 5% of query-key blocks in FP16, ThriftAttention recovers on average 89.1% of the FP4-to-FP16 performance gap. We show ThriftAttention's advantage grows with sequence length, mitigating the systematic FP4 quality degradation observed at longer contexts.

## 一句话总结

ThriftAttention 是一种 training-free 的混合精度 attention 机制，在 Blackwell GPU 上将 95% 的 QK block pair 用 FP4 计算、仅对最重要的 5% 用 FP16，通过 online softmax 合并，以接近 FP4 的推理效率恢复近 FP16 的长上下文质量（平均恢复 89.1% 的 FP4→FP16 性能差距）。

## 背景与问题

NVIDIA Blackwell 架构引入原生 FP4 Tensor Core，算力是 FP16 的 4 倍。SageAttention3 等工作利用这一硬件将 attention 计算迁移到 FP4 精度以加速推理。但在长上下文场景下，均匀 FP4 attention 导致严重的质量退化——随着序列长度增加，per-token 量化误差在更多位置上累积，质量退化系统性恶化。

**关键观察：FP4 量化误差的影响是高度非均匀的。**

从一阶扰动分析出发，attention 输出的量化误差可分解为：
```
||δo|| ≤ Σ_j |ε_j| · p_j · ||v_j - o||
```
其中 ε_j 是 score 量化误差，p_j 是 attention weight，||v_j - o|| 是 value 偏差。p_j 因子使得高 attention score 的 token 的量化误差被 softmax 指数放大——**功能相关的量化误差集中在少数包含最重要 token 的 attention block 中**。

这启发了一个简单思路：只对这些高误差 block 用 FP16，其余用 FP4。

## 核心方法

ThriftAttention 是一个两阶段的混合精度 attention 算法：

### Stage 1: Block-Importance Scoring

将 Q 分为 T_q = N/B_q 个 block，K/V 分为 T_k = N/B_k 个 block。对每个 block 计算 token 均值，然后用均值的内积作为 block pair 重要性分数：

```
S̃_ij = Q̄_i · K̄_j^T
```

这是一个极轻量的启发式——只需对 block 均值做矩阵乘法，复杂度远低于完整 QK^T。

### Stage 2: Mixed-Precision Attention

对每个 query block i，选取 top-k 个最重要的 key block 进入 FP16 路径，其余进入 FP4 路径：

- **FP16 路径**（j ∈ T_i）：标准 FP16 矩阵乘 S_ij = Q_i K_j^T / √d，follow FlashAttention-2 online softmax
- **FP4 路径**（j ∉ T_i）：用 NVFP4 microscaling 格式（E2M1 + E4M3 per-group scale），probability block 也用 SageAttention3 的两级量化方案

两条路径通过 online softmax 在线合并为统一输出。关键设计：**所有 block 都被计算**（只是精度不同），不像 sparse 方法直接丢弃 block，因此误差上界是 FP4 量化噪声而非完全丢失。

### 实现优化

- 单一 fused CUDA kernel，FP16 query fragment 仅在 selected-block 阶段加载到寄存器
- 共享内存 region 在 K/V tile 的 FP16 和 FP4 路径间复用
- FP4 KV tile 使用 double-buffering 隐藏内存加载延迟
- 不包含 top-k block 的 warps/CTA 完全跳过 FP16 路径

## 技术细节

### 量化格式

使用 NVFP4 microscaling 格式（Blackwell 原生支持）：
- 元素：E2M1 格式（1 符号 + 2 指数 + 1 尾数）
- Per-group scale：E4M3 格式（FP8）
- Group size = 16
- Q、K、V 独立量化

### 因果掩码

对 causal LLM attention，block 选择限制在因果可见的 key block 范围内。

### KV Cache 内存开销

论文提到 ThriftAttention 增加 28% 的 KV cache 内存占用（同时存储 FP16 和 FP4 cache），这是一个 trade-off。

## 实验设置

- **硬件**：RTX PRO 6000 (Blackwell)
- **模型**：Llama3.2-3B, Llama3.1-8B, Qwen3-4B, Qwen3-8B, Ministral3-8B
- **基准**：LongBench-v1, HELMET, RULER, PG-19
- **Baseline**：FlashAttention-2 (FP16), SageAttention3 (FP4), Quest (sparse), Sparse Top-k
- **Block size**：B_q = B_k = 64
- **FP16 budget**：5%, 10%, 25%

## 主要结果

### 质量恢复

| FP16 Budget | 平均恢复率（FP4→FP16 差距） |
|-------------|--------------------------|
| 5% | 89.1% |
| 10% | 91.8% |
| 25% | 92.4% |

边际收益递减明显——5% 已恢复绝大部分差距。

### 推理效率

- **Prefill kernel**：最高 1.7× 加速（vs FlashAttention-2）
- **Decode kernel**：3×–5.5× 加速（vs FlashAttention-2），接近纯 FP4 延迟
- **端到端 decode**：131k 上下文下 ~2× 加速（vs FP16）

### 长上下文优势

ThriftAttention 的优势随序列长度增长而扩大：
- 8k：FP4 保留 50% 质量，ThriftAttention ~2× FP4
- 131k：FP4 仅保留 32% 质量，ThriftAttention ~2.2× FP4

NLL 分析显示，128k 序列末尾 token 的 FP4 ΔNLL 高达 0.10，ThriftAttention 将其降至 ≤0.02（~5× 改善）。

### vs Sparse Attention（matched compute）

在等效 FLOP 预算下（ThriftAttention 5% FP16 = Sparse 28.7% FP16）：
- ThriftAttention：0.599
- Sparse Top-k：0.036
- Quest：0.142

核心差异：sparse 方法完全丢弃 block 导致不可逆信息丢失，ThriftAttention 保留所有 block 在低精度，退化更平滑。

## 优点与局限

**优点：**
- Training-free，无需模型修改或重训练
- 仅 5% FP16 block 即可恢复 ~89% 质量差距
- 优势随序列长度增长而扩大，正中长上下文推理痛点
- 相比 sparse 方法在 matched compute 下大幅领先
- 已开源 CUDA kernel

**局限：**
- KV cache 内存增加 28%（同时存 FP16 和 FP4）
- 目前仅针对 Blackwell GPU（consumer 级），data-center Blackwell (SM100) 的扩展待实现
- 当前仅用于推理加速，训练场景待探索
- Block importance 评分使用 token 均值内积，是近似启发式而非精确重要性

## 与 EfficientPaper 主题的关系

ThriftAttention 属于 **quantization** 领域，具体聚焦于 **attention 计算的混合精度量化**。与 SageAttention3（均匀 FP4 attention）直接对比，也与 Quest 等 sparse attention 方法形成互补视角：与其丢弃 block 不如保留低精度版本。这为 EfficientPaper 的量化和 attention 优化方向提供了一个新的混合精度范式。

## 可复现/实现要点

1. **硬件要求**：Blackwell GPU（RTX PRO 6000 或更高）
2. **量化格式**：NVFP4（E2M1 + E4M3 microscale），Blackwell 原生支持
3. **Block size**：B_q = B_k = 64
4. **Block importance**：token 均值内积 Q̄_i · K̄_j^T，top-k 选择
5. **Kernel**：单一 fused CUDA kernel，FP16/FP4 双路径，online softmax 合并
6. **代码**：https://github.com/joesharratt1229/ThriftAttention
7. **评估**：LongBench-v1, HELMET, RULER, PG-19，使用 NLL 和 task-specific metrics

## 个人备注

- 与 VECTOR 的思路有概念相似性：两者都识别出"重要"交互需要更高精度/更多信息，但 VECTOR 在 KV cache 层面做 token 级分配，ThriftAttention 在 attention 计算层面做 block 级精度分配
- Block importance 用均值内积是极简启发式，是否可以用 attention sink pattern 或 pre-RoPE Q/K concentration 做更精确的选择？
- 28% KV cache 内存增加是一个实际 trade-off，在内存受限场景下需要权衡
- 与 FlashAttention-4 的关系值得探索：FA4 是异构流水线，ThriftAttention 是混合精度，两者是否可组合？
- 对 Blackwell data-center GPU (SM100) 的扩展可能会带来更大加速（更高的 FP4 吞吐）
