# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

> Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen

![111](../../blank.jpg)

## 一句话总结

SageAttention2 通过对 Q、K 矩阵进行逐线程 INT4 量化、对 $\tilde{P}$ 和 V 矩阵进行 FP8 量化，并结合 Q 矩阵平滑、两阶段累加等精度增强技术，在 RTX4090 上实现对 FlashAttention2 约 3 倍的计算吞吐提升，同时在语言、图像和视频生成任务上保持几乎无损的端到端性能。

---

## 摘要翻译

尽管线性层的量化已被广泛应用，但其在加速注意力计算方面的应用仍然有限。为了在保持精度的前提下进一步提高注意力计算的效率（相比于 SageAttention），我们提出了 SageAttention2，该方法利用显著更快的 4 位矩阵乘法（Matmul）以及额外的精度增强技术。首先，我们提出以硬件友好的线程级粒度将矩阵 $(Q, K)$ 量化为 INT4，并将矩阵 $(\tilde{P}, V)$ 量化为 FP8。其次，我们提出了一种平滑 Q 的方法，以提高 INT4 $QK^\top$ 的精度。第三，我们提出了 $\tilde{P}V$ 的两阶段累加策略，以提高 FP8 $\tilde{P}V$ 的精度。SageAttention2 的每秒运算次数（OPS）在 RTX4090 上分别超越 FlashAttention2 和 xformers 约 3 倍和 4.5 倍。此外，SageAttention2 在 Hopper GPU 上与 FlashAttention3(fp8) 的速度相当，同时提供更高的精度。全面的实验确认，我们的方法在多种模型（包括语言、图像和视频生成）上产生的端到端指标损失可以忽略不计。代码已开源：https://github.com/thu-ml/SageAttention。

---

## 研究动机

注意力机制的二次时间复杂度（O(N²)）使得其高效实现至关重要，特别是在实际应用中序列长度不断增加的情况下。现有的高效注意力方法主要分为两类：

1. **线性注意力方法**：将复杂度降低到 O(N)，但仅适用于有限的模型和任务范围。
2. **稀疏注意力方法**：选择性处理部分上下文，同样有适用范围限制。
3. **硬件优化方法**（如 FlashAttention V1/V2/V3、xformers）：利用硬件特性加速计算，但对量化注意力的支持仍然有限。

SageAttention2 的核心动机在于：

- **量化注意力的空白**：虽然线性层的量化已被广泛使用，但其在注意力加速中的应用仍然有限。作者发现，直接对 Q、K 进行 INT4 量化存在三个关键挑战：
  - **C1**：INT4 的数值范围相比 FP16/INT8 非常有限，当 Q 和 K 存在异常值时，量化误差显著。
  - **C2**：GPU 张量核心中用于 FP8 矩阵乘法的 FP32 累加器实际上是 FP22（1 位符号、8 位指数、13 位尾数），这会导致 $\tilde{P}V$ 的精度损失。
  - **C3**：如果直接使用 INT4 量化，Llama3 在 MMLU 上的准确率仅为随机猜测水平的 25%。

- **SageAttention 的局限**：前作 SageAttention 使用 INT8 per-block 量化，但其对 INT4 量化的精度支持不足，且无法充分利用更快的 INT4 计算单元。

因此，SageAttention2 旨在通过更精细的量化策略和精度增强技术，在保持近乎无损精度的前提下，大幅加速注意力计算。

---

## 方法（技术细节）

SageAttention2 的核心方法包含三个关键技术组件，下面进行详细阐述：

### 1. Q/K 的 Per-thread INT4 量化

传统的 per-block 量化（SageAttention 使用的方式）在 INT4 下精度不足。SageAttention2 提出了 **per-thread 量化** 方法，基于 GPU 线程与矩阵内存布局之间的映射关系（由 PTX mma 指令决定）。

- **原理**：将对应同一 GPU 线程的 token 分组进行量化和反量化，确保每个线程仅关联一个量化尺度。
- **优势**：相比 per-block 量化实现更好的精度，且无需额外的反量化开销。
- **精度对比**（表 6）：per-thread 量化在 CosSim、Relative L1、RMSE 上与 per-token 量化接近，但远优于 per-block 和 per-tensor 量化。
  - Per-thread: CosSim 99.45%, Rel L1 0.0622, RMSE 0.0313
  - Per-block: CosSim 98.03%, Rel L1 0.1492, RMSE 0.0744
  - Per-tensor: CosSim 97.15%, Rel L1 0.1800, RMSE 0.0865

### 2. Q 和 K 的 Outlier 平滑（Smoothing）

作者发现 Q、K 在不同 token 之间的变化很小，大部分值相似，但存在通道级异常值（outlier）。SageAttention 已采用平滑 K 的方法，SageAttention2 进一步提出平滑 Q 的方法。

- **平滑公式**：
  - $\gamma(Q_i) = Q_i - \bar{q}_i$，其中 $\bar{q}_i = \text{mean}(Q_i)$ 沿 token 轴计算，广播到 block 内所有 token。
  - $\gamma(K_j) = K_j - \bar{k}$，其中 $\bar{k} = \text{mean}(K)$ 沿 token 轴计算，广播到整个张量。
- **分解**：$S_{ij} = Q_i K_j^\top = (\bar{q}_i + \gamma(Q_i))(\bar{k} + \gamma(K_j))^\top = \gamma(Q_i)\gamma(K_j)^\top + \Delta S_{ij} + b$
  - 其中 $\Delta S_{ij} = \bar{q}_i \gamma(K_j)^\top$ 是一个 1×N 向量。
  - $b = \bar{q}_i \bar{k}^\top + \gamma(Q_i)\bar{k}^\top$ 是一个 N×1 向量，但由于加在整个行上的共同偏置不影响 softmax 结果，因此不需要计算。
- **两阶段计算**：
  1. **预处理**：平滑 Q、K，进行量化，计算 $\Delta S_{ij}$。平滑、量化和 GEMV（用于计算 $\Delta S$）可融合到一个 kernel 中。
  2. **注意力**：执行低精度 GEMM，反量化，并加回 $\Delta S$：$S_{ij} = \psi^{-1}_{\delta Q \delta K}(\hat{Q}_i \hat{K}_j^\top) + \Delta S_{ij}$

- **精度对比**（表 4/5）：仅平滑 Q 可将 CosSim 提升到 98.30%，Q+K 平滑则达到 99.46%，且对端到端指标影响极小。

### 3. $\tilde{P}V$ 的两阶段累加策略（Two-level Accumulation）

GPU 张量核心中用于 FP8 矩阵乘法的 FP32 累加器实际上是 FP22（1 位符号、8 位指数、13 位尾数），这导致 $\tilde{P}V$ 的精度损失。

- **方法**：使用 FP32 缓冲区在每次 $\tilde{P}V$ 的 block 矩阵乘法后从 22 位累加器中累积结果，将误差限制在 block 范围内。
- **工作流程**：
  1. $\tilde{P}_{ij}$ 量化为 FP8（E4M3 格式）
  2. $\tilde{P}_{ij} \times V_j$ 的结果在 FP22 累加器中累加
  3. 每个 block 乘法后，将结果转移到 FP32 缓冲区进行二次累加
- **额外精度增强**：可选的平滑 V 技术，通过减去 V 的通道维度均值 $\bar{V}_m$，使 $\tilde{P}V$ 的值更接近零，利用浮点数表示在零附近更密集的特性提升精度。同时只需在最终输出中加回 $\bar{V}_m$（因为 $\tilde{P}$ 的行和为 1）。

### 4. 整体工作流程

SageAttention2 的完整流程（如图 3 所示）：

1. **平滑 Q、K、V**
2. **计算 $\Delta S$**（通过 GEMV）
3. **Per-thread 量化 Q、K，Per-channel 量化 V**
4. **执行 SageAttention2 kernel**
5. **修正输出**

### 5. 两个 Kernel 变体

- **SageAttn2-4b**：Q、K 量化为 INT4 per-thread，$\tilde{P}$、V 量化为 FP8 per-block 和 per-channel
- **SageAttn2-8b**：Q、K 量化为 INT8 per-thread，$\tilde{P}$、V 量化为 FP8 per-block 和 per-channel（适配 Hopper GPU，因其缺少原生 INT4 张量核心支持，且不使用 Q 平滑）

### 6. 量化格式说明

- **INT4 per-thread**：每个 GPU 线程对应一个量化尺度，将 Q、K 量化为 4 位整数
- **FP8 per-block/per-channel**：$\tilde{P}$ 和 V 以 FP8（E4M3）格式量化
- **FP22 累加器**：GPU 张量核心中的 FP32 累加器实际为 FP22，两阶段累加弥补精度损失

---

## 实验结果

### 1. Kernel 速度（表 9）

SageAttention2 在多种 GPU 上显著超越 FlashAttention2：

| 方法 | 3090 | 4090 | A100 | L40 | L20 | H100 | H20 |
|------|------|------|------|-----|-----|------|-----|
| FlashAttention2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| FlashAttention3 | ✗ | ✗ | ✗ | ✗ | ✗ | 1.37 | 1.57 |
| FlashAttention3 (fp8) | ✗ | ✗ | ✗ | ✗ | ✗ | 2.63 | 3.06 |
| SageAttention1 | 1.97 | 1.96 | 1.37 | 1.45 | 1.24 | 1.53 | 1.52 |
| SageAttention2 | ✗ | 2.93 | ✗ | 2.60 | 2.46 | 2.61 | 3.12 |

- 在 RTX4090 上，SageAttention2 的 OPS 比 FlashAttention2 高约 3 倍，比 xformers 高约 4.5 倍。
- 在 L20 上比 FlashAttention2 快约 2.5 倍。
- 在 H100/H20 上与 FlashAttention3(fp8) 速度相当，但精度远高于后者。
- SageAttention2-4b 的速度在所有 GPU 上均快于 SageAttention2-8b。

### 2. 端到端指标（表 2）

SageAttention2 在多个模型上的端到端指标表现：

**文本生成模型**：
- Llama3.1 (8B): SageAttn2-8b WikiText Ppl=6.019（与全精度 6.013 极其接近），Lambda Acc=0.811，MMLU Acc=0.634，LongBench=49.59
- GLM4 (9B): SageAttn2-8b WikiText Ppl=7.242（全精度 7.241），Lambda Acc=0.432，MMLU Acc=0.745，LongBench=49.60

**视频生成模型**：
- CogvideoX (1.5-5B): SageAttn2-8b CLIPSIM=0.1775（全精度 0.1778），CLIP-T=0.9980，VQA-a=69.492，VQA-t=74.415，FScore=2.487
  - SageAttention 前作在该模型上完全失败（全部 ✗），SageAttention2 成功解决
- HunyuanVideo: SageAttn2-8b 各指标接近全精度，CLIPSIM=0.1782，VQA-a=81.786
- Mochi: SageAttn2-8b 各指标接近全精度，CLIPSIM=0.1797，VQA-a=46.760

**关键发现**：
- SageAttn2-8b 在所有测试模型上几乎完全匹配全精度
- SageAttn2-4b 在部分指标上略有下降（如 MMLU 降低约 3-4%），但整体仍然远优于其他量化方法
- HadmdAttn 和 SmoothAttn 在视频生成模型上性能严重下降
- FlashAttn3(fp8) 在视频生成任务上表现不佳，多个模型无法生成有效结果

### 3. 可视化对比

- 图 6（HunyuanVideo）、图 7（CogvideoX）、图 9（Mochi、HunyuanVideo）展示 SageAttention2-8b 生成的视频质量与全精度几乎一致，而 HadmdAttn 和 SmoothAttn 生成的视频质量严重下降。
- 图 1（端到端推理）：在 L20 上 Llama3.1 生成首个 token 和 Needle-in-a-Haystack 任务（100K 序列长度）中，SageAttention2 比 FlashAttention2 快 1.7 倍（23s vs 39s）。
- 图 1（视频生成）：CogvideoX (1.5-5B) 在 RTX4090 上，SageAttention2 加速 1.8 倍（577s vs 1040s），视频质量无损失。

---

## 优势

1. **显著的计算加速**：在 RTX4090 上实现对 FlashAttention2 约 3 倍的 OPS 提升，对 xformers 约 4.5 倍提升。在 Hopper GPU 上与 FlashAttention3(fp8) 速度相当。
2. **近乎无损的精度**：SageAttn2-8b 在语言、图像、视频生成等多个模型上端到端指标与全精度几乎完全一致（如 Llama3.1 的 WikiText Ppl 仅差 0.006，MMLU Acc 仅差 0.001）。
3. **广泛的适用性**：在 10 个代表性模型（包括文本生成、图像生成、视频生成）上进行了全面验证，涵盖了 Llama2/3.1、GLM4、CogvideoX、HunyuanVideo、Mochi、Flux、Stable-Diffusion3.5、TIMM 等。
4. **克服前作局限**：SageAttention 在某些视频生成模型（如 CogvideoX）上完全失败，而 SageAttention2 成功解决了这一问题。
5. **与 FlashAttn3(fp8) 的对比优势**：SageAttention2 在 Hopper GPU 上速度与 FlashAttn3(fp8) 相当，但精度远高于后者（FlashAttn3(fp8) 在视频生成模型上多个指标严重下降）。
6. **两种 Kernel 变体**：SageAttn2-4b 和 SageAttn2-8b，分别适用于不同硬件平台（前者适用于有 INT4 张量核心的 RTX4090 等，后者适用于 Hopper GPU）。
7. **无需额外硬件支持**：仅依赖现有 GPU 张量核心能力，不需特殊硬件。
8. **可选的精度增强技术**：平滑 V 技术在扩散模型中有效提升精度，且仅需在最终输出中加回均值。

---

## 局限

1. **仅支持注意力加速**：SageAttention2 仅针对注意力机制进行量化加速，不涉及线性层或其他操作的优化。
2. **对某些 GPU 不支持**：SageAttn2-4b 无法在 3090、A100 等不支持 INT4 张量核心的 GPU 上运行（表 9 中标记 ✗）。
3. **INT4 量化在部分场景下有精度损失**：SageAttn2-4b 在某些指标上（如 MMLU 准确率）仍有 3-4% 的下降。
4. **实现复杂度**：方法涉及多个关键技术（平滑、per-thread 量化、两阶段累加、可选的 V 平滑），实现和调优有一定复杂度。
5. **额外开销**：预处理阶段需要计算 GEMV 来获得 $\Delta S$，虽然已融合到 kernel 中，但仍存在一定的额外开销。
6. **对某些任务的精度影响**：虽然整体端到端指标损失可忽略，但 SageAttn2-4b 在长序列任务（如 LongBench）上可能有轻微下降。
7. **硬件依赖**：实现依赖于特定的 GPU 张量核心特性（如 FP8 E4M3、FP22 累加器），在不同硬件架构上可能需要适配。
8. **对量化粒度的敏感性**：per-thread 量化对 GPU 线程和内存布局的映射关系有较强依赖，需要深入了解硬件细节。

---

## 与 EfficientPaper 相关的研究方向

1. **注意力机制量化加速**：SageAttention2 属于"量化注意力"方向，是 EfficientPaper 中"quantization"和"kv_cache_quant"关键词的核心研究。该方向的后续工作可能包括：
   - 对 KV Cache 进行量化（已标注为 kv_cache_quant 关键词）
   - 与线性注意力、稀疏注意力等方法的结合
   - 适用于更长序列的量化策略

2. **硬件友好的量化方法**：per-thread 量化是一种硬件友好的量化方法，可扩展到其他计算密集型操作（如线性层、卷积层的量化）。

3. **动态精度自适应**：基于不同任务和模型的特性，动态选择 INT4/INT8 量化（如 SageAttn2-4b vs SageAttn2-8b），可能成为更通用的精度管理策略。

4. **与 FlashAttention 系列的对比与融合**：SageAttention2 与 FlashAttention2/3 的关系密切，后续可能探索将 SageAttention2 的量化技术集成到 FlashAttention 框架中。

5. **视频生成模型的高效推理**：SageAttention2 在视频生成模型（如 CogvideoX、HunyuanVideo、Mochi）上表现出色，是 EfficientPaper 中视频生成高效推理方向的重要参考。

6. **Q/K/V 平滑技术**：Q、K、V 的平滑方法是一种可推广的精度增强技术，可能应用于其他低精度计算场景（如混合精度训练、模型压缩等）。

7. **张量核心精度分析**：论文对 FP8 矩阵乘法的 FP22 累加器进行了深入分析，揭示了 GPU 硬件层面的精度限制，这为其他量化方法的设计提供了重要参考。

---

## AI 生成声明

本笔记由 AI Agent 自动生成，基于论文 SageAttention2 的原文内容、元数据和 PDF 文本提取。笔记中的技术细节、实验结果和分析均基于论文原文，但部分表述经过 AI 重新组织和翻译。请注意，本笔记可能存在对原文某些细节的简化或误解，建议读者以原文为准。
