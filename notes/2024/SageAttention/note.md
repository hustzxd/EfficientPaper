# SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration

> Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen
> Tsinghua University | arXiv:2410.02367 | ICLR 2025
> Code: https://github.com/thu-ml/SageAttention

![111](../../blank.jpg)

## 一句话总结

SageAttention 提出了一种高效的 8 位量化注意力加速方法，通过对 K 矩阵进行平滑处理和使用 FP16 累加器，在保持近乎无损精度的前提下，实现了对 FlashAttention2 约 2.1 倍、xformers 约 2.7 倍的速度提升，且以即插即用的方式适用于语言、图像和视频生成等多种模型。

## 摘要翻译

Transformer 架构在各类模型中占据主导地位。作为 Transformer 的核心组件，注意力机制的计算复杂度为 O(N²)，而线性变换仅为 O(N)。在处理长序列时，注意力成为主要的时间消耗部分。虽然量化已被证明是加速模型推理的有效方法，但现有量化方法主要聚焦于优化线性层。为此，我们首先详细分析了注意力机制中量化的可行性。随后，我们提出了 SageAttention，一种高效且精确的注意力量化方法。该方法的 OPS（每秒操作数）分别超过 FlashAttention2 和 xformers 约 2.1 倍和 2.7 倍。SageAttention 在精度方面也优于 FlashAttention3。综合实验表明，该方法在多种模型（包括大语言处理、图像生成和视频生成模型）上几乎不造成端到端指标损失。

## 研究动机

1. **注意力机制是性能瓶颈**：在长序列场景下（如视频生成和语言模型预填充，序列长度可达 8K~128K），由于注意力的 O(N²) 复杂度，其延迟远超线性变换等操作，成为主要性能瓶颈。
2. **现有量化方法忽视了注意力**：已有量化工作主要集中在优化线性层（权重和激活的低精度计算），而注意力机制仍以 FP16 精度运行，未被有效加速。
3. **FlashAttention3 的局限性**：虽然 FlashAttention3 提供了 FP8 版本，但它只能在 Nvidia Hopper 架构上使用，且直接使用 FP8 量化会导致显著的精度下降。
4. **注意力量化面临独特挑战**：与线性层不同，注意力包含 softmax 和矩阵乘法 QK⊤ 及 PV，直接对 Q、K、P、V 进行 8 位量化会导致严重的精度损失。例如，Unidiffuser 在 INT8 和 FlashAttention3 FP8 实现下会生成完全模糊的图像。

## 方法（技术细节）

SageAttention 是一种基于后训练量化（post-training quantization）的方法，以即插即用的方式在推理时替换原始高精度实现。核心思路包括以下几个关键技术：

### 1. 量化格式选择：INT8 而非 FP8

- 选择 INT8 而非 FP8 进行量化，因为 INT8 Matmul 在常用 GPU（如 RTX4090 和 3090）上比 FP16 快 4 倍，比 FP8 快 2 倍。
- 表 2 实验表明，对 Q、K 进行 INT8 量化的精度高于 FP8（E4M3 和 E5M2）。

### 2. K 矩阵平滑（Smooth Matrix K）

- **挑战 (C1)**：K 矩阵存在显著的通道级离群值（channel-wise outliers），直接量化会导致严重精度下降。
- **解决方案**：对 K 进行平滑变换 γ(K) = K - mean(K)，即减去所有 token 的平均 K。
- **数学保证**：该变换不改变注意力分数 P，因为 σ(q(K - mean(K))ᵀ) = σ(qKᵀ - q·mean(K)) = σ(qKᵀ)。
- **速度开销**：小于 0.2%，几乎可以忽略。

### 3. 量化粒度设计

- **P̃（softmax 输出）**：使用 per-block 量化（每个块的 P̃ 最大值为 1，可使用静态缩放因子 s=1/127）。
- **V**：使用 per-channel 量化，以解决 V 的通道级离群值问题。
- **Q、K**：采用动态量化，结合 K 的平滑处理。

### 4. FP16 累加器方案

- **挑战 (C2)**：仅使用 INT8 量化 P̃ 和 V 在某些模型层精度极差（见表 3）。
- **解决方案**：使用 FP16 数据类型和 FP16 累加器进行 PV 矩阵乘法。
- **优势**：
  - 在 RTX4090 和 3090 上，FP16 Matmul 比 FP32 快 2 倍。
  - 使用 FP16 累加器比 FP32 节省更多寄存器资源。
  - 精度远优于 INT8（见表 3，FP16 的最差精度远高于 INT8）。

### 5. 与 FlashAttention2 集成

- SageAttention 基于 FlashAttention-2 的分块（tiling）机制构建。
- 量化、反量化操作与 FlashAttention 的分块策略对齐。
- 在 FlashAttention2 基础上增加了 Q、K、P、V 的量化器和反量化器。
- 保持 softmax 的全精度计算（FP32）。
- Q 块大小 128，K/V 块大小 64。
- 使用 OpenAI Triton 实现 CUDA 内核。

## 实验结果

### 速度提升

| 模型 | QKV 形状 | 原始注意力 | SageAttention | 加速比 |
|------|----------|-----------|--------------|--------|
| CogvideoX | (2,30,17776,64) | 163.37 (FlashAttn2) | 327.57 | 2.01x |
| Llama2 | (4,32,1536,128) | 130.99 (FlashAttn2) | 231.74 | 1.77x |
| UltraPixel | (2,32,7285,64) | 152.03 (FlashAttn2) | - | - |

- 平均真实加速比：约 2.83 倍（相对原始注意力）。
- OPS 比 FlashAttention2 快约 2.1 倍，比 xformers 快约 2.7 倍。

### 精度保持

| 模型 | 指标 | 全精度 | SageAttention |
|------|------|--------|--------------|
| Llama2 | WikiText (Ppl.) | 5.823 | 5.824 |
| Llama2 | MMLU (Acc.) | 0.46 | 0.46 |
| CogvideoX | FScore | 3.768 | 3.8339 |
| Unidiffuser | FID | 163.33 | 166.49 |
| UltraPixel | FID | 179.78 | 179.79 |
| TIMM | ImageNet (Acc.) | 84.79% | 84.74% |
| Llava1.6 | TextVQA (Acc.) | 60.25% | 60.09% |

- 在所有模型上，SageAttention 的端到端指标损失几乎可以忽略不计。
- 与 FlashAttention3 的量化版本相比，SageAttention 精度显著更高（表 1）。

### 对比其他方法

- **与 AWQ（W4A16）对比**：SageAttention 可与 AWQ 叠加使用，共同加速。
- **与 Q-diffusion（W8A8）对比**：SageAttention 在 Unidiffuser 上的 FID（166.49）远优于 Q-diffusion（395.99）。
- **与 ViDiT-Q（W8A8）对比**：SageAttention 在 CogvideoX 上的端到端加速（34.3%）优于 ViDiT-Q 理论最大值（22%），且质量更高。

## 优势

1. **即插即用**：作为后训练量化方法，无需重新训练，可直接替换原始注意力实现。
2. **广泛适用性**：已在语言模型（Llama2）、图像生成（Unidiffuser、UltraPixel）、视频生成（CogvideoX）、图像分类（TIMM）和视觉问答（Llava1.6）上验证。
3. **高性能**：比 FlashAttention2 快约 2.1 倍，比原始注意力快约 2.83 倍。
4. **高精度**：端到端指标损失几乎可忽略，优于 FlashAttention3 的量化版本。
5. **跨平台**：支持 RTX4090、RTX3090 等多种 GPU，不依赖特定架构（如 Hopper）。
6. **与其他量化方法正交**：可与 AWQ 等线性层量化方法叠加使用，进一步加速。
7. **K 矩阵平滑开销极低**：仅增加不到 0.2% 的时间开销，但显著提升精度。

## 局限

1. **仅针对注意力层**：SageAttention 主要加速注意力计算，对线性层（如 Q/K/V 投影、FFN）的量化需要其他方法（如 AWQ）。
2. **GPU 架构依赖**：INT8 加速依赖于支持高效 INT8 Matmul 的 GPU（如 RTX4090/3090），在其他 GPU 上加速效果可能不同。
3. **量化精度上限**：虽然 INT8 量化在 Q/K 上表现良好，但对 P/V 的 INT8 量化在某些层可能出现精度下降（通过 FP16 累加器部分缓解）。
4. **需要 FlashAttention2 基础**：SageAttention 基于 FlashAttention-2 构建，依赖其分块策略。
5. **无训练时量化支持**：当前仅支持后训练量化（post-training quantization），不支持量化感知训练。
6. **对特殊场景的适用性**：对于 Q/K 分布极其不均匀的模型，平滑 K 的效果可能有限，需要进一步研究。

## 与 EfficientPaper 相关的研究方向

1. **KV Cache 量化**：SageAttention 的 KV 矩阵量化技术可直接应用于 KV Cache 压缩，减少内存占用。
2. **线性层量化**：SageAttention 与 AWQ、Q-diffusion、ViDiT-Q 等方法正交，可结合使用实现全模型量化加速。
3. **稀疏注意力**：SageAttention 与稀疏注意力方法（如 Minference、LongLora）正交，可结合使用实现更高效的注意力计算。
4. **MoE 系统**：作者在后续工作中（SageAttention2、SageAttention3）探索了与 MoE 系统和训练优化的结合。
5. **扩散模型加速**：SageAttention 在 Unidiffuser、UltraPixel、CogvideoX 等扩散模型上表现优异，是扩散模型推理加速的重要方向。
6. **异构 GPU 系统**：SageAttention 的跨平台特性（支持 RTX4090/3090）使其适用于异构 GPU 环境。
7. **RAG 系统**：作者在 Sage 框架中将注意力量化技术应用于 RAG 系统的检索优化。

## AI 生成声明

本笔记由 AI Agent（Hermes Agent）基于论文 PDF 文本提取和元数据自动生成。内容经过整理和翻译，力求准确反映论文核心思想，但可能存在对原始表述的简化或误读。请以原始论文为准。
