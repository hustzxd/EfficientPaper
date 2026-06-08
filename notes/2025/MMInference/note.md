# MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention

> Yucheng Li, Huiqiang Jiang, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, Lili Qiu

![](../../blank.jpg)

## 一句话总结

MMInference 是一种基于模态感知排列的动态稀疏注意力方法，通过利用多模态 VLM 中独特的 Grid 稀疏模式和模态边界特性，将长上下文 VLM 预填充阶段加速最高 8.3 倍（在 1M token 时），且无需任何模型修改或微调。

---

## 摘要翻译

将长上下文能力与视觉理解相结合，为视觉语言模型（VLM）释放了前所未有的潜力。然而，预填充阶段的二次注意力复杂度仍是实际部署的重大障碍。为克服这一限制，我们提出了 MMInference（Multimodality Million tokens Inference），一种动态稀疏注意力方法，用于加速长上下文多模态输入的预填充阶段。

首先，我们的分析揭示了视频输入的时空局部性导致了一种独特的稀疏模式——Grid 模式。同时，VLM 在不同模态间表现出明显不同的稀疏分布。我们引入了一种基于排列的方法来利用独特的 Grid 模式并处理模态边界问题。通过离线搜索每个注意力头的最优稀疏模式，MMInference 根据输入动态构建稀疏分布。我们还提供了优化的 GPU 内核以实现高效的稀疏计算。值得注意的是，MMInference 可无缝集成到现有 VLM 管线中，无需任何模型修改或微调。

在多模态基准测试（包括 Video QA、Captioning、VisionNIAH 和 Mixed-Modality NIAH）上的实验，使用最前沿的长上下文 VLM（LongVila、LlavaVideo、VideoChat-Flash、Qwen2.5-VL）表明，MMInference 在 1M token 时可将预填充阶段加速最高 8.3 倍，同时保持精度。

---

## 研究动机

1. **长上下文 VLM 的实际部署瓶颈**：随着 VLM 上下文长度扩展到处理长视频和多模态输入，预填充阶段的注意力计算复杂度为 O(n²)，导致极高的 Time-to-First-Token 延迟。例如，256 帧视频的预填充可能需要数分钟，严重阻碍了长上下文 VLM 在实际应用中的广泛采用。

2. **现有稀疏注意力方法的局限**：虽然 MInference 等方法已提出动态稀疏注意力用于长上下文 LLM 加速，但这些方法未能有效利用 VLM 中独特的稀疏模式，且难以处理混合或交错模态输入，限制了其在多模态场景中的适用性。

3. **VLM 独特的稀疏模式**：VLM 的注意力矩阵具有独特的稀疏模式，包括由视频输入时空局部性产生的 Grid 模式，以及由不同模态间注意力分布差异产生的模态边界问题。这些特性需要专门的稀疏注意力策略。

4. **模态边界问题**：多模态输入中，不同模态（如视觉和语言）之间的注意力模式存在明显边界，现有的稀疏注意力方法无法有效处理这些边界，导致性能下降。

---

## 方法（技术细节）

### 核心框架

MMInference 由三个模块组成，覆盖了模态内和模态间稀疏注意力模式：

#### 1. Grid Head 稀疏注意力（模态内）

**动机**：视频和图像输入的时空局部性导致注意力图中出现均匀间隔的水平和垂直线，形成 Grid 模式。这种模式与文本 LLM 中的垂直斜线模式（Vertical-Slash）截然不同。

**方法**：
- **在线搜索**：利用最后一个查询向量（lastq=64）与 Key 的近似注意力矩阵，通过 view 操作快速搜索 Grid 的步长和相位（stride 和 phase）
- **排列优化**：使用识别出的 Grid 步长和相位对 Q、K、V 张量进行排列，将稀疏的 Grid 模式转换为连续的块稀疏索引，便于硬件高效计算
- **实现优化**：在内核中动态加载和写入张量，避免显式排列的开销

**算法流程**（Algorithm 1）：
1. 用 lastq 个查询向量近似注意力矩阵
2. 在线搜索 Grid 步长和相位
3. 对 Q、K、V 进行排列
4. 使用动态块稀疏注意力（FlashAttention）计算
5. 稀疏混合分数和值

#### 2. 混合模态稀疏注意力（模态间）

将 VLM 中的模态边界分为四类：

- **No-Boundary**：无明显模态边界
- **K-Boundary**：仅在 Key 维度有边界
- **Q-Boundary**：在 Query 维度有边界
- **2D-Boundary**：在 Query 和 Key 维度均有边界

**Q-Boundary Head**：
- 使用行排列（row-wise permutation）按模态分组 Q 张量
- 对每个模态分别应用离线优化的稀疏注意力（A-shape、Vertical-Slash、Grid Head）
- 利用每个模态末尾的查询向量动态近似稀疏索引

**2D-Boundary Head**：
- 在 Query 和 Key 维度均执行排列
- 对 Q、K、V 按模态分组
- 对每对模态进行迭代遍历，分别计算动态稀疏注意力
- 使用 Triton 实现注意力掩码构建

#### 3. 模态感知稀疏注意力搜索算法

分三步进行：
1. **模态内搜索**：在每个模态内搜索最优稀疏模式
2. **跨模态搜索**：在所有模态对之间搜索
3. **模态间搜索**：基于前两步的结果进行联合优化

搜索空间基于实际 GPU 内核测量的 FLOPs（而非理论估计），确保实际效率。搜索时间约 15 分钟（单张 A100），使用一个 egoschema 任务的样本作为校准集。

### 优化的 GPU 内核

- **Grid-Shape Flash Attention**：集成块稀疏 FlashDecoding（Q 端稀疏）和块稀疏 FlashAttention-2（K 端稀疏）
- **Q-Boundary Flash Attention**：沿 Query 维度引入稀疏性
- **2D-Boundary Flash Attention**：在 Query 和 Key 维度均应用稀疏性

所有内核使用 Triton 和 FlashAttention 实现，通过 PIT（Permutation Invariant Transformation）动态稀疏编译器进行优化。

---

## 实验结果

### 主要实验设置

- **模型**：Llava-Video-7B、LongVILA-7B、Qwen2.5-VL-7B、VideoChat-Flash
- **基准测试**：Video QA（ActNet-QA、EgoSchema、Next-QA）、PerceptionTest、VideoDC、VideoMME、V-NIAH、MM-NIAH
- **硬件**：单张 NVIDIA A100（bfloat16）

### 关键结果

#### 1. 视频理解任务（Table 1）

- **Llava-Video-7B**（110帧，20K tokens）：
  - MMInference：47.3% FLOPs，平均精度 57.6%，与全注意力（57.6%）持平
  - MInference：78.8% FLOPs，平均精度 57.5%
  - Tri-shape：49.0% FLOPs，平均精度 56.7%

- **LongVILA-7B**（256帧，65K tokens）：
  - MMInference：31.8% FLOPs，平均精度 55.4%，优于全注意力（55.5%）
  - MInference：47.0% FLOPs，平均精度 55.2%

- **Qwen2.5-VL-7B-Instruct**（256帧，33K tokens）：
  - MMInference：41.3% FLOPs，平均精度 59.4%，与全注意力（59.5%）持平

#### 2. Video Needle In A Haystack（V-NIAH）

- MMInference 达到 97.7% 的召回率（LongVILA-1M），几乎与全注意力（98.3%）一致
- 在 6000 帧（约 1.1M tokens）的超长上下文中保持稳定性能
- A-shape 在 300 帧即开始退化，Tri-shape 在 3900 帧后急剧下降

#### 3. Mixed-Modality NIAH（MM-NIAH）

- MMInference 达到 91.3% 的召回率，优于全注意力（90.9%）
- 在混合模态输入中，MMInference 通过模态间稀疏模式保持性能
- MInference 在 2700 帧后显著下降，而 MMInference 持续稳定

#### 4. 延迟分析

- **端到端加速**：
  - 相比 FlashAttention-2：最高 8.3× 加速（1M tokens）
  - 相比 MInference：最高 1.7× 加速（1M tokens）
- **内核级加速**：Grid 模式在 1M tokens 时达到 12× 内核级加速
- **延迟对比**：
  - Grid 模式：358ms（1M tokens，单 A100）
  - Vertical-Slash：显著更高延迟
  - Grid 模式比 Vertical-Slash 快 2-3×

#### 5. 与其他方法对比

- **与 token 压缩方法（如 VisionZip）集成**：可无缝集成，实现近乎无损性能
- **VideoChat-Flash**：MMInference 在 512 帧时保持与原始模型相当的性能
- **基线方法对比**：A-shape、SF-fixed、SF-strided、Tri-shape、MInference 在不同场景下各有优劣，但 MMInference 在所有场景下表现最佳

---

## 优势

1. **无需模型修改或微调**：MMInference 可无缝集成到现有 VLM 管线中，无需任何模型架构修改或重新训练。

2. **显著的加速效果**：在 1M token 时实现最高 8.3× 加速（端到端）和 12× 内核级加速，大幅降低预填充延迟。

3. **保持精度**：在所有基准测试中，MMInference 的精度与全注意力方法持平甚至略优，且仅需约 30-50% 的 FLOPs。

4. **处理混合模态输入**：通过 Q-Boundary 和 2D-Boundary 模式有效处理多模态输入中的模态边界问题，这是现有方法的盲区。

5. **动态稀疏模式**：基于输入动态构建稀疏分布，而非使用固定的静态模式，更好地适应不同的上下文和模态。

6. **可扩展性强**：在超长上下文（1M tokens，6000 帧）中保持稳定性能，适用于超长视频理解场景。

7. **GPU 优化实现**：使用 Triton、FlashAttention 和 PIT 编译器，实现高效的稀疏计算内核。

8. **可与 token 压缩方法集成**：与 VideoChat-Flash 等视觉 token 压缩方法无缝集成，实现近乎无损的性能。

---

## 局限

1. **搜索开销**：虽然搜索时间约 15 分钟（单 A100），但对于需要快速部署的场景，这可能构成一定开销。

2. **对 VLM 架构的依赖**：MMInference 的效果依赖于 VLM 中注意力头的特定稀疏模式（如 Grid 模式），对于不同架构的 VLM，稀疏模式可能有所不同，需要重新搜索。

3. **多模态类型的泛化**：虽然论文分析了视觉和语言模态，但对于音频、3D 等其他模态的泛化性未充分验证。

4. **训练时未涉及**：MMInference 仅在推理阶段使用，未考虑在训练阶段的加速，可能需要额外的适配。

5. **计算资源需求**：虽然加速显著，但 GPU 内核的实现和优化需要较高的技术门槛和硬件支持（如 A100、Triton 编译器）。

6. **对不同任务的适应性**：在一些任务中（如 EgoSchema），静态稀疏模式（如 Tri-shape）在特定情况下仍可能优于动态稀疏，说明不同任务可能需要不同的策略。

7. **模态边界检测的准确性**：模态边界检测依赖于在线估计，可能在复杂场景中存在误判。

---

## 与 EfficientPaper 相关的研究方向

### 相关关键词
- **sparse_pruning**：稀疏注意力模式的动态构建和优化
- **attention_sparsity**：VLM 中注意力矩阵的动态稀疏性

### 相关研究方向

1. **稀疏注意力加速**：
   - MInference（Jiang et al., 2024）：MMInference 的前作，专注于文本 LLM 的动态稀疏注意力
   - Native Sparse Attention（Yuan et al., 2025）：硬件对齐的原生可训练稀疏注意力
   - xAttention（Xu et al., 2025b）：块稀疏注意力与反对角评分
   - FlexPrefill（Lai et al., 2025）：上下文感知的稀疏注意力机制

2. **长上下文 VLM 效率**：
   - LongVILA（Chen et al., 2025）：多模态序列并行化加速视频微调
   - VideoChat-Flash（Li et al., 2025）：层级压缩用于长上下文视频建模
   - LongLLaVA（Wang et al., 2024b）：混合架构扩展多模态 LLM
   - VL-Cache（Tu et al., 2025）：稀疏性和模态感知的 KV 缓存压缩

3. **视觉 token 压缩**：
   - VisionZip（Yang et al., 2024）：视觉 token 压缩方法
   - Token Merging（Bolya et al., 2023）：Token 合并加速 ViT
   - ZipVL（He et al., 2024）：动态 token 稀疏化和 KV 缓存压缩

4. **DiT 推理加速**：
   - Sparse VideoGen（Xi et al., 2025）：时空稀疏加速视频扩散 Transformer
   - NATTEN（Hassani et al., 2023）：邻域注意力 Transformer
   - Efficient-VDIT（Ding et al., 2025）：注意力 Tile 加速视频扩散

5. **动态稀疏计算**：
   - PIT（Zheng et al., 2023）：动态稀疏深度学习模型的排列不变变换优化
   - FlashAttention-2（Dao, 2024）：更快的注意力实现
   - Triton（Tillet et al., 2019）：用于瓦片神经网络计算的中间语言和编译器

### 方法论意义

MMInference 的核心贡献在于将稀疏注意力从纯文本 LLM 扩展到多模态 VLM，通过模态感知排列解决了模态边界问题。这一思路可推广到其他多模态任务（如音频、3D 点云等），为高效多模态推理提供了新的技术路径。同时，其排列优化的思想也可应用于 DiT（Diffusion Transformer）等生成模型的推理加速。

---

> **声明**：本 note 由 AI Agent（Hermes Agent）基于论文全文自动生成，内容仅供参考。生成时间：2025年6月。
