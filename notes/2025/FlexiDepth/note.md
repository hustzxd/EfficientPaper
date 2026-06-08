# FlexiDepth: 预训练 LLM 中的自适应层跳过

> 本笔记由 AI Agent（Hermes Agent）于 2026 年 6 月 5 日自动生成，基于论文原文全文阅读。所有内容均为中文，生成代码块已包含声明。

---

## 一句话总结

FlexiDepth 通过在预训练 LLM 的每个解码层插入轻量路由器和适配器，实现逐 token 的自适应层跳过，在跳过 8/32 层的情况下仍保持 100.7% 的基准性能，显著优于现有层跳过方法。

---

## 摘要翻译

各种层跳过方法已被提出用于加速大语言模型（LLM）中的 token 生成。然而，它们忽略了一个基本问题：不同 token 生成时的计算需求如何变化？本文引入 FlexiDepth，一种动态调整文本生成中所用 Transformer 层数量的方法。通过引入即插即用的路由器和适配器，FlexiDepth 在不修改原始参数的前提下实现了 LLM 的自适应层跳过。将 FlexiDepth 应用于 Llama-3-8B 模型，在跳过 32 层中的 8 层的同时，保持了完整的 100% 基准性能。FlexiDepth 的实验结果表明，LLM 的计算需求随 token 类型显著变化：生成重复 token 或固定短语需要较少层数，而涉及计算或高不确定性的 token 需要更多层数。有趣的是，这种自适应分配模式与人类直觉一致。为了推进该领域的研究，我们开源了 FlexiDepth 及一个记录其层分配模式的数据集。

---

## 研究动机

1. **均匀计算分配的低效性**：当前 LLM 为每个 token 执行完整的前向传播，所有层一视同仁。然而，简单的任务（如复制文本）和复杂的任务（如数学推理）显然需要不同的计算量，这种均匀分配既浪费资源也可能导致过拟合。
2. **现有方法的局限**：
   - **静态层剪枝**（如 ShortGPT）：根据输入-输出差异永久删除层，无法适应不同 token 的动态需求。
   - **早期退出**（如 LayerSkip）：在中间点跳过后续所有层，粒度粗糙且对长序列生成任务性能损失严重。
   - **Mixture-of-Depth（MoD）**：需要从头训练，无法应用于预训练模型。
3. **核心问题**：不同 token 生成时的计算需求如何变化？FlexiDepth 旨在回答这一问题。

---

## 方法（技术细节）

FlexiDepth 的核心设计是在每个 Transformer 解码层中插入两个轻量级即插即用模块：**路由器（Router）**和**适配器（Adapter）**。原始 LLM 参数完全冻结，仅训练路由器和适配器。

### 2.1 路由器设计

- 输入：归一化后的隐藏状态 $X = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times d}$
- 路由器计算门控分数 $G = \sigma(\text{Router}(\text{Norm}(X)))$，其中 $\sigma$ 为 sigmoid 函数
- **关键创新**：路由器采用**瓶颈 MLP**（bottlenecked MLP）而非简单线性变换：
  $$\text{Router}(z) = W_r \cdot (W_\uparrow \cdot \text{Norm}(\tanh(W_\downarrow z)))$$
  其中 $W_\downarrow \in \mathbb{R}^{d_r \times d}$，$W_\uparrow \in \mathbb{R}^{d \times d_r}$，$W_r \in \mathbb{R}^{1 \times d}$，$d_r = d/16$。
- **路由策略**：阈值 $\tau$ 判定——若 $g_i > \tau$ 则走全处理路径（输出乘以 $g_i$ 保持梯度流），否则走跳过路径（输出乘以 $(1-g_i)$）。
- 使用 SparseMixer 确保可微性。

### 2.2 注意力跳过（Attention Skipping）

- 跳过的 token **绕过 query 计算和缩放点积操作**，但**仍计算 KV cache**。
- 这样做的原因是：如果跳过的 token 不产生 KV cache，后续 token 的 query 将无法注意到这些 token，导致上下文信息永久丢失。
- 与不做 KV cache 的版本相比，保留 KV cache 的方案性能从 84.3% 提升至 100.7%（保留率）。

### 2.3 FFN 跳过（FFN Skipping）

- 直接跳过 FFN 会导致严重的表示不一致（FFN 包含非线性变换，跳过与不跳过的隐藏状态不在同一隐空间）。
- **解决方案**：使用一个轻量级适配器（结构与 FFN 相同，但中间维度缩小 16 倍）来对齐跳过路径的表示。
- 适配器是 FlexiDepth 性能的关键（消融实验表明去掉适配器后性能降至 28.1%）。

### 2.4 层跳过损失（Layer-skipping Loss）

$$\mathcal{L}_{\text{skip}} = \frac{1}{T} \sum_{t=1}^{T} \left( \sum_{l=1}^{L} g_t^l \right)^2$$

- 平方损失对使用更多层的 token 施加更大惩罚，防止模型走入极端模式（全部跳过或不跳过）。
- 最终损失：$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{skip}} + \mathcal{L}_{\text{lm}}$，其中 $\alpha = 1 \times 10^{-3}$。

### 2.5 训练细节

- **模型**：Llama-3-8B-Instruct（32 层），将后 16 层转换为 FlexiDepth 层。
- **训练数据**：Tulu-v2 数据集，3 个 epoch。
- **优化器**：AdamW，lr=1e-4，β1=0.9，β2=0.999，warmup=0.03。
- **批量大小**：64。
- **硬件**：8× NVIDIA A100-PCIE-40GB GPU，约 7 小时。
- **路由器瓶颈维度**：$d_r = d/16$。

---

## 实验结果

### 主要结果（表 1）

| 方法 | 跳过层数 | 单 token 任务 | 多 token 任务 | 保留率 |
|------|---------|-------------|-------------|--------|
| Vanilla | 0 | 0.673/0.706/0.744 | 0.679/0.299/0.784 | 100.0% |
| LayerSkip | 4 | 0.659/0.636/0.676 | 0.004/0.0/0.350 | 54.0% |
| ShortGPT | 4 | 0.664/0.662/0.700 | 0.536/0.092/0.145 | 69.1% |
| LaCo | 4 | 0.671/0.693/0.724 | 0.581/0.031/0.778 | 81.7% |
| MindSkip | 4 | 0.664/0.698/0.722 | 0.378/0.189/0.720 | 84.2% |
| **FlexiDepth** | **4** | **0.663/0.724/0.756** | **0.695/0.390/0.810** | **106.5%** |
| **FlexiDepth** | **8** | **0.616/0.705/0.735** | **0.662/0.341/0.801** | **100.7%** |

- 跳过 4 层时：FlexiDepth 达到 106.5% 保留率，甚至超越原始模型。
- 跳过 8 层时：FlexiDepth 保留 100.7%，而基线方法在多 token 生成任务上严重退化（LayerSkip 仅 43.9%，ShortGPT 32.0%）。

### 跨模型泛化（表 2）

- **Llama-2-13B-Instruct**：跳过约 7 层，保留 100.2%。
- **Llama-3-8B-Instruct**：跳过约 6 层，保留 102.1%。
- **Qwen-2.5-3B-Instruct**：仅跳过 1-2 层，保留 101.5%。
- **结论**：更大模型具有更高冗余度，允许更激进的层跳过。

### 消融实验（表 3）

| 组件 | 保留率 | 说明 |
|------|--------|------|
| 完整 FlexiDepth | 102.1% | 基线 |
| 线性路由器 | 68.7% | GSM8K 从 0.657 降至 0.131 |
| 无 KV cache | 84.3% | 上下文信息丢失 |
| 无适配器 | 28.1% | 表示对齐失败 |

### 层分配模式（Token Depth Map）

- **文本任务**：复制（21.95层）< 摘要（28.65层）< 续写（30.27层）
- **数学任务**：重复（20.09层）< 加法（22.45层）< 乘法（23.90层）
- 这与人类直觉一致：简单复制/重复需要更少层数，复杂推理需要更多层数。
- 乘法方程的右侧 token（如 "945", "40515"）使用几乎全部层数，而左侧 token 使用较少层。

---

## 优势

1. **无需修改原始参数**：仅插入轻量路由器和适配器，适用于任何预训练 LLM。
2. **卓越的性能保留**：跳过 8/32 层后仍保持 100.7% 基准性能，远超现有方法。
3. **自适应粒度**：逐 token 决定跳过层数，而非统一跳过固定层数。
4. **KV cache 保持**：确保自回归生成的上下文完整性。
5. **可解释性**：Token Depth Map 揭示了不同 token 的计算需求差异，具有研究价值。
6. **跨模型泛化**：在多种模型（Llama-2-13B、Llama-3-8B、Qwen-2.5-3B）上均有效。
7. **潜在正则化效果**：跳过噪声或信息量少的参数可能起到隐式正则化作用，提升泛化能力。
8. **开源**：提供模型和层分配数据集，促进后续研究。

---

## 局限

1. **实际吞吐量未提升**：虽然减少了 FLOPs，但由于 batch 内 token 走不同路径，控制流管理和不规则内存访问的开销抵消了理论加速。
2. **仅对后 16 层应用**：跳过早期层会显著降低性能，因此只能对后半部分层进行自适应跳过。
3. **仅在 Llama-3-8B 上全面验证**：对更大规模模型（如 70B+）的验证不足。
4. **训练成本**：需要 8×A100 GPU 训练 7 小时，对资源要求较高。
5. **阈值超参数**：需要根据目标跳过层数调整 α 系数，不同任务可能需要不同的 α。
6. **缺乏真实硬件加速**：论文明确承认在现有 GPU 硬件上无法获得实际吞吐量提升，需要硬件感知的优化（如 token grouping、expert sharding）。
7. **仅支持 decoder-only 模型**：虽然论文聚焦于 decoder-only 架构，但其方法可推广至 encoder-decoder 模型。

---

## 与 EfficientPaper 相关的研究方向

### 关键词关联

本论文涉及的核心研究方向：
- **sparse_pruning**（稀疏剪枝）
- **structured_sparsity**（结构化稀疏）

### 相关研究方向

1. **动态计算分配（Adaptive Computation）**：
   - 与 Mixture-of-Depth、Mixture-of-Experts (MoE) 等动态路由方法密切相关。
   - 探索在推理阶段按需分配计算资源，提升效率。

2. **层跳过/层剪枝（Layer Skipping/Pruning）**：
   - 与 ShortGPT、LaCo、MindSkip、LayerSkip 等方法构成同一研究谱系。
   - 但 FlexiDepth 的逐 token 自适应跳过机制是该方向的重要进展。

3. **KV Cache 优化**：
   - FlexiDepth 的 KV cache 保持策略与高效推理中的 KV cache 管理相关。
   - 可与 vLLM、PagedAttention 等 KV cache 优化技术结合。

4. **高效微调（Parameter-Efficient Fine-tuning）**：
   - 仅训练路由器和适配器，与 LoRA、Adapter 等 PEFT 方法具有方法论上的相似性。
   - 可探索 FlexiDepth 与其他 PEFT 方法的联合优化。

5. **硬件感知优化（Hardware-Aware Optimization）**：
   - 论文明确指出需要 token grouping、expert sharding、load balancing 等技术来实现实际加速。
   - 与 GPU 架构优化、内存访问模式优化等研究方向密切相关。

6. **可解释性与层分析（Interpretability & Layer Analysis）**：
   - Token Depth Map 提供了一种可视化和理解 LLM 内部计算模式的方法。
   - 与模型可解释性、层功能分析等研究方向相关。

7. **推理效率（Inference Efficiency）**：
   - FlexiDepth 可与 speculative decoding、early exit、模型蒸馏等技术结合，进一步提升推理效率。
   - 可探索 FlexiDepth 在 edge devices、mobile devices 等资源受限场景的应用。

---

## 参考信息

- **论文标题**：Adaptive Layer-skipping in Pre-trained LLMs
- **作者**：Xuan Luo, Weizhi Wang, Xifeng Yan（UC Santa Barbara）
- **发表时间**：2025 年 3 月 31 日
- **平台**：arXiv (2503.23798v1)
- **模型**：Llama-3-8B-Instruct（32 层）
- **开源**：
  - 模型：xuan-luo/FlexiDepth-Llama-3-8B-Instruct
  - 数据集：xuan-luo/FlexiPatterns-Llama-3-8B-Instruct
- **关键词**：sparse_pruning, structured_sparsity

---

*本笔记由 AI Agent（Hermes Agent）自动生成，基于论文原文全文阅读，仅供参考。*
