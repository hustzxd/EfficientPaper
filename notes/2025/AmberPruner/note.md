# Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models

> Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang

![](../../blank.jpg)

> ⚠️ **本文由 AI Agent 自动生成**，基于论文原文提取与分析，仅供参考。

## 一句话总结

Amber Pruner 是一种面向 LLM 推理 prefill 阶段的训练无关 N:M 激活稀疏化方法，通过 Robust-Norm Scoring 和启发式层跳过策略，在 8:16 稀疏度下加速超过 55% 的线性投影计算，且 Zero-shot 平均准确率损失低于 1%。

---

## 摘要翻译

在大语言模型（LLM）时代，N:M 稀疏已成为一种关键的结构化压缩技术，用于加速推理。虽然先前工作主要集中在权重稀疏上，但常常遭受显著的精度下降。激活稀疏虽然前景广阔，但通常依赖训练且面临泛化挑战。为解决这些局限，我们提出 Amber Pruner，一种专门设计用于 prefill 阶段的训练无关 N:M 激活稀疏方法，旨在加速 LLM 中的线性投影层。在多个模型和稀疏率（2:4、4:8 和 8:16）上的大量实验表明，Amber Pruner 可以有效稀疏化并加速超过 55% 的线性计算，而无需重新训练模型。为进一步增强通用性和效率，我们提出 Outstanding-sparse，一个将 Amber Pruner 与训练后 W8A8 量化集成的统一框架。我们的方法在多个下游任务中保持了强劲的性能，在生成任务上具有显著优势。本工作开创了激活稀疏的新前沿，提供了基础性见解，有望指导下一代 AI 系统设计中算法和架构的协同演进。

---

## 研究动机

1. **权重稀疏的精度瓶颈**：传统的 N:M 权重稀疏方法（如 SparseGPT、Wanda）虽然在硬件加速方面有效，但在 MMLU 等挑战性基准上精度下降超过 20%，限制了实际部署。
2. **激活稀疏的潜力与不足**：激活稀疏虽然前景良好，但现有方法（如 Q-Sparse、TEAL）通常依赖训练（训练感知），且在多批推理场景中效果有限；Squared-ReLU 等激活函数方法需要训练感知且泛化能力存疑。
3. **Prefill 阶段的计算瓶颈**：LLM 推理的 prefill 阶段（线性投影层）计算密度极高，是性能瓶颈所在，但此前缺乏专门针对 prefill 阶段的激活稀疏化方法。
4. **N:M 稀疏的硬件友好性**：N:M 稀疏（如 NVIDIA Ampere 架构的 2:4 稀疏）是一种半结构化稀疏模式，能与 GPU/NPU 等硬件架构良好对齐，提供实际的推理加速。
5. **DeepSeek V3 的验证**：DeepSeek V3 的发布重新确认了稀疏计算是下一代 AI 系统的关键方向，N:M 稀疏受到学术界和工业界广泛关注。

---

## 方法（技术细节）

### 1. 整体框架

Amber Pruner 是一种**训练无关（training-free）**的 N:M 激活稀疏化算法，专门针对 LLM 推理的 prefill 阶段，对线性投影层的输入激活进行结构化稀疏化。其核心思想是：

- **激活稀疏优于权重稀疏**：实验观察到 LLM 线性层的激活值比权重更稀疏（约 50% 接近零），更适用于结构化稀疏化。
- **稀疏激活 × 密集权重**：将稀疏化后的激活与密集权重矩阵相乘，形成稀疏-密集矩阵乘法（SpMM），实现硬件级加速。

### 2. Weight-Aware Scoring（权重感知评分）

借鉴 Wanda 的思想，但反转为对激活进行评分。对于权重矩阵 W ∈ R^{d_out × d_in} 和激活矩阵 X，评分函数为：

$$S_{ij} = |X_{ij}| \cdot f(W_{:,j}) = |X_{ij}| \cdot \frac{\|W_{:,j}\|_2}{\min_k \|W_{:,k}\|_2}$$

其中：
- 使用权重列的 L2 范数的最小值归一化，防止低精度推理中的数值下溢
- 模型权重在推理时保持静态，因此评分系数可提前离线预计算并存储为辅助权重（开销 < 0.05% 模型大小）
- 通道级缩放可通过算子融合技术无缝集成到计算图中

### 3. Robust-Norm Scoring（鲁棒范数评分）

为解决 L2 范数评分在权重分布集中时区分度不足的问题，提出 Robust-Norm Scoring，包含三个关键步骤：

1. **异常值去除**：将权重 W 中 0.5%-99.5% 百分位之外的值丢弃：
   $$W = \{ \omega_k \mid Q_{0.005}(W) \leq \omega_k \leq Q_{0.995}(W) \}$$

2. **归一化**：对剩余权重进行标准化：
   $$\hat{W}_{ij} = \frac{W_{ij} - E[W]}{\sqrt{Var[W]}}$$

3. **通道级评分**：对每个通道 $\hat{W}_{:,j}$ 计算 L2 范数，最终通过 Wanda-like 规则分配激活元素的评分：
   $$S^*_{ij} = |X_{ij}| \cdot f(\hat{W}_{:,j})$$

Robust-Norm Scoring 通过数值归一化增强局部评分分辨率，在不违反 OBS 最优性约束的前提下提升端到端精度。

### 4. Layer Skipping Strategy（层跳过策略）

不同线性投影层对稀疏化的敏感度差异显著。通过量化稀疏引入的扰动误差来确定敏感层：

$$e_q(Y, Y') = \frac{\|Y - Y'\|_2}{\|Y\|_2 + \epsilon}$$

**层跳过规则**：
- **始终跳过**：k_proj 和 v_proj（由于 GQA，计算负担低）
- **始终跳过**：o_proj 和 up_proj（敏感度高）
- **始终剪枝**：down_proj（敏感度最低）
- **选择性跳过**：q_proj 和 gate_proj（层间敏感度变化大）

具体实现：
- **LLaMA3.1-8B**：q_proj 和 gate_proj 在第 19、21、28、30、31 层跳过，加速 56.1% 线性计算
- **Qwen2-7B**：q_proj 和 gate_proj 在第 0、6、23、26、27 层跳过，加速 57.6% 线性计算
- **Qwen3-30B-A3B**：q_proj 和 gate_proj 在第 41、46、47 层跳过，加速 56.9% 线性计算

### 5. Outstanding-sparse（统一框架）

将 Amber Pruner 与训练后 W8A8 量化（SmoothQuant）集成：

- **SmoothQuant 缩放因子**：$s_j = \frac{\max(|X_{:,j}|)^\alpha}{\max(|W_{:,j}|)^{1-\alpha}}$，$\alpha \in [0, 1]$
- **Outstanding-sparse 重新定义**：$\hat{s}_j = 1/s_j$，显式扩展激活范围，增强稀疏化效果
- **$\alpha = 0.10$**：偏好小 $\alpha$，放大激活异常值，增强稀疏选择性
- **量化敏感层跳过**：对不同模型的线性投影有选择地跳过量化

**与 Amber Pruner 的协同机制**：
- 量化（SmoothQuant）将激活范围扩展，暴露更多结构化稀疏模式
- 稀疏化（Amber Pruner）选择性保留关键激活，降低量化误差
- 两者堆叠实现预填充和解码阶段的双重加速

---

## 实验结果

### 1. 实验设置

**模型**：
- Dense 模型：LLaMA3.1-8B-Instruct、Qwen2-7B-Instruct
- Sparse MoE 模型：Qwen3-30B-A3B

**稀疏率**：2:4、4:8、8:16

**评估任务**：
- Zero-shot：ARC-Challenge、ARC-Easy、BoolQ、MMLU、CEVAL、OpenBookQA、PIQA、RTE、Winogrande
- Few-shot：GSM8K（5-shot）
- 长上下文：LongBench（如 TriviaQA）

**平台**：8× Ascend 910B 处理器

### 2. Amber Pruner 主要结果

| 模型 | 稀疏率 | 方法 | 平均准确率下降 |
|------|--------|------|----------------|
| LLaMA3.1-8B | 2:4 | Naïve top-k | -10.3% |
| LLaMA3.1-8B | 2:4 | Amber-P (l.s.) | **-2.7%** |
| LLaMA3.1-8B | 8:16 | Naïve top-k | -5.4% |
| LLaMA3.1-8B | 8:16 | Amber-P (l.s.) | **-1.4%** |
| LLaMA3.1-8B | 8:16 | Amber-P (all) | **-0.7%** |
| Qwen2-7B | 2:4 | Naïve top-k | -8.3% |
| Qwen2-7B | 2:4 | Amber-P (l.s.) | **-1.7%** |
| Qwen2-7B | 8:16 | Naïve top-k | -4.5% |
| Qwen2-7B | 8:16 | Amber-P (l.s.) | **-1.4%** |
| Qwen2-7B | 8:16 | Amber-P (all) | **-0.8%** |
| Qwen3-30B-A3B | 8:16 | Naïve top-k | -3.1% |
| Qwen3-30B-A3B | 8:16 | Amber-P (l.s.) | **-0.4%** |

**关键发现**：
- 随着 M 值增大（8:16），性能更好，8:16 稀疏度下 Zero-shot 准确率损失低于 1%
- Robust-Norm Scoring 进一步提升精度，尤其在 8:16 稀疏度下
- 对 MoE 模型具有强泛化性（Qwen3-30B-A3B 仅激活 3B 参数，仍表现良好）
- Layer Skipping 是有价值的技术，选择性跳过敏感层可显著提升精度

### 3. Outstanding-sparse 结果

| 模型 | 稀疏率 | 方法 | 平均准确率下降 |
|------|--------|------|----------------|
| LLaMA3.1-8B | 8:16 | O-sparse (all) | **-1.1%** |
| Qwen2-7B | 8:16 | O-sparse (all) | **-0.4%** |
| Qwen3-30B-A3B | 8:16 | O-sparse (l.s.) | **-0.8%** |

**关键发现**：
- 量化与稀疏化协同有效，可实现可控的精度损失
- Outstanding-sparse 重塑激活分布，暴露更多稀疏模式（原始 BFloat16 激活中未观察到）
- **稀疏化是主要精度瓶颈**：相比量化，激活稀疏化对精度的影响更大
- **MoE 模型兼容性好**：2:4 稀疏下仅 1.5% 精度下降，8:16 下低于 1.0%

### 4. Few-shot 和 LongBench 结果

| 模型 | 稀疏率 | GSM8K 下降 | LongBench 下降 |
|------|--------|-----------|---------------|
| LLaMA3.1-8B | 2:4 | -0.6% | -1.7% |
| LLaMA3.1-8B | 8:16 | +0.9% | +0.3% |
| Qwen2-7B | 8:16 | -0.4% | +0.2% |
| Qwen3-30B-A3B | 8:16 | +2.1% | -0.2% |

- 生成性能保持稳定，GSM8K 和 LongBench 上无显著精度下降
- Qwen3-30B-A3B 的 thinking mode 结果显示 prefill 阶段稀疏对 KV cache 的影响不足以影响解码阶段的生成质量

### 5. 权重稀疏 vs 激活稀疏对比（Appendix A）

在 LLaMA3.1-8B-Instruct 上，即使使用简单的 Naïve top-k 激活稀疏化，也显著优于 SparseGPT、Wanda、Pruner-Zero 等权重稀疏方法：

| 稀疏度 | 权重稀疏 (SparseGPT) | 权重稀疏 (Wanda) | 激活稀疏 (Naïve top-k) |
|--------|---------------------|-----------------|---------------------|
| 2:4 | -13.6% | -18.0% | **-10.3%** |
| 4:8 | -10.3% | -12.2% | **-7.4%** |

---

## 优势

1. **训练无关（Training-free）**：无需重新训练或微调，直接在推理时应用，部署简便，可扩展性强
2. **显著的加速效果**：在 8:16 稀疏度下加速超过 55% 的线性投影计算，同时保持零样本准确率损失低于 1%
3. **鲁棒的精度保持**：Robust-Norm Scoring 和 Layer Skipping 策略有效平衡了精度与加速，8:16 下平均准确率损失仅 0.7%
4. **与量化正交且兼容**：Outstanding-sparse 将激活稀疏与 W8A8 量化结合，实现协同优化，不冲突
5. **跨模型泛化性**：在 Dense 模型（LLaMA3.1-8B、Qwen2-7B）和 Sparse MoE 模型（Qwen3-30B-A3B）上均表现良好
6. **生成能力保持**：GSM8K 和 LongBench 上无显著性能下降，生成质量未受影响
7. **激活稀疏优于权重稀疏**：实验表明即使简单的激活稀疏化也优于成熟的权重稀疏方法（SparseGPT、Wanda）
8. **低开销辅助权重**：Robust-Norm Scoring 系数可预计算离线，开销 < 0.5% 模型大小，可通过算子融合实现高效计算

---

## 局限

1. **动态 Mask 计算开销大**：虽然权重是静态的，评分系数可预计算，但运行时仍需动态计算 top-k mask，开销较大
2. **硬件限制**：当前通用硬件对 SpMM（稀疏-密集矩阵乘法）支持有限，实际加速收益可能不如理论预期
3. **仅针对 Prefill 阶段**：方法专门针对 prefill 阶段设计，未涉及 decode 阶段的优化
4. **量化敏感层跳过**：Outstanding-sparse 需要为不同模型设计不同的量化跳过策略，增加了部署复杂性
5. **MoE 模型中 Robust-Norm Scoring 不适用**：由于 token 动态路由到不同专家，Robust-Norm Scoring 无法直接应用于 MoE 模型
6. **稀疏度受限于 N:M 结构**：稀疏度由 N:M 结构决定（如 2:4、4:8、8:16），灵活性有限
7. **评估基准局限**：主要在 Zero-shot、Few-shot 和 LongBench 上评估，未涉及更复杂的推理或代码生成任务
8. **无训练感知优化**：与训练感知方法（如 Squared-ReLU、ProSparse）相比，训练无关方法的性能天花板可能较低
9. **无代码开源**：当前无公开代码实现，可复现性受限

---

## 与 EfficientPaper 相关的研究方向

1. **N:M 稀疏与激活稀疏**：Amber Pruner 与 Wanda、SparseGPT 等权重稀疏方法形成互补，与 Q-Sparse、TEAL 等激活稀疏方法形成对比
2. **Prefill 阶段优化**：针对 LLM 推理 prefill 阶段的加速，可与 FlashAttention、Ring Attention 等注意力优化方法结合
3. **稀疏与量化协同**：Outstanding-sparse 将 N:M 激活稀疏与 W8A8 量化结合，与 SDQ、JSQ、GQSA 等方法形成对比
4. **训练无关剪枝**：Amber Pruner 是训练无关的激活剪枝方法，可与训练感知方法（如 SLoPe、S-STE、AST）进行对比
5. **MoE 模型优化**：Amber Pruner 在 MoE 模型（如 Qwen3-30B-A3B）上的泛化能力，可与 MoE 模型的专家选择和路由策略结合
6. **硬件对齐稀疏计算**：N:M 稀疏与 GPU/NPU 硬件加速的对齐，与 FlashAttention、Triton 等硬件优化方法相关
7. **结构化稀疏与低秩分解**：Amber Pruner 的结构化稀疏与 R-Sparse（top-k 激活稀疏 + 低秩补偿）等方法相关
8. **LLM 推理效率优化**：Amber Pruner 为 LLM 推理的线性投影层加速提供了新方向，可与其他推理加速技术（如 KV Cache 压缩、投机解码）结合
9. **权重稀疏 vs 激活稀疏**：Amber Pruner 的实验表明激活稀疏优于权重稀疏，为未来研究提供了重要的对比基准

---

## 元数据

- **论文标题**: Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models
- **作者/机构**: Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang (Huawei)
- **发表渠道**: arXiv
- **年份**: 2025
- **代码**: 无
- **关键词**: sparse_pruning, activation_sparsity
- **更新时间**: 2025-06-05

---

*本文由 AI Agent 自动生成，基于论文 PDF 原文提取与分析，仅供参考。*
