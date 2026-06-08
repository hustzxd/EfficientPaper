# Forgetting Transformer: Softmax Attention with a Forget Gate

> **本文由 AI Agent 自动生成，基于 arXiv 论文全文阅读撰写。生成日期：2025-06-04。**

![](../../blank.jpg)

## 一句话总结

FoX（Forgetting Transformer）在标准 softmax attention 中引入数据依赖的遗忘门机制（forget gate），通过在注意力 logits 上加减对数遗忘偏置，实现了类似循环模型的遗忘能力，同时保留了 Transformer 在长上下文建模和检索方面的优势，且兼容 FlashAttention，无需位置编码。

---

## 摘要翻译

现代循环序列模型的一个核心组件是遗忘门（forget gate）。尽管 Transformer 没有显式的循环形式，我们证明可以通过以数据依赖的方式降低未归一化注意力分数的权重，将遗忘门自然地融入 Transformer 中。我们将这种注意力机制命名为**遗忘注意力（Forgetting Attention）**，对应的模型称为**遗忘 Transformer（FoX）**。我们证明 FoX 在长上下文语言建模、长度外推和短上下文下游任务上优于 Transformer，同时在长上下文下游任务上与 Transformer 持平。此外，它兼容 FlashAttention 算法，且不需要任何位置编码。多项分析（包括"大海捞针"测试）表明，FoX 也保留了 Transformer 相对于 Mamba-2、HGRN2 和 DeltaNet 等循环序列模型在长上下文能力方面的优势。我们还引入了"Pro"块设计，整合了循环序列模型中常见的几种架构组件，发现其显著提升了 FoX 和 Transformer 的性能。

---

## 研究动机

1. **循环模型 vs Transformer 的困境**：虽然近年来循环序列模型（如 Mamba-2、HGRN2、DeltaNet）受到广泛关注，但在长上下文能力方面仍不如 Transformer，可能因为其固定大小的隐状态限制了信息容量。

2. **Transformer 缺乏遗忘机制**：Transformer 在处理长上下文方面表现优异，但缺乏以数据依赖方式遗忘过去信息的显式机制。而这种机制（遗忘门）在循环模型的成功中被证明至关重要。

3. **关键问题**：能否在 Transformer 中引入遗忘门？作者利用了一个重要事实——许多带遗忘门的循环序列模型可以写成与 softmax attention 类似的并行线性注意力形式（如 GLA、Mamba-2），从而将遗忘门机制自然地迁移到 softmax attention 中。

---

## 方法（技术细节）

### 2.1 核心思想：从循环线性注意力到遗忘注意力

标准 softmax attention 的输出为：

$$o_i = \frac{\sum_{j=1}^{i} \exp(q_i^\top k_j) v_j}{\sum_{j=1}^{i} \exp(q_i^\top k_j)}$$

带遗忘门的线性注意力（如 GLA、Mamba-2）的并行形式为：

$$o_i = \frac{\sum_{j=1}^{i} F_{ij} \cdot k_\phi(q_i, k_j) \cdot v_j}{\sum_{j=1}^{i} F_{ij} \cdot k_\phi(q_i, k_j)}$$

其中 $F_{ij} = \prod_{l=j+1}^{i} f_l$，$f_l$ 是每个时间步的遗忘门值。

**关键洞察**：如果将上述公式中的核函数 $k_\phi$ 替换为指数点积核 $k_{\exp}$，就得到了带遗忘门的 softmax attention——即 FoX。

### 2.2 FoX 的注意力计算

FoX 在每个时间步计算一个标量遗忘门 $f_t = \sigma(w_f^\top x_t + b_f) \in \mathbb{R}$，注意力输出为：

$$o_i = \frac{\sum_{j=1}^{i} F_{ij} \exp(q_i^\top k_j) v_j}{\sum_{j=1}^{i} F_{ij} \exp(q_i^\top k_j)} = \frac{\sum_{j=1}^{i} \exp(q_i^\top k_j + D_{ij}) v_j}{\sum_{j=1}^{i} \exp(q_i^\top k_j + D_{ij})}$$

其中 $D_{ij} = \log F_{ij} = \sum_{l=j+1}^{i} \log f_l$。

矩阵形式：
$$O = \text{softmax}(QK^\top + D) V$$

这里 $D \in \mathbb{R}^{L \times L}$ 是下三角矩阵，非零元素为 $D_{ij}$。这等价于在注意力 logits 上添加了一个数据依赖的偏置。

### 2.3 硬件感知实现

上述 logit bias 形式可以通过对 FlashAttention 算法进行简单修改来实现：
1. 预先计算 $c_i = \sum_{l=1}^{i} \log f_l$ 并存储在 GPU 高带宽内存（HBM）中
2. 在 FlashAttention 的 SRAM 计算中，同时加载 $c_i$ 和 $c_j$，计算 $D_{ij} = c_i - c_j$，加到 attention logit 上
3. 避免实例化 $L \times L$ 的 $D$ 矩阵，额外计算和参数开销可忽略不计

### 2.4 与 ALiBi 的关系

FoX 也可以视为 ALiBi 的数据依赖和可学习版本。ALiBi 使用固定、与头相关、数据无关的遗忘门 $f_t^{(h)} = \exp(-m_h)$，而 FoX 使用数据依赖的遗忘门。实验表明数据依赖遗忘门优于 ALiBi。

### 2.5 位置编码

虽然 RoPE 有时能轻微提升 FoX 性能，但 FoX 默认不需要任何位置编码。消融实验表明，使用 RoPE 对 FoX (Pro) 几乎没有改善。

### 2.6 架构设计

#### FoX (LLaMA)
将 LLaMA 架构中的 RoPE 替换为遗忘门。

#### FoX (Pro)
在 FoX (LLaMA) 基础上增加以下组件：
- **输出门（Output Gate）**：对注意力输出进行门控（类似 GLA 和 Mamba-2）
- **输出归一化（Output Norm）**
- **QK-norm**：对查询和键进行归一化
- **数据依赖的 Token 移位（KV-shift）**：对键和值进行简化的数据依赖位移

键的计算方式：
$$\tilde{k}_t = W_k x_t, \quad \alpha_t^{key} = \sigma(w_k^\top x_t), \quad k_t = \text{RMSNorm}(\alpha_t^{key} \tilde{k}_{t-1} + (1 - \alpha_t^{key}) \tilde{k}_t)$$

值的计算方式类似，但不使用 RMSNorm。

每个 FoX 层的结构为：
- RMSNorm → Forgetting Attention → RMSNorm → ShiftLinear → Linear → Linear → SwiGLU MLP → RMSNorm

---

## 实验结果

### 实验设置
- **数据集**：LongCrawl64（RedPajama-v2 的长序列子集）
- **模型规模**：760M（非嵌入）参数
- **训练数据**：~48B tokens
- **训练上下文长度**：16384 tokens
- **验证上下文长度**：65536 tokens（用于测试长度外推）
- **基线模型**：Transformer (LLaMA/Pro)、Mamba-2、HGRN2、DeltaNet
- **优化器**：AdamW，β₁=0.9，β₂=0.95
- **实现**：基于 Flash Linear Attention 仓库

### 长上下文语言建模
- **主要指标**：不同 token 位置的每 token 损失 L(i) 和不同上下文长度的困惑度 P(l)
- **关键发现**：
  - FoX 在训练上下文长度内外均优于标准 Transformer
  - FoX 维持单调递减的每 token 损失，表明模型有效利用了整个训练上下文
  - 循环序列模型（Mamba-2、HGRN2、DeltaNet）的损失在约 5k token 后开始平坦化，10k 后趋于平台期
  - FoX (Pro) 在绝对损失值和困惑度上明显优于 HGRN2、DeltaNet 和 Mamba-2

### 大海捞针测试（Needle in the Haystack）
- FoX 在训练上下文长度内实现了近乎完美的检索准确率
- Transformer (Pro) 和 Transformer (LLaMA) 在简单模式下也达到完美准确率，但在标准模式下有时失败
- Mamba-2、DeltaNet 和 HGRN2 表现较差
- FoX (Pro)、FoX (LLaMA) 和 Transformer (Pro) 在一定程度上可外推到训练上下文长度之外

### 短上下文下游任务（LM-eval-harness）
在 Wikitext、LAMBADA、PiQA、HellaSwag、WinoGrande、ARC-e、ARC-c、COPA、OBQA、SciQA、BoolQ 等任务上：
- **FoX (Pro) 平均得分 50.88**，最佳
- Transformer (Pro) 50.39
- FoX (LLaMA) 49.82
- Transformer (LLaMA) 49.09
- Mamba-2 50.21
- HGRN2 49.45
- DeltaNet 47.99

### 长上下文下游任务（LongBench）
在 14 个 LongBench 任务上：
- FoX 与 Transformer 表现持平
- FoX 优于循环序列模型（Mamba-2、HGRN2、DeltaNet）
- 不同架构（LLaMA vs Pro）各有优势

### 消融实验（Ablation Studies）
使用 360M 参数模型训练 7.5B tokens：
- 所有组件（遗忘门、QK-norm、输出门、输出归一化、KV-shift）均对 FoX 有正贡献
- 无遗忘门且无 RoPE 的模型表现最差（困惑度 29.30）
- 仅使用遗忘门（无 RoPE）即可获得与使用 RoPE 的 Transformer 相当的性能
- 添加 RoPE 对 FoX (Pro) 几乎无改善

### 数据依赖 vs 数据无关遗忘门
- 数据依赖遗忘门始终优于数据无关和固定遗忘门
- 固定遗忘门（等价于 ALiBi）性能最差

### 模型规模和训练上下文长度的影响
- FoX 相对于 Transformer 的优势随训练上下文长度增加而增大
- 随模型规模增大而减小（大模型可更好地建模长上下文，遗忘机制的重要性降低）
- 长上下文训练会损害短上下文性能（已知现象，可能由于训练批次内文档多样性降低）

---

## 优势

1. **简洁优雅的遗忘机制**：仅需一个标量遗忘门（$w_f$ 和 $b_f$），额外参数和计算开销可忽略不计
2. **兼容 FlashAttention**：无需修改 FlashAttention 的核心算法，仅需简单扩展
3. **无需位置编码**：RoPE 对 FoX 几乎无改善，简化了模型设计
4. **长上下文能力强**：在长上下文语言建模上优于 Transformer，在长上下文下游任务上与 Transformer 持平
5. **短上下文也优秀**：在短上下文下游任务上也优于 Transformer
6. **保留检索能力**：在大海捞针测试中表现优异，接近完美
7. **超越循环模型**：在长上下文能力方面显著优于 Mamba-2、HGRN2、DeltaNet
8. **Pro 架构提升显著**：引入的 Pro 块设计（输出门、输出归一化、QK-norm、KV-shift）显著提升了 FoX 和 Transformer 的性能
9. **代码开源**：基于 PyTorch，可用于公平对比不同架构

---

## 局限

1. **规模有限**：主实验仅使用最大 760M 参数、48B tokens、训练上下文长度 16384 tokens，未在更大规模（如 7B、13B、70B）上验证
2. **仅限因果建模**：未考虑非因果（双向）序列建模场景
3. **长度外推不稳定**：外推行为可能依赖超参数（如训练 token 数和学习率），更多训练 token 可能导致更差的外推（可能"过拟合"到训练上下文长度）
4. **长上下文训练损害短上下文性能**：长上下文训练会导致短上下文性能下降（已知现象）
5. **KV 缓存优化未探索**：虽然提到可能基于遗忘门值进行自适应 KV 缓存驱逐，但未实现和验证
6. **下游任务表现与 Transformer 持平**：在长上下文下游任务上未显示出明显优势

---

## 与 EfficientPaper 相关的研究方向

1. **Attention 机制的高效变体**：FoX 是对 softmax attention 的直接改进，可与 FlashAttention、FlashLinearAttention 等高效注意力方法结合
2. **结构设计（structure_design）**：FoX 属于 Transformer 架构设计的改进方向，与 GLA、Mamba-2、HGRN2 等循环模型在设计空间上互补
3. **KV 缓存优化**：遗忘门机制可用于自适应 KV 缓存驱逐，实现更高效的推理
4. **混合架构探索**：FoX 的遗忘门机制可与循环层、二次注意力层结合，探索更高效的混合架构
5. **Pro 块设计**：FoX (Pro) 的 Pro 块设计（输出门、输出归一化、QK-norm、KV-shift）也被证明对标准 Transformer 有效，可作为通用的架构改进组件
6. **基准线推荐**：作者建议未来工作将 FoX (Pro) 和 Transformer (Pro) 作为基准线，替代常用的 LLaMA 架构
7. **位置编码的简化**：FoX 不需要位置编码，这为 Transformer 架构的简化提供了新思路
8. **遗忘门的可学习性**：数据依赖遗忘门显著优于 ALiBi 等数据无关方法，这一结论可推广到其他注意力改进方法

---

> **声明**：本 note 由 AI Agent（Hermes Agent）基于 arXiv 论文全文自动生成，内容仅供参考，建议阅读原文获取完整信息。
