# Multi-matrix Factorization Attention (MFA)

> Jingcheng Hu, Houyi Li, Yinmin Zhang, Zili Wang, Shuigeng Zhou, Xiangyu Zhang, Heung-Yeung Shum, Daxin Jiang
> 机构：StepFun、清华大学、复旦大学、旷视科技
> 发表：arXiv, 2024
> 链接：http://arxiv.org/abs/2412.19255v2
> 关键词：structure_design

![](../../blank.jpg)

## 一句话总结

MFA 通过低秩矩阵分解在 Query-Key 电路中高效扩展注意力头的数量和维度，在严格 KV cache 预算下实现了与标准 MHA 相当甚至更优的模型容量，并通过 MFA-KR 变体进一步将 KV cache 使用量减少高达 93.7%。

## 摘要翻译

本文提出了一种新型注意力架构——多矩阵分解注意力（MFA）及其变体 MFA-Key-Reuse（MFA-KR）。现有的标准多头注意力（MHA）变体，包括当前最先进的 MLA 方法，在严格的 Key-Value 缓存（KV cache）约束下无法维持与 MHA 同等的性能。MFA 通过在 Query-Key（QK）电路中使用低秩矩阵分解，高效地扩展注意力头的数量和维度，从而增强模型容量。基于 MFA 的基础上，MFA-KR 通过值投影重参数化，将 Key 缓存复用为 Value，进一步降低内存需求。MFA 的设计使其在严格 KV cache 预算下具有强大的模型容量，而 MFA-KR 适用于更苛刻的 KV cache 限制场景，仅带来微小的性能折损。值得注意的是，在大规模实验中，所提架构超越了 MLA，并在将 KV cache 使用量分别减少 56% 和 93.7% 的同时，性能与 MHA 相当。

## 研究动机

### 核心问题

大型语言模型（LLM）在推理阶段面临严重的 KV cache 内存瓶颈。标准 MHA 的 KV cache 内存占用随 batch size 和序列长度线性增长，成为 LLM 解码阶段的主要瓶颈。

### 现有方法的局限

现有的 KV cache 优化方法包括：
- **MQA（Multi-Query Attention）**：通过跨头共享 key/value 投影减少 KV cache，但牺牲了模型容量
- **GQA（Grouped-Query Attention）**：在 MQA 和 MHA 之间取得折中，但仍受制于容量损失
- **MLA（Multi-head Latent Attention）**：通过低秩压缩 key/value 投影并仅缓存 latent，但对位置编码（如 RoPE）的兼容性存在额外复杂度

这些方法在严格的 KV cache 约束下，都无法匹配 MHA 的性能，因为对 key/value 投影的约束限制了注意力模块的容量。

### 关键洞察

本文通过统一的广义多头注意力（GMHA）框架分析发现，注意力头的数量和维度是维持模型容量的关键因素——这一设计方面在现有方法中未被充分探索。因此，需要高效地扩展这些因素来缓解现有 KV cache 保存技术导致的容量退化。

## 方法（技术细节）

### 理论基础：GMHA 框架

MFA 基于广义多头注意力（GMHA）框架进行分析。GMHA 包含所有具有线性 QK 和 VO 电路、逐头 softmax 注意力的多头机制。论文定义了完全参数化双线性注意力（FPBA）作为理论上限，其中每个通道有独立参数化矩阵。

### MFA 核心设计

MFA 的推理和分解表达式为：

**推理公式：**
$$O_i = \sum_{c=1}^{n} \left[ \sum_{j=1}^{i} \phi(x_i S_q Q_c (x_j S_k)^T / \sqrt{d}) \cdot x_j S_v \right] O_c^T$$

**分解公式：**
$$O_i = \sum_{c=1}^{n} \left[ \sum_{j=1}^{i} \phi(x_i (S_q Q_c S_k^T) x_j^T / \sqrt{d}) \cdot x_j S_v O_c^T \right]$$

其中 $S_q, S_k, S_v \in \mathbb{R}^{H \times C}$ 是跨头共享的投影，$Q_c, O_c \in \mathbb{R}^{C \times C}$ 是逐头的投影，$C$ 表示低秩分解维度。

**关键设计要点：**

1. **可扩展的头数**：MFA 允许以最小的参数开销（每增加一个头仅需约 CH 个额外参数）增加头数，且 KV cache 大小与头数无关
2. **增强的头表达力**：每个头的秩为 C，高于其他方法中通常使用的 d（其中 C > d）
3. **单 Key-Value 头技术**：使用共享的 key/value 投影保持最小的 KV cache 使用量

**KV cache 优势**：每 token 的 KV cache 仅为 2C（C 为低秩分解维度），远小于 MHA 的 2H。

### MFA-KR 变体

MFA-KR 在 MFA 的基础上进一步优化：

- **核心思想**：通过值投影重参数化，将 key 缓存复用为 value
- **实现方式**：使用原始 key 投影和轻量级门控投影的组合
- **效果**：在 MFA 的基础上进一步将 KV cache 使用量减半（额外 50% 的节省）
- **性能损失**：微乎其微

### 与其他方法的对比

| 方法 | KV Cache | 参数量 | 头数 | 分解秩 | 总有效秩 |
|------|----------|--------|------|--------|----------|
| FPBA | 2H² | 2H³ | H | H | H² |
| MHA | 2H | 4H² | n | d | nd |
| MQA | 2d | (2+2/n)H² | n | d | nd |
| GQA | 2gd | (2+2g/n)H² | n | d | gd |
| MLA | 2C+dr | H(3C+dr+md)+mC(3d+dr) | m | d | md |
| MFA | 2C | H(3C+mC)+mC² | m | C | mC |

MFA 实现了最高的总有效秩（TER），使其最接近 FPBA 的理论容量上限。

### 与 MLA 的关键区别

- MFA 天然兼容 RoPE 等位置编码，无需额外操作
- MLA 需要为 RoPE 提供额外的操作支持
- MFA 对初始化方法更加鲁棒，而 MLA 对初始化敏感

## 实验结果

### 7B MoE 模型实验（1T tokens 训练）

在 7B 参数的 MoE 模型上，使用 1T tokens 进行训练，比较 MFA、MFA-KR 与 MHA：

| 指标 | MHA | MFA-KR | MFA |
|------|-----|--------|-----|
| 激活参数 | 1.2B | 1.2B | 1.2B |
| 总参数 | 6.9B | 6.9B | 6.9B |
| KV Cache/Token | 196.6K | 12.3K | 24.6K |
| 平均准确率 | 49.0% | 48.0% | 49.9% |

**关键结果：**
- MFA 的平均准确率（49.9%）超过 MHA（49.0%），同时 KV cache 使用量减少 87.5%
- MFA-KR 进一步将 KV cache 减少至 MHA 的 6.25%（仅 12.3KB/token），性能损失极小
- 在多个下游任务中，MFA 均表现出色或与 MHA 相当

### 消融实验（1B Dense 模型）

- MFA 在验证困惑度（Val PPL）方面优于所有其他架构
- MFA 的 KV cache 使用量仅为 20K/token，远低于 MHA 的 163K/token
- MFA-KR 的 KV cache 为 10K/token，性能与 MHA 相当

### MFA-KR 架构设计消融

| 架构 | KV Cache/Token | Val PPL |
|------|---------------|---------|
| MHA | 163K | 6.41 |
| MFA | 20K | 6.35 |
| +vanilla KR | 10K | 6.55 |
| +extra value proj. | 10K | 7.88 |
| +residual connect | 10K | 6.65 |
| +gating = MFA-KR | 10K | 6.45 |

### 位置编码兼容性

MFA 和 MFA-KR 在使用 ALiBi 位置编码时同样保持优势，验证了架构的通用性。

### 扩展性实验

在 1B-7B 的扩展性实验中：
- MFA 达到了与 MHA 相当的损失缩放曲线
- KV cache 节省随模型规模增大而增长
- MFA-KR 在更大规模下仍保持良好性能

### MLA 初始化敏感性

| 架构 | 初始化方法 | Val PPL |
|------|-----------|---------|
| MLA | Ours | 6.73 |
| MLA | DeepSeek | 6.48 |
| MHA | Ours | 6.41 |
| MFA | Ours | 6.36 |

MFA 对初始化方法不敏感，而 MLA 存在较大的性能差异。

## 优势

1. **容量-效率平衡**：MFA 在严格 KV cache 约束下实现了最高总有效秩，最接近理论容量上限
2. **KV cache 大幅减少**：MFA 减少 87.5%，MFA-KR 减少 93.7%，显著降低推理内存开销
3. **无需额外复杂度**：天然兼容 RoPE 等位置编码，无需 MLA 那样的额外操作
4. **参数效率**：每增加一个头仅需约 CH 个额外参数，KV cache 大小与头数无关
5. **鲁棒性强**：对初始化方法不敏感，无需特殊初始化策略
6. **与现有生态兼容**：可直接集成到当前 LLM 训练和推理生态系统中，无需引入额外架构复杂度
7. **大规模验证**：在 7B MoE 模型和 1T tokens 的大规模实验中得到验证
8. **性能超越 MHA**：MFA 在平均准确率上甚至超过了标准 MHA

## 局限

1. **缺少系统级评估**：未直接评估 KV cache 减少对端到端推理效率的影响，特别是在大规模长上下文模型中的表现
2. **未探索架构组合**：MFA 与 CLA 或线性注意力等其他架构创新的集成未被探索，这些组合可能进一步优化内存使用和性能
3. **未验证更大规模**：MFA 和 MFA-KR 的性能在更大规模模型上尚未得到验证
4. **预训练风险**：虽然进行了详细处理以过滤有害内容，但预训练模型仍可能生成有害或偏见内容
5. **代码未开源**：论文未提供开源代码，限制了社区的复现和改进

## 与 EfficientPaper 相关的研究方向

1. **注意力机制优化**：MFA 代表了一种在 KV cache 约束下优化注意力容量的新思路，与 EfficientPaper 中的结构设计（structure_design）方向密切相关
2. **KV cache 优化**：MFA 和 MFA-KR 提供了在不牺牲性能的前提下大幅减少 KV cache 的方法，这对长上下文推理和资源受限场景尤为重要
3. **低秩分解技术**：MFA 的核心思想——通过低秩矩阵分解扩展注意力容量——可与其他低秩方法（如 LoRA）结合，探索更高效的模型设计
4. **MoE 与注意力协同**：MFA 在 MoE 架构中的实验表明，注意力机制与专家混合模型的协同优化是一个重要方向
5. **推理效率**：KV cache 的减少直接降低推理延迟和内存占用，是实现高效 LLM 推理的关键技术之一

## AI 生成声明

> 本笔记由 AI Agent（Hermes Agent）基于论文 PDF 文本提取和元数据自动生成。内容仅供参考，可能存在信息遗漏或理解偏差。建议结合原文进行核实和深入学习。
