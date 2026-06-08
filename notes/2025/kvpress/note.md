# kvpress: LLM KV Cache Compression Made Easy

![](../../blank.jpg)

> **一句话总结**：kvpress 提出 Expected Attention 方法，通过利用 LLM 激活值的高斯分布特性来估计未来查询对 KV 对的重要性，实现免训练的 KV cache 压缩，同时发布了包含 20+ 种压缩技术的开源基准库。

---

## 摘要翻译

大语言模型（LLM）推理中，键值（KV）缓存的内存消耗是高效推理的主要瓶颈。虽然基于注意力分数的 KV cache 剪枝方法前景光明，但存在关键的实际限制：压缩时无法获取未来 token 的注意力分数，且 Flash Attention 等现代实现不会生成完整的注意力矩阵，导致无法访问过去的分数。为克服这些挑战，我们引入 **Expected Attention**，一种免训练的压缩方法，通过预测未来查询如何关注 KV 对来估计其重要性。我们的方法利用 LLM 激活值的分布特性，以闭合形式计算每个 KV 对的期望注意力分数。这些分数实现了对 KV 对的有原则的排序和剪枝，对残差流的影响极小，从而在不降低性能的情况下实现有效压缩。重要的是，我们的方法在预填充和解码阶段都能无缝运行，在两种场景下都持续超越最先进的基线方法。最后，我们发布了 **KVPress**，一个综合库，使研究人员能够实现和基准测试 KV cache 压缩方法，已包含 20 多种技术。

---

## 研究动机

### 问题背景
- LLM 推理中 KV cache 内存消耗随序列长度线性增长，成为长上下文推理的主要瓶颈。
- 一个 70B 模型处理 100 万 token 时，KV cache 需要约 320GB GPU 显存，远超多数 GPU 容量。
- 推理模型（如 DeepSeek-R1）生成大量中间推理 token，加剧了内存压力。
- 现有架构改进（如多头潜在注意力、滑动窗口注意力）需要在训练时实现，不适用于已预训练的模型。

### 核心问题
1. **未来查询不可知问题**：现有方法依赖历史注意力分数（如 SnapKV、TOVA），但未来查询的注意力分数在压缩时不可用。
2. **Flash Attention 不兼容**：Flash Attention 在前向传播时不物化完整注意力矩阵，使得即使是过去的注意力分数也无法访问。
3. **启发式方法的局限**：基于位置（如 StreamingLLM）、Key 范数（如 KNorm）或 Key 差异（如 KeyDiff）的启发式方法缺乏理论依据，性能不稳定。
4. **仅覆盖单一阶段**：许多方法仅针对预填充或解码，而高效的压缩方法必须在两个阶段都表现良好。

### 动机总结
现有 KV cache 压缩方法要么依赖不可用的注意力分数，要么使用缺乏理论支撑的启发式方法，且通常仅针对推理的单一阶段。因此需要一种理论上严谨、能在预填充和解码阶段都有效工作的免训练压缩方法。

---

## 方法（技术细节）

### 核心思想：Expected Attention

#### 1. 理论基础：残差流贡献度量

在 Transformer 架构中，注意力机制的输出被加到隐藏状态（残差流）上：

$$h^{out}_t = h_t + \sum_{i=1}^{t} a_{ti} W_o v_i$$

其中每个 KV 对 $(k_i, v_i)$ 对输出的贡献为：

$$\|\Delta h_{ti}\| = a_{ti} \|W_o v_i\|$$

这个度量同时考虑了注意力权重（query 对 key 的关注度）和转换后的 value 向量的大小（value 对输出的影响）。

**关键洞察**：如果能计算所有缓存 KV 对的这个分数，就可以选择性地剪枝缓存，移除对输出影响最小的 KV 对。然而，计算这个分数需要未来查询的注意力分数，而这些分数在压缩时不可用。

#### 2. 核心创新：基于未来查询分布估计注意力

**激活值的分布特性**：
- 现代 LLM 的隐藏状态近似服从高斯分布：$h \sim \mathcal{N}(\mu, \Sigma)$
- 通过线性变换，查询也继承高斯特性：$q_t = R_t W_Q h_t \sim \mathcal{N}(\mu_{q_t}, \Sigma_{q_t})$
- 这一特性在 Llama3.1-8B、Qwen3-8B、Gemma3-12B 等多种模型架构上得到验证

**位置平均查询分布**：
为创建可处理的注意力表示，对未来 $T$ 个位置的 RoPE 矩阵取平均：
$$\bar{q} \sim \mathcal{N}(\bar{\mu}_q, \bar{\Sigma}_q)$$
其中 $\bar{\mu}_q = \bar{R} W_Q \mu$，$\bar{\Sigma}_q = \bar{R} W_Q \Sigma W_Q^T \bar{R}^T$，$\bar{R} = \frac{1}{T} \sum_{j=1}^{T} R_{t+j}$

**期望注意力分数**：
利用高斯分布的矩生成函数，期望未归一化注意力分数为：
$$\hat{z}_i = \mathbb{E}_{\bar{q} \sim \mathcal{N}(\bar{\mu}_q, \bar{\Sigma}_q)} \left[ \exp\left(\frac{\bar{q}^T k_i}{\sqrt{d}}\right) \right] = \exp\left(\frac{\bar{\mu}_q^T k_i}{\sqrt{d}} + \frac{k_i^T \bar{\Sigma}_q k_i}{2d}\right)$$

归一化后得到期望注意力权重：
$$\hat{a}_i = \frac{\hat{z}_i}{\sum_{j=1}^{t} \hat{z}_j}$$

**期望贡献度量**：
$$\|\hat{\Delta} h_i\| = (\hat{a}_i + \epsilon) \|W_o v_i\|$$

其中 $\epsilon$ 是小超参数，$\|W_o v_i\|$ 是转换后的 value 向量的范数。

#### 3. 压缩算法

Expected Attention 压缩算法对所有缓存 KV 对按期望贡献度量排序，移除贡献最小的 $r\%$ KV 对（$r \in [0, 1]$ 为压缩比率）。

**伪代码（Listing 1）**：
```python
def compress(queries, keys, values, compression_ratio):
    # 计算查询统计量
    mean_query, cov_query = compute_statistics(queries)
    # 计算未归一化注意力分数 (z_i)
    scores = matmul(mean_query, keys.T) / math.sqrt(d)
    scores += einsum("i,ij,j->", keys, cov_query, keys) / (2 * d)
    # 归一化分数并用 value 范数加权
    scores = softmax(scores, dim=-1) * values.norm(dim=-1)
    # 保留分数最高的 KV 对
    n_kept = int(keys.size(0) * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    return keys[indices], values[indices]
```

#### 4. Head-Adaptive Compression（头自适应压缩）

不同注意力头在模型中承担不同角色。采用自适应的逐层压缩策略（参考 AdaKV），允许更重要的头保留更多 KV 对。

#### 5. KVPress 库

KVPress 是一个基于 PyTorch 的综合库，通过 PyTorch forward hooks 附加到每个注意力层，利用 Hugging Face transformers 的现有管道，无需修改模型架构。

- 使用 forward hooks 在每个注意力层前向传播后触发，计算 KV 对的重要性分数
- 根据选择的压缩策略（如 Expected Attention）选择性地淘汰分数最低的 KV 对
- 已集成 20+ 种压缩技术（包括后训练和可训练方法）
- 提供公开的 KVPress Leaderboard，建立标准化评估协议

---

## 实验结果

### 实验设置

**模型**：
- 预填充：Llama3.1-8B (128k)、Qwen3-8B (32k)、Gemma3-12B (128k)，均为指令微调
- 解码：Qwen-15B-R1、Qwen-7B-R1、OpenMath-Nemotron-14B（推理模型）

**基准测试**：
- 预填充：LongBench（6类任务）、Ruler（4子集：NIAH、VT、CWE、FWE）、Needle in a Haystack（最大 125k tokens）
- 解码：Aime25、MATH-500

**基线方法**：
- 预填充：SnapKV（基于查询注意力分数）、TOVA（基于查询注意力分数）、KeyDiff（基于 key 嵌入距离）、DuoAttention（可训练）
- 解码：KNorm（基于 L2 范数）、StreamingLLM（初始 token 保留）、KeyDiff

**硬件**：8× H100 GPU，batch size 1，bfloat16 精度

### 核心结果

#### 预填充性能

**LongBench**（平均分数）：

| 模型 | 方法 | 0% | 10% | 25% | 50% | 75% | 90% |
|------|------|-----|-----|-----|-----|-----|-----|
| Qwen3-8B | Expected Attention | 48.63 | 48.30 | 50.25 | 50.10 | 48.06 | 39.71 |
| Qwen3-8B | TOVA | 48.63 | 48.41 | 48.14 | 46.49 | 43.19 | 37.21 |
| Qwen3-8B | SnapKV | 48.63 | 48.40 | 47.85 | 46.25 | 42.42 | 34.57 |
| Qwen3-8B | KeyDiff | 48.63 | 48.13 | 46.23 | 40.08 | 29.42 | 20.69 |

**Ruler**（4k 和 16k 上下文长度）：

| 模型 | 方法 | Ruler 4k (50%) | Ruler 16k (50%) |
|------|------|----------------|-----------------|
| Qwen3-8B | Expected Attention | 94.7 | 92.7 |
| Qwen3-8B | KeyDiff | 78.6 | 74.5 |
| Gemma3-12B | Expected Attention | 92.7 | 76.6 |
| Gemma3-12B | KeyDiff | 79.8 | 72.6 |
| Llama3.1-8B | Expected Attention | 92.2 | 86.0 |
| Llama3.1-8B | KeyDiff | 85.5 | 82.6 |

**Needle in a Haystack**：
- Expected Attention 在 50% 压缩比率下，与 DuoAttention 性能相当，显著优于其他基线
- 在最长 125k token 的上下文中，无论针位置和上下文大小，均表现出稳健性能

#### 解码性能

**Aime25**：
- Expected Attention 在所有模型（Qwen-R1-1.5B、Qwen-R1-7B、OpenMath-Nemotron-14B）上持续优于或匹配基线方法
- 在高压缩比率（4×、16×）下优势更明显

**MATH-500**（分数）：

| 模型 | 方法 | 0× | 2× | 4× | 12× |
|------|------|-----|-----|-----|------|
| Qwen-R1-7B | Expected Attention | 0.57 | 0.55 | 0.53 | 0.49 |
| Qwen-R1-7B | KeyDiff | 0.57 | 0.54 | 0.48 | 0.35 |
| Qwen-R1-7B | KNorm | 0.57 | 0.47 | 0.32 | 0.12 |
| Nemotron-14B | Expected Attention | 0.57 | 0.55 | 0.54 | 0.47 |
| Nemotron-14B | KeyDiff | 0.57 | 0.56 | 0.51 | 0.44 |

**关键发现**：
- 2× 压缩时大多数方法性能损失极小，说明推理 token 中存在大量冗余信息
- Expected Attention 在高压缩比率（12×）下优势最为明显
- 在 50% 压缩比率下，Expected Attention 与无压缩基线性能持平，同时实现 KV cache 大小减半

#### 内存节省与效率

**峰值内存使用**（Llama3.1-8B，bfloat16，单张 H100）：
- 序列长度 120k token 时，50% 压缩实现约 15GB 内存节省
- 90% 压缩时内存节省更为显著

**Needle in a Haystack 内存-性能权衡**（Qwen3-8B）：
- 50% 压缩：KV cache 大小从 14.65 GB 降至 7.32 GB，性能无损失
- 90% 压缩：KV cache 大小降至 1.46 GB，性能略有下降

#### 重建误差分析

- Expected Attention 的残差流重建误差 $\|h - h_{compr}\|$ 在所有方法中最低
- 说明其更有效地保留了隐藏状态的完整性，这对维持下游性能至关重要

---

## 优势

1. **理论严谨**：基于 LLM 激活值的高斯分布特性，以闭合形式计算期望注意力分数，有明确的理论依据。
2. **免训练**：无需额外训练或微调，可直接应用于已预训练的模型。
3. **双阶段兼容**：在预填充和解码阶段都能无缝运行，是少数同时适用于两个阶段的方法。
4. **高性能**：在多个基准测试上持续超越最先进的基线方法，尤其在高压缩比率下优势更明显。
5. **无需未来查询信息**：通过预测未来查询分布，解决了现有方法依赖未来查询注意力分数的局限性。
6. **与 Flash Attention 兼容**：不依赖完整注意力矩阵，可与 Flash Attention 等现代实现配合使用。
7. **开源基准库**：发布包含 20+ 种压缩技术的 KVPress 库，提供标准化评估协议。
8. **简单易实现**：伪代码简洁，仅需少量修改即可集成到现有模型中。
9. **低重建误差**：在所有比较方法中，Expected Attention 的残差流重建误差最低，更有效地保留了隐藏状态的完整性。
10. **内存节省显著**：在 50% 压缩比率下，KV cache 大小减半且性能无损失；在 90% 压缩比率下，内存节省更为显著。

---

## 局限

1. **性能不及可训练方法**：Expected Attention 作为免训练方法，其性能不及可训练方法（如 DuoAttention、Dynamic Memory Compression），但这是有意的设计选择，以避免大规模训练的计算成本。
2. **手动设置压缩比率**：用户需要手动指定压缩比率，缺乏自动确定最优压缩水平的机制，这是一个有前景的未来研究方向。
3. **PyTorch 实现未优化**：当前 PyTorch 实现有效展示了理论原理，但未针对效率优化。使用自定义 CUDA 内核的高性能实现将显著提高速度和实用性。
4. **依赖高斯分布假设**：虽然高斯分布假设在多种模型上得到验证，但在某些特殊情况下可能不成立，且存在少量重尾异常值。
5. **统计量计算开销**：在解码阶段，需要维护隐藏状态的统计信息（均值和协方差），虽然使用 128 个 token 的缓冲区，但仍有一定计算开销。
6. **未评估所有模型架构**：主要在 Llama、Qwen、Gemma 等模型上验证，未覆盖更多架构（如 DeepSeek、Mistral 等）。
7. **单线程实现**：当前实现未针对多线程或分布式推理进行优化，可能在大规模部署中存在瓶颈。
8. **未与 KV cache 量化方法结合评估**：虽然理论上可与 KV cache 量化方法（如 KIVI、KVQuant）正交结合，但未进行实验验证。

---

## 与 EfficientPaper 相关的研究方向

### 关键词关联：`kv_cache_sparse`

kvpress 属于 EfficientPaper 项目中 **KV cache 稀疏化** 方向的研究，与以下研究方向密切相关：

1. **KV Cache 压缩与剪枝**
   - 与 H2O（注意力分数丢弃）、SnapKV、TOVA、KeyDiff 等方法直接竞争
   - Expected Attention 的独特之处在于利用期望注意力分数进行有原则的排序，而非依赖不可用的未来查询信息

2. **KV Cache 量化**
   - 与 NQKV、KVQuant、KIVI 等量化方法正交，可结合使用
   - Expected Attention 减少序列维度的 KV 对数量，量化减少每个 KV 对的精度，两者可同时应用

3. **注意力稀疏性与动态选择**
   - 与 DuoAttention（可训练注意力掩码）、AdaKV（自适应预算分配）、PyramidKV（金字塔信息漏斗）等方法相关
   - Expected Attention 提供了基于分布特性的重要性估计方法

4. **推理引擎优化**
   - 与 vLLM（PagedAttention）、TensorRT-LLM 等推理引擎的 KV cache 管理相关
   - KVPress 库可作为推理引擎的组件集成

5. **长上下文推理优化**
   - 与 LongLoRA、YaRN 等上下文扩展方法互补
   - Expected Attention 使在有限内存下处理更长上下文成为可能

6. **工具与框架**
   - KVPress 作为研究框架，支持快速实现和测试新的压缩方法
   - 公开 Leaderboard 促进公平的基准比较
   - 元数据中的 `tool` 关键词体现了其作为研究工具的定位

### 核心研究价值
kvpress 提出了一个基于期望注意力的 KV cache 压缩方法，通过利用 LLM 激活值的高斯分布特性，实现了对未来查询注意力分数的估计。这一方法不仅在理论上严谨，而且在实践中表现出色，同时发布了包含 20+ 种压缩技术的开源基准库，为 KV cache 压缩领域提供了标准化的评估平台。该方法与现有技术高度互补，具有很强的实用价值和研究意义。

---

*本 note 由 AI Agent（Hermes Agent）自动生成。*
*生成时间：2025年6月4日*
*论文来源：arXiv:2510.00636（2025年10月）*
*论文 URL：https://github.com/NVIDIA/kvpress*
*代码仓库：https://github.com/NVIDIA/kvpress*
