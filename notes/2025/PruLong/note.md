# Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs?

> Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, Danqi Chen

![111](../../blank.jpg)

> **一句话总结**：PruLong 提出 KV footprint 统一度量指标，系统评估长上下文推理中 KV 缓存淘汰方法，并设计基于端到端优化的注意力头角色学习方法 PruLong，在保持长上下文性能的同时，比 DuoAttention 在召回任务上实现 12% 的 KV footprint 缩减。

---

## 摘要翻译

语言模型在书籍摘要等任务中处理越来越长的上下文，但这导致键值（KV）缓存的内存成本不断增长。许多先前的工作提出了从内存中丢弃 KV 的方法，但它们的方法针对有利的场景定制，掩盖了峰值内存高和性能下降等注意事项，而且方法之间的公平比较很困难。本文提出 **KV footprint** 作为统一指标，同时考虑存储的 KV 条目数量及其在内存中的生命周期。我们在保留长上下文理解（高达 128K token）和生成性能的前提下，评估方法所达到的最小 footprint。该指标揭示了先前 KV 淘汰方法的高峰值内存问题。一类方法——**后填充淘汰（post-fill eviction）**——由于与预填充期间的淘汰不兼容，导致 footprint 较高。我们调整这些方法使其能够在预填充期间淘汰 KV，从而大幅降低 KV footprint。然后我们转向**近因淘汰（recency eviction）**方法，提出 **PruLong**——一种端到端优化方法，用于学习哪些注意力头需要保留完整的 KV 缓存，哪些不需要。PruLong 在保持长上下文性能的同时节省内存，比先前方法实现 12% 更小的 KV footprint，同时在具有挑战性的召回任务中保持性能。本文阐明了长上下文推理方法的复杂脉络，为未来最小化 KV footprint 的发展铺平了道路。

---

## 研究动机

### 问题背景

- LLM 推理中 KV cache 内存消耗随序列长度线性增长，是长上下文推理的主要瓶颈。
- 一个 70B 模型处理 128K token 的 prompt 时，KV cache 需要约 42GB 内存——这是长上下文推理的重大资源需求。
- 推理模型（如 DeepSeek-R1）的长链式思维产生长输出序列（数万 token），进一步加剧了内存压力。
- 现有 KV 淘汰方法在公平比较方面存在困难，原因包括：
  - 方法针对不同推理阶段（预填充 vs 解码），各有侧重。
  - 方法使用不同的稀疏性概念（注意力稀疏性 vs KV cache 大小）。
  - 性能保持的容忍度不统一，难以进行公平比较。

### 核心问题

1. **缺乏统一评估指标**：现有方法在不同阶段、不同稀疏性概念下比较，无法公平对比。
2. **后填充淘汰的峰值内存问题**：后填充淘汰方法（如 PyramidKV）在预填充阶段保留全部 KV，导致峰值内存高。
3. **注意力头角色学习不精准**：DuoAttention 等方法使用合成数据和重建损失学习注意力头类型，存在训练-测试差距。
4. **预填充分块大小敏感性**：长上下文推理中 chunked pre-filling 是标准实践，但现有方法对分块大小敏感。

---

## 方法（技术细节）

### 1. 统一框架：KV Footprint 指标

**KV footprint** 定义为所有时间步中未淘汰的 KV 条目数量之和（归一化为全因果注意力）。它同时捕获预填充和解码阶段的内存开销：

$$\text{KV footprint} = \frac{\sum_t (\text{active}_t + \text{inactive}_t)}{\text{full attention entries}}$$

其中每个 KV 条目在任意时刻被分类为：
- **active**（当前步使用）
- **inactive**（已存储但当前步未使用）
- **evicted**（已被淘汰，不再使用）

**关键 KV footprint**：保留原始模型性能的 90%（F=90%）时所需的最小 footprint。该指标揭示了先前方法的峰值内存问题。

### 2. Chunked Eviction：让后填充淘汰兼容预填充

**核心思想**：将预填充分块处理（chunked pre-filling）与淘汰结合，在每个 chunk 处理完后立即淘汰 KV。

**两种实现方式**：
- **朴素分块淘汰（Naive chunked eviction）**：在每个 chunk 的前向传播后，使用 chunk 末尾的 k 个 token（k=64）作为重要性指标，淘汰不重要的 KV。
- **修补分块淘汰（Patched chunked eviction）**：使用原始 prompt 末尾的 k 个 token 作为重要性指标（而非每个 chunk 末尾）。将这些 query token 附加到每个 chunk 末尾，但仅在处理最后一个 chunk 时保留其 KV。

**重要改进**：对于使用 GQA（分组查询注意力）的模型（如所有 Llama 模型），PyramidKV 和 SnapKV 会在每个查询组中复制 KV 条目，导致内存浪费。PruLong 通过基于跨查询的总注意力分数选择单组 KV 条目，减少 8x 内存使用（对 Llama-3.1-8B-Instruct）。

### 3. PruLong：端到端注意力头角色学习

PruLong 将注意力头分为两类（类似 DuoAttention）：
- **Retrieval heads（检索头）**：需要完整 KV cache 以从整个上下文中检索信息。
- **Streaming heads（流式头）**：仅关注最近 token 和少量初始 "sink tokens"，可安全淘汰远距离 KV。

**与 DuoAttention 的三大创新点**：

#### (1) 下一 token 预测损失

- DuoAttention 使用 L2 重建损失（比较原始模型和插值模型的最终隐藏状态）。
- PruLong 直接最小化下一 token 预测损失，更符合文本生成的实际使用方式。

#### (2) 离散掩码优化

- DuoAttention 学习连续门控变量 $z_{i,j} \in [0,1]$，推理时需要离散化为 0 或 1，产生训练-测试差距。
- PruLong 使用 **hard concrete 重参数化**，将 $z_{i,j}$ 视为伯努利分布的二进制掩码，参数化为 $\pi_{i,j}$，通过已建立的剪枝文献中的方法实现端到端优化。

**优化目标**：

$$\max_{\lambda_1, \lambda_2} \min_{\pi} \mathbb{E}_{x \sim D, z \sim \text{Bern}(\pi)} \left[ \frac{1}{N} \sum_{n=0}^{N-1} \log p_\theta(x_{n+1}|x_{:n}; z) \right] + \lambda_1 (s(\pi) - t) + \lambda_2 (s(\pi) - t)^2$$

其中 $s(\pi)$ 是掩码的期望 L0 稀疏度，$t$ 是目标稀疏度，$\lambda_1, \lambda_2$ 是可训练的 Lagrange 参数。目标稀疏度 $t$ 在训练过程中从 0 逐步升温到目标值。

#### (3) 利用自然长上下文数据

- DuoAttention 使用合成针入草堆（passkey retrieval）任务，仅需简单长距离召回。
- PruLong 使用**自然长上下文预训练数据**（如代码仓库、书籍），包含多样化的长距离依赖关系。

### 4. Hard Concrete 重参数化

PruLong 采用 hard concrete 重参数化来优化离散掩码：

1. 从均匀分布采样 $u \sim \text{Uniform}(0,1)$。
2. 通过 Gumbel-softmax 重参数化计算 $s = \sigma(\frac{1}{\tau} \log \frac{u}{1-u} + \log \alpha)$。
3. 将分布拉伸到 $[-0.1, 1]$ 区间，多余概率集中在 0 和 1 上。
4. 当温度 $\tau \to 0$ 时，分布快速收敛到离散支撑集 $\{0,1\}$。

这允许通过蒙特卡洛采样对期望 $E_\pi$ 进行梯度反传，参数 $\log \alpha$ 通过梯度下降学习。

---

## 实验结果

### 评估设置

- **模型**：Llama-3.1-8B-Instruct
- **任务来源**：HELMET（长输入→短输出）和 LongProc（短/长输入→长输出）
- **上下文长度**：128K token，21 个数据集，8 个任务类别
- **任务类别**：Recall、RAG、Re-Ranking、ICL、LongQA、Summarization、HTML→TSV、Travel Planning
- **评估指标**：关键 KV footprint（保留 90% 性能所需的最小 footprint）

### 主要结果（Table 2）

| 方法 | Recall | RAG | Re-Rank | ICL | LongQA | Summ | HTML | Travel |
|------|--------|-----|---------|-----|--------|------|------|--------|
| DuoAttention | 58.0 | 49.0 | 69.0 | 49.0 | 60.0 | 63.0 | 87.0 | 91.0 |
| **PruLong** | **46.0** | **37.0** | **61.0** | **38.0** | **49.0** | **59.0** | **83.0** | 93.0 |
| PyramidKV (Naive) | >93.0 | 44.0 | >94.0 | 42.0 | 62.0 | 53.0 | 97.0 | >98.0 |
| PyramidKV (Patched) | 64.0 | <34.0 | 94.0 | <36.0 | <35.0 | 49.0 | 97.0 | >98.0 |

### 关键发现

1. **PruLong 在召回任务上表现最佳**：比 DuoAttention 减少约 12 个百分点的 KV footprint，在 reranking 和 HTML→TSV 任务上也优于 DuoAttention。
2. **修补分块淘汰对后填充方法至关重要**：使 PyramidKV 能显著降低 KV footprint，但 PyramiD 在 recall 任务上仍落后于 DuoAttention。
3. **预填充分块大小敏感性**：DuoAttention 和 PruLong 在减小分块大小时（32K→8K）性能下降更明显（高达 20%），但更小的分块尺寸在高 KV 减少时主导 Pareto 前沿。
4. **PruLong 的优势因素**：
   - 使用自然长上下文数据（而非合成数据）效果更好。
   - 精确的正则化（训练时目标稀疏度与评估时一致）。
5. **训练阶段影响**：PruLong 可在 SFT 之前或之后应用，但在 SFT 之后应用时对分块大小更敏感。
6. **无人能在所有任务上表现最佳**：在旅行规划等推理密集型任务上，没有任何方法能实现有意义的 KV footprint 减少。

---

## 优势

1. **统一评估框架**：KV footprint 指标首次统一了预填充和解码阶段的内存评估，使不同方法的公平比较成为可能。
2. **端到端优化**：通过下一 token 预测损失和 hard concrete 重参数化，PruLong 直接优化最终任务性能，避免了 DuoAttention 的训练-测试差距。
3. **12% KV footprint 缩减**：在召回任务上比 DuoAttention 减少约 12% 的 footprint，整体表现优于先前方法。
4. **无需修改模型架构**：PruLong 是后处理方法，可应用于已预训练的模型，无需重新训练整个模型。
5. **自然数据训练**：使用自然长上下文数据（如代码仓库和书籍）而非合成数据，更贴近真实应用。
6. **与量化等方法正交**：PruLong 可与 KV cache 量化（如 AWQ、GPTQ）结合使用，进一步减少内存。
7. **开源代码**：完整代码已发布（https://github.com/princeton-pli/PruLong），便于复现和进一步研究。

---

## 局限

1. **预填充分块大小敏感性**：PruLong 对分块大小敏感，在更小的分块（如 8K）下性能可能下降，这限制了其在实际部署中的灵活性。
2. **推理密集型任务效果有限**：在旅行规划等推理密集型任务上，PruLong 和其他方法都未能实现有意义的 KV footprint 减少。
3. **仅在单一模型上评估**：由于计算限制，实验仅在 Llama-3.1-8B-Instruct 上进行，缺乏对更大模型（如 70B）和不同架构的验证。
4. **KV footprint 的理想化假设**：KV footprint 是一个理想化指标，可能不完美地反映实际吞吐量或硬件利用率。
5. **训练成本**：PruLong 需要额外的训练步骤（即使不更新模型权重），增加了训练成本。
6. **无法在所有任务上保持最佳**：没有单一方法能在所有任务类别上达到最低的 critical KV footprint，需要根据任务特点选择方法。
7. **SFT 前后的权衡**：在 SFT 之前应用 PruLong 可能导致对预填充分块大小的更大敏感性（由于训练-推理分布不匹配）。

---

## 与 EfficientPaper 相关的研究方向

1. **KV Cache 稀疏化**：PruLong 属于 **kv_cache_sparse** 关键词，与 DuoAttention、StreamingLLM、MoA 等方法形成直接竞争关系。
2. **注意力头角色学习**：PruLong 的注意力头分类思想与 DuoAttention（检索头 vs 流式头）、MoA（混合稀疏注意力）密切相关。
3. **预填充优化**：PruLong 的 chunked eviction 与 MInference、FTP 等预填充稀疏方法互补。
4. **KV cache 压缩**：PruLong 与 SnapKV、PyramidKV、H2O、FastGen 等后填充淘汰方法形成对比，且为其提供了改进方向（chunked eviction）。
5. **长上下文推理**：PruLong 的评估使用了 HELMET 和 LongProc 基准，与长上下文 LLM 评估（如 RULER、∞Bench）密切相关。
6. **训练-推理对齐**：PruLong 的端到端优化和对预填充分块大小的敏感性，与长上下文训练方法（如 ProLong）的研究方向相关。
7. **与量化/低精度方法结合**：PruLong 与 AWQ、GPTQ 等 KV cache 量化方法正交，可以组合使用。

---

## 生成声明

> 本文档由 AI Agent（Hermes Agent）自动生成，基于论文原文和元数据信息。生成时间：2025年。所有内容均为中文翻译和分析，可能存在翻译偏差或理解不准确之处，请以原文为准。
