# AdaSplash: Adaptive Sparse Flash Attention

> ⚠️ **声明**：本 note 由 AI Agent（Hermes Agent）于 2025 年 6 月自动生成，基于论文原文的全文提取与中文总结。内容可能存在遗漏或翻译偏差，建议结合原文阅读。

![](../../blank.jpg)

## 一句话总结

AdaSplash 提出了一种硬件感知的 α-entmax 稀疏注意力实现，通过混合 Halley-bisection 算法和自定义 Triton 内核，将自适应稀疏注意力的计算效率提升至与 FlashAttention-2 可比的水平，同时保持强大的下游任务性能。

---

## 摘要翻译

基于 softmax 的注意力机制的计算代价限制了 Transformer 在长上下文任务中的应用。自适应稀疏性（α-entmax 注意力是一个例子）提供了一种灵活的、数据依赖的替代方案，但现有实现效率低下，且未利用稀疏性来获得运行时和内存增益。在本文中，我们提出了 AdaSplash，它将 GPU 优化算法的效率与 α-entmax 的稀疏性优势相结合。我们首先引入了一种混合 Halley-bisection 算法，使计算 α-entmax 变换所需的迭代次数减少了 7 倍。然后，我们实现了自定义 Triton 内核来高效处理自适应稀疏性。在 RoBERTa 和 ModernBERT（用于文本分类和单向量检索）以及 GPT-2（用于语言建模）上的实验表明，我们的方法在运行时和内存效率方面相比现有 α-entmax 实现取得了显著改进。它接近——在某些情况下超越——高度优化的 softmax 实现（如 FlashAttention-2）的效率，从而在保持强大任务性能的同时支持长上下文训练。

---

## 研究动机

Transformer 中的注意力机制依赖 softmax 将概率分配到所有 token 上。对于长上下文输入，小概率的累积会导致注意力分散（dilution）。研究表明，注意力概率倾向于集中在少量 token 上，这暗示可以通过利用注意力稀疏性来提高模型性能和计算效率。

已有的稀疏注意力方法（如窗口稀疏注意力、低秩近似注意力等）往往需要架构修改或粗糙近似，缺乏灵活性和通用性。α-entmax 作为一种自适应稀疏激活函数，虽然可以将无关 token 的概率设为零，但现有实现并未利用这种稀疏性来加速计算，反而比 softmax 注意力更慢，难以扩展到长上下文。

此外，现有硬件优化实现（如 FlashAttention-2）仅支持 softmax，不支持 α-entmax 等复杂变换。因此，本文旨在填补自适应稀疏激活与高效长上下文建模之间的差距。

---

## 方法（技术细节）

### 1. 混合 Halley-bisection 算法（计算 α-entmax）

α-entmax 变换的核心是求解归一化阈值 τ，使得输出概率之和为 1。传统方法使用二分法（bisection），收敛速度为线性，需要大量迭代。

AdaSplash 提出了一种混合算法，结合 Halley 方法（三次收敛）和二分法（保证收敛）：

- **Halley 方法更新**：利用函数 f(τ) 的一阶和二阶导数，更新规则为：
  
  τ_H = τ - 2f(τ)f'(τ) / (2f'(τ)² - f(τ)f''(τ))

- **回退机制**：当 Halley 方法的更新超出二分法的上下界时，回退到二分法更新，保证收敛性。

- **效果**：将迭代次数从 23 次（标准二分法）减少到仅 3 次（达到机器精度），实现约 15 倍的运行时加速，内存使用减少 1.75 倍。

### 2. 自定义 Triton 内核（AdaSplash 前向传播）

借鉴 FlashAttention 的分块（tiling）和重计算（recomputation）策略：

- **分块策略**：将 Q、K、V 矩阵分块，每次只加载少量数据到 SRAM 进行计算，避免将完整的 n×n 矩阵写入 HBM。
- **阈值 τ 的块级计算**：由于 f(τ) 及其导数具有可加性，可以分块累加计算，无需物化整个 S 矩阵。
- **重计算**：在计算 τ 和反向传播时重新计算 S 和 P 矩阵，以减少 HBM 读写（虽然增加了 FLOPs，但总体运行时更优）。

### 3. 稀疏感知实现（块级掩码）

- **动态块掩码 M**：在 Halley-bisection 迭代后构建一个二值块掩码矩阵 M ∈ {0,1}^(Tr×Tc)，标记哪些块包含非零注意力。
- **查找表**：创建两个指针增量查找表：
  - K_j = {i | M_ij = 1}：记录哪些行块对列块 j 有非零贡献
  - Q_i = {j | M_ij = 1}：记录哪些列块对行块 i 有非零贡献
- **效果**：跳过全零块的计算，显著减少不必要的计算。

### 4. 反向传播优化

- **稀疏雅可比矩阵**：α-entmax 的雅可比矩阵是稀疏的，仅依赖于输出 P。
- **分离内核**：将 dQ、dK、dV 的计算分离为不同的 Triton 内核。
- **高效梯度计算**：利用查找表跳过 null 块，仅计算非零块的梯度。
- **额外内存开销**：仅需存储 O(2) ∈ R^(n×d) 和二值掩码 M（可在层间共享）。

---

## 实验结果

### 效率基准测试

- **序列长度扩展**：AdaSplash 可处理高达 64k 的序列长度，而标准二分法在 4k 以上就 OOM。
- **与 FlashAttention-2 比较**：
  - 当稀疏度较低时，AdaSplash 略慢于 FlashAttention-2（因为需要额外的 HBM 读取）。
  - 当稀疏度增加（如 80%+ 块稀疏），AdaSplash 超越 FlashAttention-2。
- **运行时**：在 1024 token 序列上，Torch 二分法需要 36.67 ms，Halley-bisection 仅需 2.38 ms（约 15 倍加速）。

### 下游任务性能

#### 文本分类（ECtHR 长文档分类）
- RoBERTa + α-entmax 在 4096 token 长度下达到 78.0 F1（vs softmax 的 77.9），且内存和运行时与 FlashAttention-2 相当。
- 传统 Torch 二分法在 8192 token 时需要 4 小时 12 分钟，而 AdaSplash 仅需 38 分钟。

#### 单向量检索（BEIR 基准）
- ModernBERT (α=1.5) 在所有任务上一致优于 dense 版本：
  - SciFact: 58.4 vs 57.7
  - NFCorpus: 25.7 vs 22.4
  - FiQA: 29.6 vs 25.7
  - TREC-COVID: 75.2 vs 67.6

#### GLUE 基准
- RoBERTa (α=1.5) 平均得分 83.9（与标准 softmax 相同）。
- ModernBERT (α=1.5) 平均得分 83.5（略低于 dense 版本 83.7，但差异很小）。

#### 语言建模（GPT-2）
- Sparse GPT-2 (α=1.5) 验证损失 3.263（vs softmax 3.283），HellaSwag 准确率 30.6%（vs 30.4%）。
- 运行时与 FlashAttention-2 接近（1.03 s/step vs 0.98 s/step），内存匹配（52.5 GB）。
- 大幅超越 Torch 排序和二分法版本（3.61 和 7.78 s/step）。

### 稀疏性分析
- GPT-2 (α=1.5) 在 1024 token 输入下，除第一层外，所有后续层都表现出高稀疏性。
- ModernBERT (α=1.5) 的整体稀疏度为 95%，(α=2.0) 为 99%。

---

## 优势

1. **高效实现**：混合 Halley-bisection 算法将迭代次数减少至原来的 1/7，Triton 内核充分利用 GPU 硬件特性。
2. **可扩展性**：支持高达 64k 的序列长度，远超传统 α-entmax 实现。
3. **性能保持**：在文本分类、信息检索、语言建模等任务上保持或超越 softmax 注意力的性能。
4. **灵活的稀疏性**：α-entmax 提供输入依赖的自适应稀疏，比固定模式（如窗口注意力）更灵活。
5. **兼容性**：与 FlashAttention-2 的效率可比，可直接替代。
6. **无需架构修改**：作为注意力机制的替代，无需改变模型架构。
7. **模块化**：提供带/不带块掩码的两种模式，用户可根据任务需求和硬件约束选择。

---

## 局限

1. **前向传播开销**：由于需要计算阈值 τ，前向传播始终比 FlashAttention-2 慢（需要额外的 HBM 读取和迭代）。
2. **块掩码开销**：动态块掩码需要额外的 O(Tr × Tc) 内存（虽然可跨层共享，但仍是额外开销）。
3. **推理优化未涉及**：论文主要关注训练效率，对 KV cache 压缩和推理优化的讨论有限。
4. **α 参数选择**：需要手动选择 α 参数，且 α 值会影响稀疏度和性能之间的平衡。
5. **GPU 依赖**：当前实现基于 Triton，依赖 NVIDIA GPU 硬件。
6. **仅验证了中小模型**：实验仅在 RoBERTa (125M)、ModernBERT (149M) 和 GPT-2 (124M) 上验证，未在大型模型上测试。
7. **硬件限制**：实验在 Nvidia A6000 (48GB) 和 H100 (80GB) 上进行，未在消费级 GPU 上验证。

---

## 与 EfficientPaper 相关的研究方向

1. **注意力机制优化**：属于高效注意力的核心研究方向，与 FlashAttention 系列、稀疏注意力、低秩近似注意力等研究直接相关。
2. **自适应稀疏性**：为自适应稀疏注意力提供了高效的硬件实现，推动了 α-entmax 等稀疏激活在实际训练中的应用。
3. **长上下文训练**：解决了长上下文训练中的效率瓶颈，为未来超长序列模型（如百万级 token）提供技术基础。
4. **硬件感知算法设计**：展示了如何利用 GPU 内存层次结构（HBM/SRAM）和 Triton 编程模型来优化复杂注意力计算。
5. **KV cache 优化**：虽然论文未直接涉及 KV cache，但其稀疏注意力模式可与 KV cache 压缩方法结合，进一步提升推理效率。
6. **模型效率与可解释性**：α-entmax 提供的稀疏性不仅提升效率，还可能改善模型的可解释性（通过稀疏注意力分布）。
7. **替代 softmax 的激活函数**：为未来研究替代 softmax 的注意力机制（如 sparsemax、entmax、top-k 等）提供了高效的硬件实现参考。

---

## 参考信息

- **论文标题**：AdaSplash: Adaptive Sparse Flash Attention
- **作者**：Nuno Gonçalves, Marcos Treviso, André F. T. Martins
- **机构**：Universidade de Lisboa (Instituto Superior Técnico, Instituto de Telecomunicações, Unbabel)
- **发表会议**：ICML 2025
- **arXiv**：http://arxiv.org/abs/2502.12082
- **代码**：https://github.com/deep-spin/adasplash
- **关键词**：attention_sparsity
- **基线方法**：FlashAttention-2
