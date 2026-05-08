# SparseForge: Efficient Semi-Structured LLM Sparsification via Annealing of Hessian-Guided Soft-Mask

> Liu Hanzuo, Chaofan Lin, Weixuan Sun, Yulong Wang, Key, Rayying, Mingyu Gao

![111](../../blank.jpg)

## Abstract

Semi-structured sparsity provides a practical path to accelerate large language models (LLMs) with native hardware support, but post-training semi-structured pruning often suffers from substantial quality degradation due to strong structural coupling. Existing methods rely on large-scale sparse retraining to recover accuracy, resulting in high computational cost.
  We propose SparseForge, a post-training framework that improves recovery efficiency by directly optimizing the sparsity mask rather than scaling up retraining tokens. SparseForge combines Hessian-aware importance estimation with progressive annealing of soft masks into hardware-executable structured sparsity, enabling stable and efficient sparse recovery. On LLaMA-2-7B under 2:4 sparsity, SparseForge achieves 57.27% average zero-shot accuracy with only $\textbf{5B}$ retraining tokens, surpassing the dense model's 56.43% accuracy and approaching the 57.52% result of a state-of-the-art method using $\textbf{40B}$ tokens. Such improvements on the accuracy-efficiency trade-off from SparseForge are shown to be consistent across model families.


---

*以下总结由 MiMo 生成：*

这篇论文针对大语言模型半结构化稀疏化中因结构耦合导致精度显著下降的问题，提出了一种名为SparseForge的后训练框架。该方法通过结合Hessian感知的重要性估计与软掩码的渐进退火，直接优化稀疏掩码而非扩大重训练数据量，从而高效生成硬件可执行的结构化稀疏模式。在LLaMA-2-7B模型2:4稀疏度下，SparseForge仅用5B重训练词元就实现了57.27%的零样本准确率，超越了稠密模型的56.43%，并接近使用40B词元的先进方法水平，显著优化了精度与效率的权衡。

---

## 论文详细总结

### 1. 研究背景与动机

半结构化稀疏性（如 2:4 格式）是加速 LLM 的实用路径，硬件原生支持。但训练后剪枝面临严重**质量退化**，原因是强结构性耦合。现有方法依赖大规模稀疏重训练（40B+ tokens），计算成本极高。

### 2. SparseForge 核心思想

**训练后框架**，直接优化稀疏掩码（sparsity mask），而非增加重训练数据量来恢复性能。

### 3. 两大关键技术

| 技术 | 说明 |
|------|------|
| **Hessian-Guided Soft-Mask** | 利用 Hessian 矩阵信息估计权重重要性，生成连续软掩码 |
| **Progressive Annealing** | 渐进式退火，逐步将软掩码硬化为满足 2:4 硬件约束的离散结构化稀疏模式 |

### 4. 实验结果（LLaMA-2-7B, 2:4 稀疏）

| 指标 | 结果 |
|------|------|
| 重训练 token 数 | **5B**（vs 其他方法 40B，减少 **8 倍**）|
| 零样本精度 | **57.27%**（超越稠密模型 56.43%）|
| 接近 SOTA | 用 5B token 接近其他方法 40B token 的 57.52% |

### 5. 核心贡献

1. 将重训练数据需求从 40B 降低到 **5B token**（8 倍减少）
2. 融合 **Hessian 信息 + 渐进式退火**，实现软掩码到硬掩码稳定过渡
3. 极低计算开销下**恢复甚至超越**稠密模型精度，且跨模型族一致性好
