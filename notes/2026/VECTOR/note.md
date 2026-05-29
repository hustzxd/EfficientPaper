# A Simple Plug-in for Improving Eviction-Based KV Cache Compression

> Yuping Lin, Jiayuan Ding, Yue Xing, Pengfei He, Jiliang Tang, Subhabrata Mukherjee

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

KV cache growth is a major bottleneck for long-context inference in large language models. Existing methods are often dominated by binary eviction or representation approximation, which may underutilize tokens that are not critical for exact retention but are still reconstructable. We present VECTOR, a plug-and-play augmentation for eviction-based pipelines that introduces three-way token routing: retention, approximation, and eviction. VECTOR combines an importance signal from the base scorer with a reconstructability signal from an offline-calibrated regression-based value estimation. By leveraging reconstructability, VECTOR recovers useful value information that would otherwise be irreversibly lost under binary eviction, while preserving key vectors for attention routing stability. Experimental results show that VECTOR improves quality-memory trade-offs under medium-to-high compression, with especially clear gains in stricter budget regimes.

## 一句话总结

VECTOR 是一个即插即用的 KV cache 压缩增强框架，通过在二元 eviction 基础上引入"近似"第三态（保留 Key、用 OLS 线性回归从 Key 重建 Value），在不增加显存占用的前提下恢复被 eviction 丢失的 Value 信息，在中高压缩率下显著提升长上下文推理质量。

## 背景与问题

LLM 长上下文推理的 KV cache 线性增长是部署的核心瓶颈。现有 KV cache 压缩方法主要分两类：

1. **Importance-based eviction**（如 SnapKV、KeyDiff、KVzip）：根据 token 重要性评分直接丢弃低分 token。简单高效，但决策是二元的——token 要么完整保留，要么完全丢弃。在高压缩率下，这种不可逆的 eviction 会导致严重的性能下降。
2. **Representation approximation**（如 AQUA-KV、EliteKV、DeltaKV）：通过量化、投影或重建来压缩 KV 表示。但很多方法需要模型架构修改、重训练或昂贵的在线计算。

**核心洞察：**
- 现有 eviction 方法忽视了 **reconstructability**（可重建性）维度：有些被 evict 的 token 的 Value 实际上可以从 Key 线性重建。
- Key 和 Value 对压缩误差的敏感性是不对称的：Key 扰动会被 softmax 非线性放大，而 Value 扰动仅通过加权求和线性传播。因此，近似 Value 比近似 Key 安全得多。

## 核心方法

VECTOR (Value Estimation via Collinearity and Three-way Orthogonal Routing) 是一个 plug-and-play 框架，在 eviction-based 管线基础上引入三路 token 分配：

| 路由 | Key | Value | 内存消耗 |
|------|-----|-------|---------|
| **Retain** | 保留 | 保留 | 完整 KV pair |
| **Approximate** | 保留 | 丢弃，运行时用 OLS 从 K 重建 | 仅 Key |
| **Evict** | 丢弃 | 丢弃 | 0 |

### 三步 Pipeline

1. **Budget Relaxation**：调用 base eviction scorer 识别一个扩展候选池（大小为 1 - p_c + p_a），其中 p_c 为压缩率，p_a 为近似率。
2. **Residual Evaluation**：对候选池中每个 token，用离线校准的 OLS 投影矩阵 W_OLS 计算 per-token 重建误差 ε_i = ||V_i - W_OLS K_i||²。
3. **Asymmetric Truncation**：
   - 候选池中所有 token 的 Key 被保留
   - 重建误差最小的 2p_a 个 token 进入 Approximate（丢弃 Value，运行时重建）
   - 剩余 1 - p_c - p_a 个 token 完整保留

最终内存占用与纯 eviction 的 1 - p_c 完全相同，但额外恢复了 Approximate 层的信息。

## 技术细节

### OLS Value 估计

由于 K = W_K h 和 V = W_V h 都是同一隐藏状态 h 的线性投影，它们共享低秩结构，使得从 K 线性预测 V 成为可能。

OLS 目标函数：
```
W_OLS = arg min_W E_{h~D} ||W K - V||²_F
```

论文选择 OLS 而非 Moore-Penrose 伪逆，因为 MP 伪逆最小化的是 h 的重建误差（||W_K h' - K||²），而非目标 V 的预测误差（||V - V'||²）。

### RoPE 解耦

现代 LLM 普遍使用 Rotary Position Embedding，Key 被位置相关旋转（K_post-RoPE = R_m W_K h），而 Value 与位置无关。为避免为每个位置训练不同的 OLS 矩阵，VECTOR 在近似阶段对缓存的 Key 施加逆旋转 R⁻¹_m，从而暴露 K-V 的内在共线性，使静态 W_OLS 能准确重建 V。

### 理论分析

论文定义了 importance-weighted signal loss E，并推导了 Proposition 1：扩展 Approximation 层能降低全局失真的条件是：

```
R²_approx > w̄ / (w* + w̄)
```

其中 w̄ 是扩展候选池的平均重要性分数，w* 是 eviction 边界处 token 的重要性分数。对于重要性分数高度偏斜的分布（w̄ ≫ w*），所需 R²_approx 阈值较低，更容易获得正收益。

近似率的实用公式：p_a = min(p_c/2, (1 - p_c - ε)/2)

### 跨模型 K→V 可预测性

论文在 C4 数据集上对多个模型验证了 OLS 的 K→V 预测能力（R²_global）：

| 模型 | R²_global |
|------|-----------|
| Llama-3.1-8B | 0.6964 |
| Qwen3-14B | 0.6946 |
| Qwen3-0.6B | 0.9392 |
| Gemma-3-4B | 0.8896 |
| Qwen3-30B-A3B | 0.6863 |

## 实验设置

- **框架**：KVPress evaluation framework
- **模型**：Llama-3.1-8B-Instruct, Qwen3-14B, Qwen3-0.6B
- **基准**：LongBench（16 tasks，排除 5 个中文任务）, Needle-in-a-Haystack (NIAH)
- **压缩率**：p_c ∈ {0.25, 0.50, 0.75, 0.90}，对应 p_a ∈ {0.125, 0.25, 0.125, 0.05}
- **Baseline**：KeyDiff, SnapKV, KVzip, PyramidKV（及其 +VECTOR 变体）
- **校准**：C4 数据集 10,000 序列（长度 4096），per-layer OLS

## 主要结果

### LongBench

**Query-agnostic baselines 改善最显著：**
- KeyDiff + VECTOR 在 Qwen3-14B 上：p_c=0.50 时平均 +7.03，p_c=0.75 时 +9.15，p_c=0.90 时 +9.73
- KVzip + VECTOR 在 Llama-3.1-8B 上 p_c=0.90 时 +3.97

**Query-aware baselines 改善较温和：**
- SnapKV + VECTOR 在 p_c=0.90 时跨模型一致正向，但中等压缩率下改善有限
- PyramidKV 结果较为分散，改善趋势不明显

**关键发现**：收益在中高压缩率（p_c = 0.75, 0.90）下最为明显，低压缩率下 base eviction 已保留足够上下文。

### NIAH

在 p_c=0.90 下的 NIAH 热力图显示，VECTOR 不仅提升了平均分，还改变了失败模式——从大面积连续低分区域变为更局部化的困难单元，表明检索鲁棒性增强。

### 近似率敏感性

在 p_c = 0.50, 0.75, 0.90 下扫描 p_a，性能先升后降，p_a 过小则近似层太少，p_a 过大则 retention 层被压缩、高重要性 token 被迫近似。公式值 p_a 处于较优区间。

## 优点与局限

**优点：**
- 即插即用，无需修改模型架构或重训练
- 仅需一次离线 OLS 校准，运行时开销低
- 理论分析给出了 R²_approx 与压缩性能的显式关系
- 对 query-agnostic eviction 方法提升显著

**局限：**
- p_a 由经验公式设定，未逐样本/逐层优化
- 对 query-aware 方法（SnapKV、PyramidKV）增益有限，因为这些方法已有效保留了查询相关 token
- 离线校准需要额外的计算和数据（C4 数据集 10k 序列）

## 与 EfficientPaper 主题的关系

VECTOR 属于 **KV cache compression** 领域，具体聚焦于 **KV cache management** 和 **approximation** 的交叉方向。它提出了一种新的 token 分配范式（三路 vs 二路），将 reconstructability 作为 eviction 之外的第二分配维度。这与 EfficientPaper 中的 H2O、SnapKV、PyramidKV 等 eviction 方法互补，也与 AQUA-KV、EliteKV 等 approximation 方法有概念联系但实现路径完全不同。

该工作的核心创新在于利用 K-V 的线性共线性做 Value 近似，而非传统的量化或低秩投影，是一个轻量级但有效的方向。

## 可复现/实现要点

1. **OLS 校准**：使用 C4 数据集，10,000 个 4096-token 序列，逐层拟合 W_OLS（标准最小二乘）
2. **RoPE 解耦**：在近似阶段对 Key 施加逆旋转（R⁻¹_m），计算开销极低
3. **近似率选择**：p_a = min(p_c/2, (1 - p_c - ε)/2)，ε > 0 为小常数
4. **集成方式**：替换 base eviction scorer 的输出为三路分配，不修改原有评分逻辑
5. **评估框架**：KVPress evaluation framework（统一长上下文压缩评估协议）

## 个人备注

- RoPE 解耦是关键工程细节，确保跨位置的 OLS 矩阵一致性
- 与 KVQuant/KIVI 等量化方法是正交方向，理论上可组合使用
- query-agnostic vs query-aware 的差异值得关注：reconstructability 维度对 query-agnostic 方法更有价值，因为它捕捉了 importance 之外的正交信息
- 未来方向：scorer-aware allocation（根据 base scorer 的特性自适应分配）、per-layer adaptive p_a
