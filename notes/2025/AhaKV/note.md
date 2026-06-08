# AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models

> Yifeng Gu, Zicong Jiang, Jianxiu Jin, Kailing Guo, Ziyang Zhang, Xiangmin Xu
>
> South China University of Technology / Pazhou Laboratory

> **一句话总结：** AhaKV 通过自适应步增 Softmax（SG-softmax）消除累积注意力分数的位置偏差，并利用 Value 向量模长作为先验权重，实现无偏、全局的 KV Cache 驱逐策略，在固定缓存预算下达到 SOTA 性能。

![](../../blank.jpg)

## 摘要翻译

大语言模型（LLM）在人工智能领域取得了显著进展。然而，其部署成本高昂，不仅因为模型参数量大，还因为推理过程中 KV 缓存消耗大量内存。现有方法通过驱逐不必要的 token 来减少 KV 缓存，但这些方法依赖累积注意力分数作为驱逐分数来衡量 token 的重要性。本文发现累积注意力分数存在偏差——其期望值随 token 位置递减，导致保留的 token 集中在序列初始位置，限制了模型获取全局上下文信息的能力。为此，本文提出 **AhaKV**（Adaptive Holistic Attention KV），通过根据注意力分数信息熵的期望自适应调节 softmax 的缩放来解决累积注意力分数的偏差问题，并利用自注意力机制中被忽视的 Value 向量信息来细化驱逐分数。理论上证明了该方法适合偏差消除。实验表明，AhaKV 在多个基准任务上成功缓解偏差，在全局上下文中保留关键 token，并在多项对比中达到 SOTA 结果。

## 研究动机

### 背景问题
- LLM 推理中的 **KV 缓存内存瓶颈**：以 LLaMA-2-7B 为例，当 batch size=8、上下文长度达 32K tokens 时，KV 缓存可达 128GB，远超模型参数本身的内存需求。
- 现有 KV 缓存驱逐方法（如 H2O）依赖**累积注意力分数**作为驱逐分数，但存在两个关键问题：

### 问题一：累积注意力分数的位置偏差（Positional Bias）
- 由于因果掩码（causal mask），位置 j 的 token 只能被位置 i≥j 的 token 注意到，因此 token j 的累积注意力分数为 $S_j = \sum_{i=j}^{n} a_{i,j}$，其累加项数量随 j 增大而减少。
- 通过数学证明：$E(S_{j+1} - S_j) < 0$，即累积注意力分数的期望值单调递减。
- 实际表现为：位置靠后的 token 被系统性低估重要性，导致大量上下文信息丢失（如 3600 token 序列中，位置 1500 之后的 token 被不成比例地驱逐）。

### 问题二：忽视 Value 向量信息
- 现有方法仅使用 Query 和 Key 的信息，忽略了 Value 向量在自注意力机制中的关键作用。
- Value 向量携带上下文预测信息，利用 Value 的模长可以提供额外的 token 重要性信号。

## 方法（技术细节）

AhaKV 由三个核心组件组成，共同实现无偏、全局的 KV 缓存驱逐：

### 1. 步增 Softmax（Step Gain Softmax, SG-softmax）

**动机：** 随着序列长度增加，softmax 将固定总概率（和为 1）分配到更多 token 上，导致每个 token 的注意力分数被稀释（平铺化）。

**方法：** 引入缩放参数 λ，对 softmax 进行自适应调整：
$$\text{SG-softmax}(x_i, \lambda_i) = \frac{\lambda x_i}{\sum_{j=0}^{n} e^{\lambda x_j}}$$

**理论推导：** 
- 信息熵的期望为 $E[H_i] = \log i - \frac{\lambda^2 d}{2}$
- 目标：使期望信息熵等于最大信息熵（与 token 数量无关），即 $\log i - \frac{\lambda^2 d}{2} = -\log \frac{1}{k}$
- 求解得：$\lambda = \sqrt{\frac{2 \log(i/k)}{d}}$，其中 i 为总 token 数，k 为缓存预算。

### 2. 近期累积（Recent Accumulation）

**动机：** 直接对所有行累积注意力分数，位置靠前的 token 有更多累积项，导致偏差。

**方法：** 仅对最近 r 行的注意力分数进行累积，确保每个 token 的累积项数量一致：
$$S_j = \sum_{i=n-r}^{n} a_{i,j}$$

### 3. Value 先验增强（Value-Prior Enhance）

**动机：** 利用 Value 向量的信息来增强驱逐分数。

**方法：**
1. 计算每个 token 的 Value 向量 L2 范数的平方：$\nu_i = \|V_i\|^2$
2. 使用均值池化平滑模长：$\gamma = \text{AvgPool}(\nu)$
3. 归一化：$\bar{\gamma}_i = \frac{\gamma_i}{\max(\gamma)}$
4. 用归一化后的 Value 先验作为先验权重，细化自适应累积注意力分数：
$$\hat{S}_i = \bar{\gamma}_i \cdot S_i$$

**算法流程：**
- **Prefill 阶段：** 计算 QKV，对最近 Br 行使用 SG-softmax 计算驱逐分数，结合 Value 先验，通过 TopK 选择 Bs 个 token
- **Generation 阶段：** 新 token 加入缓存后，继续使用 SG-softmax 更新驱逐分数，动态维护固定预算

## 实验结果

### 实验设置
- **模型：** LLaMA3-8B-Inst、Qwen2-7B-Inst、LLaMA2-7B-Chat、Gemma-7B-Inst、Qwen2-1.5B-Inst、Qwen1.5-4B-Chat、Qwen1.5-14B-Chat
- **基准：** LongBench（21 个数据集，6 个类别）、ARC-E、OpenBookQA、WiC、WinoGrande
- **缓存预算：** 720（LLaMA2-7B）/ 1000（其他），recent budget 固定为 32
- **硬件：** NVIDIA A800 80GB GPU

### 主要结果（LongBench 平均分）

| 方法 | LLaMA3-8B-Inst | Qwen2-7B-Inst | LLaMA2-7B-Chat | Gemma-7B-Inst |
|------|----------------|---------------|----------------|---------------|
| Full Cache | 41.94 | 42.47 | 27.28 | 33.25 |
| Sink | 33.55 | 32.46 | 23.27 | 27.18 |
| H2O | 38.93 | 36.24 | 24.51 | 31.09 |
| SnapKV | 40.99 | 41.30 | 26.39 | 32.39 |
| NACL | 40.77 | 39.27 | 26.04 | 31.77 |
| TOVA | 40.18 | 37.99 | 24.66 | 30.80 |
| **AhaKV** | **41.63** | **41.84** | **26.78** | **33.08** |

- AhaKV 在所有 4 个 7B 模型上平均分最高，接近 Full Cache 水平
- 在代码完成任务上，LLaMA3-8B-Inst 和 Gemma-7B-Inst 上分别超越 Full Cache 2.67% 和 2.46%
- 在段落检索任务上，LLaMA3-8B-Inst 上比次优方法高 0.5%

### 多尺度模型验证（Appendix B）

| 方法 | Qwen2-1.5B-Inst | Qwen1.5-4B-Chat | Qwen1.5-14B-Chat |
|------|-----------------|-----------------|-------------------|
| Full Cache | 32.03 | 34.02 | 44.11 |
| H2O | 25.78 | 31.61 | 42.48 |
| **AhaKV** | **31.39** | **33.09** | **43.87** |

- 在 1.5B、4B、14B 规模模型上均取得 SOTA，验证了方法的可扩展性

### 短文本结果
- 在 ARC-E、OpenBookQA、WiC、WinoGrande 上，AhaKV 在极低压缩率（仅 4% 预算）下仍表现优异
- 随着压缩率增大（预算减小），AhaKV 相比 H2O 和其他方法准确率更高

### 消融实验（Ablation Study）

| 组件 | 2Wiki | TriQA | Multi-EN | lcc | Samsum | 平均 |
|------|-------|-------|----------|-----|--------|------|
| w/o RA | 30.26 | 84.47 | 36.44 | 55.77 | 39.70 | 49.33 |
| w/o SGS | 30.50 | 82.85 | 37.89 | 58.00 | 39.60 | 49.76 |
| w/o VPE | 29.54 | 82.94 | 33.96 | 58.16 | 41.06 | 49.13 |
| **AhaKV** | **30.91** | **84.34** | **39.21** | **58.40** | **40.10** | **50.59** |

- 去除 RA（近期累积）：平均准确率下降 1.26%，确认了稳定累积项的必要性
- 去除 SGS（步增 Softmax）：平均准确率下降 0.83%
- 去除 VPE（Value 先验增强）：平均准确率下降 1.46%，证明了 Value 先验对细化驱逐分数的有效性

### 推理速度
- 结合 FlashAttention，AhaKV 可显著降低推理延迟
- 32K 输入 + 512 输出时，原始 FlashAttention 的 KV 缓存占 5.98G 内存，AhaKV（保留 2048 KV 对）仅占 0.39G

## 优势

1. **理论基础扎实：** 从数学上证明了累积注意力分数的位置偏差，并提出了系统性的偏差消除方案
2. **全局上下文保留：** 通过 Recent Accumulation 和 SG-softmax 消除位置偏差，保留序列中任意位置的关键 token
3. **充分利用 QKV 信息：** 率先利用 Value 向量信息增强驱逐分数，实现对 self-attention 机制的更全面利用
4. **计算开销低：** SG-softmax 仅需计算最近 r 行的注意力分数，额外计算量很小，与 FlashAttention 兼容
5. **泛化性强：** 在 1.5B、4B、7B、14B 多种规模模型上均表现优异
6. **内存节省显著：** 从 5.98G 降至 0.39G（32K 输入场景），部署成本大幅降低
7. **与现有方法正交：** 可与其他优化技术（量化、剪枝等）结合使用

## 局限

1. **长文本验证不足：** 受资源限制，未在超长文本上进行实验，虽然理论推导支持适应更长文本
2. **累积策略选择有限：** Prefill 阶段使用最近行累积，虽然有效但非唯一选择，例如 TopK 策略可能也是一种可行方案
3. **无代码开源：** 代码 URL 为空，无法直接复现实验结果
4. **训练无关：** 该方法仅在推理阶段应用，无法通过训练进一步优化驱逐策略
5. **参数选择：** λ 的计算依赖于缓存预算 k 的预设，不同任务可能需要调整
6. **与 Full Cache 的差距：** 虽然接近 Full Cache，但在部分任务上仍有一定差距

## 与 EfficientPaper 相关的研究方向

### KV Cache 优化
- 本文属于 **KV Cache 驱逐** 类方法，与 H2O、SnapKV、NACL、StreamingLLM 等工作直接相关
- 可与 **KV Cache 量化**（如 KVQuant、KIVI）结合，进一步压缩内存
- 可与 **稀疏注意力**（如 BigBird、Longformer、SpAtten）正交集成

### 高效推理
- 与 **剪枝**（如 SparseGPT、LLM-Pruner）、**量化**（如 GPTQ）等推理优化技术互补
- 可与 **FlashAttention** 等注意力计算加速方法结合使用

### 长上下文处理
- 与 **长上下文扩展**（如 YaRN、LongLoRA、RingAttention）相关
- 可应用于 **长文档处理**、**多轮对话** 等场景

### 注意力机制研究
- 本文对注意力分数的位置偏差分析为理解 Transformer 的注意力机制提供了新视角
- Value 向量的信息利用为未来研究提供了思路

### 检索增强生成（RAG）
- AhaKV 在段落检索任务上的优异表现，与 RAG 中的信息检索密切相关
- 可作为 RAG 系统中的高效 token 筛选策略

---

> **生成声明：** 本 note 由 AI Agent 自动生成，基于 arXiv 论文（arXiv:2506.03762v1）的全文内容，使用 fitz 提取文本并进行分析。生成时间：2025 年。所有内容以中文撰写。
