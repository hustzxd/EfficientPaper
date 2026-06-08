# QuickSilver —— 通过动态 Token 停止、KV 跳过、上下文 Token 融合和自适应 Matryoshka 量化加速 LLM 推理

> Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh

![](../../blank.jpg)

---

> **⚠️ 本 Note 由 AI Agent 自动生成（Hermes Agent），基于论文全文阅读与分析。生成时间：2025-06-28。**

---

## 一句话总结

QuickSilver 是一种纯推理时、零样本、模型无关的 token 级推理加速框架，通过四种协同机制（动态 Token 停止、KV 缓存跳过、上下文 Token 融合和自适应 Matryoshka 量化），在不修改模型权重或架构的前提下，实现最高 39.6% 的 FLOP 降低，同时保持困惑度退化 ≤0.2。

---

## 摘要翻译

推理占据了大语言模型（LLM）部署中大部分的延迟和能耗，通常超过总成本的 90%。虽然训练时效率已取得显著进展，但运行时优化仍然是关键瓶颈，尤其是在自回归解码中。现有方法——如剪枝、量化、早退出和投机解码——通常需要重新训练、架构更改或会破坏解码兼容性。我们提出了 QuickSilver，一个模块化的 token 级框架，在推理时实现语义自适应性，且无需修改模型权重或结构。QuickSilver 集成了四种协同机制：(i) 动态 Token 停止，对表示已收敛的 token 终止计算；(ii) KV 缓存跳过，选择性地抑制内存写入以减少注意力开销；(iii) 上下文 Token 融合，将冗余 token 折叠到共享路径中以缩短序列长度。与投机解码或 MoE 路由不同，QuickSilver 完全在冻结的稠密模型上运行，不需要辅助网络。在 GPT-2 和 Llama-2 上，QuickSilver 在 WikiText-103 和 C4 数据集上实现了最高 39.6% 的 FLOP 降低，同时困惑度退化可忽略不计（≤0.2）。

---

## 研究动机

### 推理效率的紧迫性

LLM 在许多 NLP 任务中已超越人类水平，但推理（而非训练）已成为部署的主要瓶颈。现实世界使用模式使推理负责超过 90% 的总能耗和计算成本。在实时应用（如聊天机器人、翻译工具）中，LLM 要求亚秒级 token 延迟，即使轻微延迟也会降低用户体验。

### 现有方法的局限

| 方法类别 | 代表工作 | 主要局限 |
|---------|---------|---------|
| 早退出 | PABEE、FastBERT | 需要训练时协调、需要退出分类器 |
| MoE 路由 | Switch Transformer | 需要重新训练和架构变更 |
| 投机解码 | Speculative Decoding | 需要双模型同步、增加架构复杂度 |
| 剪枝 | LayerDrop | 需要重新训练、需要模型修改 |
| 量化 | SmoothQuant、AWQ | 需要训练时量化感知 |
| Token 合并 | Sparse Merger | 有时需要重新训练 |

### QuickSilver 的差异化定位

QuickSilver 的核心创新在于：
- **纯运行时**：不需要重新训练或修改模型架构
- **模型无关**：适用于任何冻结的 Transformer 模型
- **Token 级别**：在 token 级别进行细粒度优化
- **可组合**：四种机制可堆叠使用，效果叠加
- **与投机解码互补**：投机解码减少生成步数，QuickSilver 减少每步计算量

---

## 方法（技术细节）

### 1. 动态 Token 停止（Dynamic Token Halting, DTH）

**核心思想**：标准 Transformer 推理中，每个 token 即使表示已稳定也必须经过所有层。DTH 通过检测语义收敛，在 token 表示稳定后提前终止其计算。

**收敛信号**：定义 token t 在层 ℓ 的 L2 更新范数：
$$\Delta^{(\ell)}_t = \|h^{(\ell)}_t - h^{(\ell-1)}_t\|_2$$

**停止策略**：给定阈值 τ > 0，token t 在层 ℓ 停止当 $\Delta^{(\ell)}_t < \tau$：
- $\Delta^{(\ell)}_t \geq \tau$ → 继续（H=1）
- $\Delta^{(\ell)}_t < \tau$ → 停止（H=0）

**覆盖逻辑**：
- **强制停止**：无论 Δ 值如何都停止（用于延迟关键场景）
- **完全处理**：绕过停止（用于特殊 token 或领域敏感词）

**实验参数**：
- $\tau_{drift} = 0.045$
- $\tau_{halt} = 1.15$ bits

**计算影响**：一旦停止，token 从第 ℓ+1 层到 L 层的计算和内存流中被移除，显著降低 FLOP。

### 2. KV 缓存跳过（KV Cache Skipping）

**核心思想**：Transformer 模型在每个注意力层维护 KV 缓存，无论语义效用如何都写入每个 token 的投影。KV 跳过利用 token 停止信号，跳过已收敛 token 的冗余 KV 更新。

**KV 计算**：在层 ℓ：
$$K^{(\ell)} = [k^{(\ell)}_1, ..., k^{(\ell)}_T]^\top, \quad V^{(\ell)} = [v^{(\ell)}_1, ..., v^{(\ell)}_T]^\top$$

**跳过逻辑**：定义 KV 跳过掩码：
$$S^{(\ell)}_t = \begin{cases} 0 & \text{if } H^{(\ell)}_t = 0 \text{ (token 已停止)} \\ 1 & \text{otherwise} \end{cases}$$

当 $S^{(\ell)}_t = 0$ 时，跳过 KV 更新：
$$k^{(\ell)}_t \leftarrow S^{(\ell)}_t \cdot k^{(\ell)}_t, \quad v^{(\ell)}_t \leftarrow S^{(\ℓ)}_t \cdot v^{(\ell)}_t$$

**对注意力的影响**：
$$\text{Attention}(u, t, \ell) = \frac{(q^{(\ell)}_u)^\top (S^{(\ell)}_t \cdot k^{(\ell)}_t)}{\sqrt{d}}$$

如果 $S^{(\ell)}_t = 0$，则 $k^{(\ell)}_t = 0$，token t 实际上从注意力窗口中被移除。

**与 DTH 的协同**：使用 DTH 时，KV 跳过可贡献最高 40% FLOP 降低，同时困惑度退化可忽略。

### 3. 上下文 Token 融合（Contextual Token Fusion）

**核心思想**：通过合并语义相似的 token（表示已收敛到几乎相同的表示）来减少深度 Transformer 层中的冗余。

**融合触发**：定义 token t 和 u 在层 ℓ 的隐藏状态 L2 距离：
$$\|h^{(\ell)}_t - h^{(\ell)}_u\|_2 < \tau_{fuse}$$

其中 $\tau_{fuse}$ 是可调的相似性阈值。融合仅限于相邻 token 或通过图/注意力推导的邻近 token，以保持语义保真度。

**融合表示**：将 token {t1, ..., tk} 替换为单一的超级 token $\tilde{e}_t$，其表示为：
$$h^{(\ell)}_{\tilde{e}_t} = \frac{\sum_{i=1}^{k} \alpha_{t_i} h^{(\ell)}_{t_i}}{\sum_{i=1}^{k} \alpha_{t_i}}, \quad \alpha_{t_i} \propto \text{score}(t_i, \ell)$$

其中 α 可以反映注意力权重、token 概率或均匀平均。

**下游传播**：从层 ℓ+1 开始，只有 $\tilde{e}_t$ 贡献 key/value：
$$k^{(\ell+1)}_{\tilde{e}_t} = W^{(\ell+1)}_K h^{(\ell)}_{\tilde{e}_t}, \quad v^{(\ell+1)}_{\tilde{e}_t} = W^{(\ell+1)}_V h^{(\ell)}_{\tilde{e}_t}$$

**效率收益**：融合减少序列长度并缩小深层的计算/内存成本。与 token 停止和 KV 跳过结合时，显著降低 FLOP 且质量损失极小。

### 4. 自适应 Matryoshka 量化（Adaptive Matryoshka Quantization, AMQ）

**核心思想**：一种熵感知方法，动态调整 token 级位宽以实现高效压缩。与均匀量化不同，它更强烈地压缩可预测的 token，同时保留复杂 token 的精度。

**熵估计**：对每个 token t，计算其 softmax 归一化潜在分布的熵：
$$H(t) = -\sum_i p_i \log p_i$$

**精度分配**：AMQ 为 token t 分配位宽 $b_t$：
$$b_t = \begin{cases} 8 & \text{if } H(t) > \tau_{high} \\ 4 & \text{if } \tau_{low} \leq H(t) \leq \tau_{high} \\ 2 & \text{if } H(t) < \tau_{low} \end{cases}$$

**决策层**：选择中间网络层（如 30 层中的第 15 层）来计算 H(t) 和确定 $b_t$。早期层缺乏足够的语义上下文，而晚期层留下最小的节省空间。

**效率收益**：一旦分配位宽，后续矩阵乘法和内存存储在混合精度约束下运行，提供 FLOP 和激活内存的显著节省，同时困惑度影响可忽略。

### 5. Halting vs. Merging 决策边界

QuickSilver 在运行时通过两种互补策略减少 token：
- **Halting**：当 token 低熵且低表示漂移时（两者都满足时）
- **Merging**：当 token 语义冗余时（与相邻 token 相似度高时）

**决策优先级**：Halting 优先，因为它完全避免计算，而合并仍会产生成熟的下游计算。

---

## 实验结果

### 实验设置

- **模型**：GPT-2 (774M)、Llama-2 (7B)
- **数据集**：WikiText-103、C4
- **硬件**：NVIDIA A100 (40GB)，PyTorch 2.1 + CUDA 11.8，FP16
- **评估维度**：速度提升（FLOP 降低）和精度保持（困惑度退化）

### 主要结果

| 指标 | 数值 | 说明 |
|------|------|------|
| **最高 FLOP 降低** | 39.6% | 在 GPT-2 和 Llama-2 上 |
| **最高加速比** | 55% | 累积添加所有模块到 GPT-2 |
| **困惑度退化** | ≤0.2 | 可忽略不计 |
| **推理速度** | 0.40× 基线 | 相比量化基线 |

### GLUE/SuperGLUE 任务精度（表 2）

| 任务 | 类型 | 指标 | 基线 | QuickSilver | Δ |
|------|------|------|------|------------|---|
| MNLI (Matched) | NLI | Accuracy | 84.5 | 83.9 | -0.6 |
| QNLI | QA | Accuracy | 91.2 | 90.7 | -0.5 |
| SST-2 | 情感 | Accuracy | 94.8 | 94.6 | -0.2 |
| CoLA | 语法 | Matthews Corr. | 60.1 | 59.1 | -1.0 |
| BoolQ | Boolean QA | Accuracy | 78.4 | 77.6 | -0.8 |
| RTE | 蕴含 | Accuracy | 74.0 | 73.1 | -0.9 |

### 消融实验（各模块单独贡献）

- **动态 Token 停止 (DTH)**：贡献 18-24% 的延迟降低
- **KV 缓存跳过**：单独效果较温和，但增强 DTH 的效果
- **上下文 Token 融合**：贡献 18-24% 的延迟降低
- **自适应 Matryoshka 量化**：主要改善内存和 I/O 效率，对精度影响最小
- **累积效果**：添加所有模块后实现 55% 加速，仅增加 0.21 困惑度

---

## 优势

1. **纯推理时优化**：无需重新训练或架构变更，直接应用于冻结模型
2. **模型无关**：适用于 encoder-only (BERT)、decoder-only (GPT)、encoder-decoder (T5) 等任何 Transformer 模型
3. **可组合性**：四种机制正交，可堆叠使用，效果叠加
4. **与投机解码互补**：投机解码减少生成步数，QuickSilver 减少每步计算量，两者可组合
5. **黑盒模型适用**：适用于闭源模型或 API 服务，无需微调
6. **能源节约**：推理能耗降低 30-45%，支持低碳 AI 部署
7. **流式推理兼容**：支持自回归生成场景，适合聊天机器人、翻译系统
8. **理论保证**：提供 Lipschitz 连续性下的误差边界分析
9. **部署简单**：通过 tensor mask 实现，无需分支控制流，GPU 利用率高
10. **可与剪枝/蒸馏结合**：与结构化剪枝或知识蒸馏互补，实现复合加速

---

## 局限

1. **缺乏训练时耦合**（4/5 严重度）：不与模型参数联合训练，无法利用端到端自适应
2. **阈值敏感性**（3/5 严重度）：L2 漂移和熵阈值依赖手动定义，缺乏自动校准
3. **粒度与并行度权衡**（2/5 严重度）：Token 级自适应引入非均匀执行路径，但开销较小
4. **量化范围**（3/5 严重度）：当前量化策略较浅，静态从第 15 层开始，使用离散熵桶
5. **边缘情况语义退化**（3/5 严重度）：某些长程共指、稀有领域表达或诗歌/哲学结构可能受影响
6. **实验规模有限**：主要在 GPT-2 和 Llama-2 上验证，缺少更大规模模型的实验
7. **缺少与竞品的直接对比**：缺少与 FlashAttention、vLLM 等推理优化库的直接性能对比

---

## 与 EfficientPaper 相关的研究方向

### 与 KV 缓存相关

QuickSilver 的 KV 缓存跳过机制直接关联 **kv_cache_quant** 和 **kv_cache_sparse** 关键词：
- **KV Cache 量化**：AMQ 的熵感知位宽分配可用于 KV 缓存压缩
- **KV Cache 稀疏化**：KV 跳过通过抑制冗余 KV 写入实现稀疏化

### 与高效推理相关的方向

1. **动态计算分配**：QuickSilver 的 token 级自适应计算分配可与 LayerDrop、Early Exit 等方法结合
2. **推理时压缩**：与 SmoothQuant、AWQ 等量化方法互补
3. **投机解码集成**：QuickSilver 与投机解码正交，可实现复合加速
4. **长序列优化**：Token 融合减少有效序列长度，适合长上下文场景
5. **边缘部署**：通过降低计算量和内存需求，使大模型更适合边缘设备

### 潜在改进方向

- **学习型阈值**：使用强化学习或贝叶斯方法自适应调整阈值
- **与 FlashAttention 集成**：结合 FlashAttention 的 IO 感知优化
- **多模态扩展**：将 token 级自适应推广到视觉、语音等模态
- **推理时对抗过滤**：论文提出的未来扩展方向
- **Agentic 投机解码**：论文提出的未来扩展方向

---

## 参考文献

- arXiv: [2506.22396v1](http://arxiv.org/abs/2506.22396v1)
- 代码：[https://anonymous.4open.science/r/Quicksilver/codes/kvc.py](https://anonymous.4open.science/r/Quicksilver/codes/kvc.py)
- 关键词：kv_cache_quant, kv_cache_sparse
- 发表时间：2025年6月27日
- 机构：Manipal University Jaipur, Vellore Institute of Technology, NIT Silchar, Harrisburg University of Science and Technology, Meta AI, Amazon AI, IISER Kolkata, BITS Pilani

---

*本 Note 由 AI Agent 自动生成，基于 arXiv 论文 2506.22396v1 的全文阅读与分析。内容可能存在偏差，请以原文为准。*
