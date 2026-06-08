# LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation

> Yiqun Shen, Song Yuan, Zhengze Zhang, Xiaoliang Wang, Daxin Jiang, Nguyen Cam-Tu

![111](../../blank.jpg)

> ⚠️ 本 note 由 AI Agent 自动生成（Hermes Agent），基于论文全文阅读与分析。生成时间：2026-06-04。

---

## 一句话总结

LAVa 是一种基于 Transformer 残差流信息损失最小化框架的统一 KV Cache 压缩方法，首次实现了无需训练的逐层动态 head 与 layer 预算分配，在长上下文推理中显著降低内存占用并保持最优性能。

---

## 摘要翻译

KV Cache 被广泛用于加速 LLM 的长上下文推理，但其高内存需求促使了缓存压缩技术的发展。然而，现有压缩方法大多依赖启发式策略，缺乏动态预算分配机制。为解决这一问题，本文提出了一个统一的缓存压缩框架，通过最小化 Transformer 残差流中的信息损失来设计算法。基于此框架，我们分析了层级注意力输出损失（Layer Attention Output Loss），推导出一个新的度量指标，用于跨 head 比较缓存条目，从而实现带有动态 head 预算的逐层压缩。此外，通过对比跨层信息，我们还实现了动态 layer 预算分配。LAVa 是首个统一的缓存淘汰和动态预算分配策略，与先前方法不同，它不依赖训练或多种策略的组合。在多个基准测试（LongBench、Needle-In-A-Haystack、Ruler、InfiniteBench）上的实验表明了其优越性。实验还揭示了一个新发现：动态 layer 预算对生成任务（如代码补全）至关重要，而动态 head 预算在提取任务（如抽取式 QA）中起关键作用。作为完全动态的压缩方法，LAVa 在不同任务类型上持续保持顶级性能。

---

## 研究动机

1. **长上下文推理的内存瓶颈**：LLM 支持 128K+ token 的上下文长度（如 Claude 3.5、GPT-4、Qwen2.5），KV Cache 的内存消耗成为推理的主要瓶颈。

2. **现有方法的局限性**：现有 KV Cache 压缩方法（如 H2O、SnapKV、PyramidKV）大多依赖启发式策略，缺乏理论基础；虽然 AdaKV 和 CAKE 探索了动态 head/layer 预算，但没有方法同时实现 head 和 layer 级别的动态分配。

3. **缺少统一框架**：KV Cache 淘汰与预算分配通常被视为独立问题，缺乏统一的理论框架来指导算法设计。

4. **实践需求**：需要一种无需训练、无需超参数调优、简单易部署的 KV Cache 压缩方法。

---

## 方法（技术细节）

### 3.1 统一框架：基于信息损失的 KV Cache 淘汰

LAVa 从 Transformer 残差流的信息流视角出发，将 KV Cache 压缩建模为最小化信息损失的优化问题。

**核心思想**：将 LLM 解码过程视为在当前残差流上的操作，每个残差流对应一个 token，注意力头从过去的残差流中复制信息到当前流。

**优化目标**：最小化最后一个层的 logits 信息损失，同时满足预算约束：
$$\min_{I,B} P(x_1^N, I, B)$$
约束条件：
- $\sum_{i \in [N]} I_{l,h}[i] = B_{l,h}$（每个 head 有固定预算）
- $\sum_{h \in [H]} B_{l,h} = B_l$（每个 layer 的总预算）
- $\sum_{l \in [L]} B_l = B$（总预算固定）
- 最近 $w$ 个 token 保留

**简化方案**：由于搜索空间是组合的，通过引入评分函数 $s_{l,h}[i]$ 贪心选择最不重要的条目进行淘汰。

### 3.2 Layer Attention Output Loss（层级注意力输出损失）

**定理 1**：层级注意力输出损失的 L1 范数上界为：
$$\|y_l^N - \hat{y}_l^N\|_1 \leq 2\hat{C} \sum_{h \in [H]} \sum_{i \in [N]} A_{l,h}^N[i] \bar{V}_{l,h} (1 - I_{l,h}[i])$$

其中 $\bar{V}_{l,h} = \max_{k \in [N]} \|V_{l,h}[k]\|_1$ 是 head 级别的 value 范数最大值。

**关键创新**：该上界引入了 head 级别的 value 范数 $\bar{V}_{l,h}$，这使得跨 head 比较成为可能，不同于 AdaKV 只使用 attention score。

### 3.3 LAVa 评分函数

基于上述上界，LAVa 定义了新的评分函数（LAVa Score）：

$$s_{l,h}[i] = \frac{\max_{k \in [N]} \|V_{l,h}[k]\|_1}{w} \sum_{j=N-w}^{N} A_{l,h}^j[i]$$

该评分函数结合了：
1. **最近 w 个残差流的 attention score**（类似 SnapKV）
2. **head 级别的 value 范数最大值**（作为缩放因子）

### 3.4 动态 Head 预算分配

在层 $l$ 内部，将所有 head 的 LAVa 评分展平为一维数组 $s_l$，然后对所有 head 的缓存条目进行跨 head 排名，自动实现动态 head 预算分配。

### 3.5 动态 Layer 预算分配

**核心思想**：不确定性更大的层应获得更大预算。

1. **不确定性度量**：通过归一化熵计算每层的不确定性：
$$e_l = -\frac{\sum_{h,i} \hat{s}_{l,h}[i] \log \hat{s}_{l,h}[i]}{H \times N}$$

2. **逐层压缩**：prefill layer $l$ 后，重新压缩低于 $l$ 的层（借鉴 CAKE），预算随层数增加而减小。

### 3.6 GQA 支持

对于 Group Query Attention（GQA）模型，LAVa 采用保守策略：组内所有 head 的 score 取最大值，只要该 token 对组内任一 head 重要就保留。

### 3.7 与现有方法的对比

| 方法 | Head 预算 | Layer 预算 | 评分函数 | 损失函数 |
|------|-----------|------------|----------|----------|
| SnapKV | 固定 | 固定 | 最近注意力得分 | Head Attention |
| CAKE | 固定 | 动态 | 注意力得分 + 注意力偏移 | Head Attention |
| AdaKV | 动态 | 固定 | 最近注意力得分 | Layer Attention Output |
| PyramidKV | 固定 | 固定（金字塔） | 近似 SnapKV | Layer Attention Output |
| **LAVa** | **动态** | **动态** | **最近注意力得分 × value 范数** | **Layer Attention Output** |

---

## 实验结果

### 5.1 实验设置

- **模型**：Mistral-7B-Instruct-v0.2（32K）、Qwen2.5-7/14/32B-Instruct（32K）、Llama3-8B-Instruct（8K）
- **基准测试**：LongBench（21 个数据集）、Needle-In-A-Haystack、Ruler、InfiniteBench
- **基线**：PyramidKV、SnapKV、Ada-SnapKV、Ada-PyramidKV、CAKE

### 5.2 主要结果（Mistral-7B，LongBench）

**关键发现**：

1. **LAVa 在所有预算下全面超越所有基线**，在小预算下优势更明显：
   - B=128HL：LAVa 36.74 vs Ada-SnapKV 35.82 vs CAKE 35.06
   - B=256HL：LAVa 40.12 vs Ada-SnapKV 39.40 vs CAKE 38.84
   - B=512HL：LAVa 42.59 vs Ada-SnapKV 42.11 vs CAKE 41.76
   - B=1024HL：LAVa 43.65 vs Ada-SnapKV 43.34 vs CAKE 43.36

2. **代码任务表现突出**：在 RepoBench-P（B=128HL）上，LAVa（48.92）和 CAKE（48.53）显著优于 Ada-SnapKV（46.85）。

3. **任务类型发现**：
   - **提取任务**（如 QA）：压缩后性能损失较小，动态 head 预算更关键
   - **生成任务**（如摘要、代码补全）：压缩后性能损失较大，动态 layer 预算更关键
   - **LAVa 在两类任务上均保持顶级性能**

### 5.3 效率评估

- **解码延迟**：相比 Full Cache，LAVa 在 128K 上下文长度下实现 **9× 加速**
- **内存峰值**：LAVa 有效控制内存峰值，相比 Full Cache 避免 OOM；相比 CAKE 内存额外开销极小
- **额外开销**：相比 SnapKV，LAVa 额外计算开销仅 0.01%，额外内存开销仅 0.6%

### 5.4 消融实验

- **动态 head 预算**：去除后性能显著下降，尤其在小预算下
- **动态 layer 预算**：去除后生成任务性能下降明显
- **LAVa Score vs AdaKV Score**：LAVa 评分在多数任务上胜出

### 5.5 多模型验证

- Qwen2.5-7B/14B/32B-Instruct 和 Llama3-8B-Instruct 上的实验结论一致
- 在 Needle-In-A-Haystack、Ruler、InfiniteBench 上也表现出优越性

---

## 优势

1. **统一框架**：首次将 KV Cache 淘汰和动态预算分配统一在同一理论框架下
2. **理论基础**：基于 Transformer 残差流信息损失最小化，提供严格的理论推导
3. **无需训练**：完全免训练，无需任何额外参数调优
4. **双层动态**：同时实现 head 和 layer 级别的动态预算分配
5. **性能优越**：在 LongBench、Needle-In-A-Haystack、Ruler、InfiniteBench 上全面超越基线
6. **效率显著**：128K 上下文下实现 9× 解码加速，额外计算和内存开销极小
7. **简单易部署**：相比 CAKE 无需调参，相比 PyramidKV 无需超参数搜索
8. **GQA 兼容**：支持 Group Query Attention 等现代 LLM 架构

---

## 局限

1. **理论分析与实验的覆盖面有限**：统一框架提供了多种优化方向，但论文只探索了其中一种
2. **与 Full Cache 的性能差距**：在某些任务上，LAVa 与 Full Cache 仍存在性能差距，尤其是生成任务
3. **动态 layer 预算的理论解释不足**：论文指出需要进一步研究为何动态 layer 预算对生成任务至关重要
4. **推理框架集成有限**：目前仅集成了 FlashAttention-2，尚未集成 vLLM 等广泛使用的推理框架
5. **额外的内存开销**：需要存储 head 级别的 value 范数，但开销极小
6. **跨层信息利用有限**：虽然实现了动态 layer 预算，但未充分利用跨层信息交互

---

## 与 EfficientPaper 相关的研究方向

### 1. KV Cache 稀疏化（kv_cache_sparse）
- LAVa 是 KV Cache 稀疏化领域的前沿方法，属于 **无训练** 的动态压缩策略
- 相关 baseline：SnapKV（2024）

### 2. 长上下文高效推理
- LAVa 通过减少 KV Cache 内存占用，实现 9× 解码加速
- 可与其他高效推理技术（如 FlashAttention、量化）结合

### 3. 注意力机制压缩
- LAVa 的动态 head 预算分配与 DuoAttention、Retrieval Head 等方法思路相通
- 研究如何更智能地分配注意力资源

### 4. 模型剪枝与压缩
- LAVa 框架可扩展到模型剪枝（通过最小化信息流损失来选择性剪枝参数）
- 与 LoRA、量化等技术的结合

### 5. 缓存优化与卸载
- LAVa 框架可扩展到缓存卸载问题（决定哪些缓存部分卸载到 CPU）
- 与 KV Cache merge、retrieval 等技术结合

### 6. 在线强化学习优化
- 未来方向：将 KV Cache 淘汰建模为在线强化学习任务
- 优化策略以最大化预期奖励（最小化未来残差流的预期损失）

---

## 参考信息

- **论文来源**：arXiv:2509.09754v1 (2025)
- **作者**：Yiqun Shen, Song Yuan, Zhengze Zhang, Xiaoliang Wang, Daxin Jiang, Nguyen Cam-Tu
- **机构**：南京大学, Stepfun
- **代码**：https://github.com/MGDDestiny/Lava
- **关键词**：kv_cache_sparse
- **Baseline**：2024/SnapKV
