# Moment-KV: Momentum-Based Decode-Time KV Cache Compression for Long Generation

> Soumyadeep Jana, Sagar Nishad, Sanasam Ranbir Singh

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

KV cache 是长生成任务中部署大语言模型的主要瓶颈。已有 KV cache 压缩工作通常统一处理 prefill cache 与 decoding cache，或者主要关注 prefill 阶段；但压缩 prefill cache 可能破坏关键上下文和指令语义，造成性能下降。相反，decoding cache 会随输出长度持续增长，是长生成时显存增长的直接来源，却相对缺少专门研究。本文分析长生成中的 attention dynamics，发现重要 token 往往具有跨长时间的持续影响，但 attention 会出现短暂低谷；局部推理也会带来短期 burst。基于 instantaneous attention 或固定 recency window 的方法容易过早淘汰重要 token，或者保留大量已经 stale 的近邻 token。

Moment-KV 提出一种 decode-time KV cache compression 方法：冻结完整 prefill cache，只对 decoding cache 施加容量预算；每个 decoding token 维护一个带动量的 attention importance state，用指数衰减聚合历史 attention，并在超出预算时淘汰 importance 最低的 token。实验表明，在 LongGenBench 和 HelloBench 长输出任务中，Moment-KV 在 512/1024 decode-token budget 下优于现有 baselines，在保持相近 decoding throughput 的同时提升生成质量；并且可以作为 plug-in decode-time 策略与 SnapKV、PyramidKV 等 prefill-time compression 方法结合。

## 一句话总结

Moment-KV 的核心判断是：长生成中的 decoding KV 不应该靠“最近 token 必然重要”或单步 attention 来管理，而应该把 token 重要性看成随时间演化的状态；用 momentum 聚合 attention 后，系统可以保留长期 heavy hitters、淘汰真正 stale 的 decoded tokens，同时完整保留 prefill context。

## 背景与问题

长输出任务（大段代码生成、长文写作、多步推理）会让 autoregressive decoding 的 KV cache 逐步增长。以 Llama-3.1-8B-Instruct 为例，16-bit KV cache 每 token 约 128KB；生成 100K tokens 时，仅 KV cache 就接近 13GB，可能超过模型权重本身的显存占用。

论文强调应区分两个 cache 阶段：

- **Prefill cache**：由输入 prompt 一次并行计算得到，编码任务指令、约束与上下文语义；
- **Decoding cache**：生成过程中逐步追加，每步新增一个 token 的 K/V，是长输出时显存线性增长的主要来源。

已有方法的不足在于：

1. **统一压缩 prefill + decode**：可能误删 prompt 中关键指令或上下文，影响 reasoning 和 instruction following；
2. **固定 recent window**：假设最近 token 一定重要，但长生成中最近窗口内的 attention mass 往往高度集中，许多近邻 token 贡献很低；
3. **instantaneous attention**：单步 attention 会短暂波动，长期重要 token 可能在某些局部推理阶段出现 attention dip，被错误淘汰。

因此，decode-time KV compression 的关键问题不是“如何压缩所有 KV”，而是：**如何在完整保留 prefill context 的前提下，对持续增长的 generated-token KV 做时间感知的容量管理。**

## 核心方法

Moment-KV 的策略非常直接：

1. 将 KV cache 拆成两个 sub-pools：
   - $\Phi_p$：prefill pool，冻结并完整保留；
   - $\Phi_d$：decoding pool，受固定 budget $B_d$ 约束。
2. 每个 decoding token i 维护一个 importance score $I_i(t)$。
3. 每个 decoding step t 后，根据当前 query 对 decoding cache 的 attention 更新 importance：

   $$
   I_i(t) = \alpha \cdot I_i(t-1) + \bar{a}_i(t)
   $$

   其中 $\bar{a}_i(t)$ 是 across-head mean attention，$\alpha$ 是 momentum factor。
   若 $|\Phi_d| > B_d$，淘汰 importance 最低的若干 decoded tokens。

这个设计同时处理两个问题：

- 对长期 heavy hitters：即使某一步 attention 下降，历史 importance 仍可保护其不被立即淘汰；
- 对 recent stale tokens：即使 token 很近，只要累计 attention 很低，也会被淘汰，不再占用固定 recent window。

## 技术细节

### 1. Cache architecture

总 cache pool 表示为：

$$
\Phi_t = \Phi_p \cup \Phi^d_t
$$

$$
\Phi^d_t = \{(K_s, V_s) : s \in \{1, \ldots, t\}\}
$$

Moment-KV 令 prefill pool $\Phi_p$ 的大小 $M$ 始终固定，且不参与 eviction。decoding pool $\Phi_d$ 的容量受 $B_d$ 限制，因此总 cache 大小满足：

$$
|\Phi_t| \le M + B_d
$$

这种拆分非常关键：它承认 prompt/prefill token 与 generated token 的角色不同。前者承载任务语义，错误淘汰代价高；后者持续增长，需要动态管理。

### 2. Momentum-based attention aggregation

每一步 decoding 后，模型会计算当前 query 对完整 cache 的 attention。Moment-KV 只取其中 decoding sub-cache 对应的 attention，并对 heads 求平均：

$$
\bar{a}_i(t) = \frac{1}{H} \sum_h a^h_t[i],\quad i \in \Phi^d_t
$$

然后更新 importance：

$$
I_i(t) = \alpha \cdot I_i(t-1) + \bar{a}_i(t)
$$


$\alpha$ 控制记忆 horizon：

- $\alpha$ 大：历史 attention 衰减慢，更保护长期 heavy hitters；
- $\alpha$ 小：更强调近期 attention，更接近短期 relevance。

论文实验显示，较大的 $\alpha$ 通常更适合长生成，因为长生成依赖跨较长 horizon 的语义保持；同时性能对 $\alpha$ 并不过度敏感。

### 3. 新 token 处理

新生成 token 的 importance 初始化为 0，但 eviction 在 importance update 之后执行。因此新 token 至少获得一次当前 step 的 self-attention contribution：

$$
I_x(t) = \alpha \cdot 0 + \bar{a}_x(t)
$$

这避免了“新 token 还没机会证明自己就被立即淘汰”的问题；但如果新 token 的初始 attention 仍低于其他历史 token，它也可以被淘汰。

### 4. Capacity enforcement

若 decoding cache 超出 budget：

$$
overflow = \max(0, |\Phi^d_t| - B_d)
$$

Moment-KV 选择 importance 最低的 $overflow$ 个 token 淘汰：

$$
E_t = \arg\min_{S \subseteq \Phi^d_t,\ |S| = overflow} \sum_{i \in S} I_i(t)
$$

$$
\Phi^d_{t+1} = \Phi^d_t \setminus E_t
$$

这个 rule 没有 recency bias：任何 decoding token，无论新旧，只要 score 足够低都可淘汰；任何旧 token，只要长期 attention 高，都可保留。

## 实验设置

### Benchmarks

论文选择长输出任务，因为 Moment-KV 针对 decoding-time growth：

- **LongGenBench**：输出长度 4K / 8K，包含 GSM8K、CSQA、MMLU；
- **HelloBench HTG**：启发式文本生成任务，输出长度 2K / 4K / 8K / 16K；
- **∞Bench EnSum**：长文本摘要，平均输出约 1.1K tokens。

### Models

- LLaMA-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

### Baselines

- Full Cache
- StreamingLLM
- H2O
- PyramidInfer
- SCOPE
- 与 prefill compression 组合：SnapKV、PyramidKV

### Compression settings

- decoding budget：1024 tokens 与 512 tokens；
- prefill cache 完整保留，不压缩；
- 对 unified compression baselines，作者约束其总容量为 prompt length + decode budget，以保证公平比较。

## 主要结果

### 1. LongGenBench

在 LLaMA-3.1-8B-Instruct 上：

- 4K 输出、1024 decode budget：Moment-KV 平均 `60.27`，超过 Full Cache `59.63` 与 SCOPE `58.91`；
- 4K 输出、512 decode budget：Moment-KV 平均 `59.89`，明显高于 SCOPE `58.62`、StreamingLLM `57.30`、H2O `51.58`；
- 8K 输出、512 decode budget：Moment-KV 平均 `49.63`，是该设置下最高平均分。

在 Mistral-7B-Instruct-v0.3 上，Moment-KV 也在多数设置下达到最好或高度竞争的平均表现，尤其在更紧的 decoding budget 下优势更明显。

论文摘要中给出的总体提升为：LongGenBench 上平均相对提升 `2.32%`，HelloBench 上平均相对提升 `3.26%`。

### 2. HelloBench

HelloBench HTG 任务衡量模型生成原创长文本的能力。Moment-KV 在 2K、4K、8K generation length 下达到最佳 overall performance；在 16K 时与 Full Cache 接近。说明 momentum importance 对 generation-intensive tasks 有效，尤其当 token relevance 随生成过程持续变化时。

### 3. 与 prefill compression 的组合

论文将 Moment-KV 与 SnapKV、PyramidKV 结合，在 LongGenBench-4K GSM8K 上测试。结果显示，在 prefill budget 为 2048、decode budget 为 512 时，Moment-KV 相对 SCOPE 更稳健：`25.19` vs `22.56`。论文报告组合 prefill-time compression 时平均提升 `7.26%`。

这说明 Moment-KV 并不是替代 SnapKV/PyramidKV，而是补上 decode-time 管理维度。

### 4. Throughput

吞吐实验显示：

- Full Cache：`34.32 tokens/s`，最高，因为没有 cache management overhead；
- StreamingLLM：`22.37 tokens/s`；
- H2O：`20.78 tokens/s`；
- PyramidInfer：`20.81 tokens/s`；
- SCOPE：`20.71 tokens/s`；
- Moment-KV：`20.65 tokens/s`。

Moment-KV 比 SCOPE 略低，但差距很小，说明 momentum tracking 的 overhead 可控。

### 5. 分析结果

论文通过 attention distribution 和 sequential reasoning accuracy 分析表明：

- SCOPE 更集中在最新 tokens，大量 recent-window slots 实际低 attention；
- Moment-KV 的 cache utilization 更分散，更能覆盖仍有用的 tokens；
- 随着生成长度和 reasoning step 增加，SCOPE accuracy 下降更快，而 Moment-KV 更稳；
- $\alpha$ 较大时性能更好，支持“长期 heavy hitters 需要历史记忆”的假设。

## 优点与局限

### 优点

1. **问题拆分正确**：明确区分 prefill cache 与 decoding cache，避免把两类 token 混在一起压缩。
2. **方法简单可插拔**：只需要维护 attention momentum score，不需要训练模型或修改权重。
3. **适合长输出**：直接针对生成长度增加导致的 decode KV growth，而不是只看长 prompt。
4. **与 prefill compression 兼容**：可叠加 SnapKV/PyramidKV，形成 prefill + decode 双阶段压缩。
5. **overhead 小**：吞吐接近 SCOPE/H2O/PyramidInfer 等轻量方法。

### 局限

1. **importance proxy 仍是 attention**：attention 不一定等于语义重要性，尤其在某些 reasoning 或 copy 场景下可能误判。
2. **只处理 decoding tokens**：prefill cache 完整保留，适合 prompt 不极长但 output 很长的任务；若 prompt 本身也很长，仍需额外 prefill compression。
3. **没有显式 layer/head 差异建模**：论文使用 across-head mean attention，对不同 head/layer 的功能差异利用有限。
4. **系统实现细节较少**：论文报告 throughput，但未深入讨论 paged KV layout、eviction compaction、batch serving 下的碎片管理。
5. **budget policy 固定**：decode budget 512/1024 是实验设定，真实 serving 中可根据 request SLO、生成长度预测、batch pressure 自适应调整。

## 与 EfficientPaper 主题的关系

Moment-KV 属于 **KV Cache 稀疏/淘汰**，也和 KV lifecycle management 方向直接相关。

它对现有 EfficientPaper 脉络的贡献是：

- SnapKV/PyramidKV/KeyDiff 等主要关注 prefill cache 的筛选；
- StreamingLLM/H2O 等统一处理 cache，但没有充分区分 prefill 与 decode 的语义差异；
- SCOPE 开始关注 decode-time compression，但仍依赖较短期/启发式信号；
- Moment-KV 把 decode-time token importance 建模为时间状态，强调 heavy hitter 的长期持续性与 temporary attention dip。

这与 Research Brainstorm 中“KV lifecycle optimizer”方向高度一致：KV 管理不应只看静态 importance，而应引入时间动态、reuse horizon、token role（prefill vs decode）和 phase-specific action space。

## 可复现/实现要点

1. 推理 runtime 中需要将 prefill KV 与 decode KV 在逻辑上分池管理。
2. prefill pool 不参与 Moment-KV eviction；decode pool 设置固定 budget。
3. 每步 attention 计算后取 decoding sub-cache 对应 attention，最好按 head/layer 做可配置聚合。
4. 为每个 decode token 维护 $I_i$，按 $I_i \leftarrow \alpha I_i + attention_i$ 更新。
5. 超出 budget 时，淘汰 score 最低 token，同时同步删除对应 K/V 和 score。
6. 若使用 paged attention，需要处理 token eviction 后的 page compaction 或 free-list 回收。
7. Moment-KV 可与 prefill compression 组合：prefill 先由 SnapKV/PyramidKV 压缩，decode 再由 Moment-KV 管理。
8. 真实 serving 中可把 $\alpha$ 与 budget 设为 workload-aware 参数，例如长输出任务使用更大 $\alpha$，短生成任务降低 history horizon。

## 个人备注

- Moment-KV 最有价值的点不是公式复杂，而是把 decode-time KV 管理独立出来。很多 KV compression 论文默认 prompt KV 和 generated KV 可统一处理，但在 agentic/long generation 中二者生命周期完全不同。
- 这篇可以和 KVBuffer、VECTOR、TriAttention 一起看：KVBuffer 关注 linear attention state write-back，VECTOR 关注 retain/approximate/evict 的 action space，TriAttention 关注稳定 importance signal，Moment-KV 则补充了 temporal state。
- 未来很自然的方向是把 Moment-KV 的 momentum score 与质量损失估计结合：例如 high momentum + low reconstructability 保留，low momentum + high reconstructability 近似或淘汰。
- 系统侧还缺一块：decode KV 的动态淘汰会影响 paged attention layout、batch 内不同请求的 compaction 和 tree/speculative verification 的临时 KV 管理。这些问题若解决得好，Moment-KV 才能从算法 heuristic 变成 serving scheduler 的一部分。
