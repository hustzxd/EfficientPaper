# KVpop -- Key-Value Cache Compression with Predictive Online Pruning

> Lukas Hauzenberger, Niklas Schmidinger, Anamaria-Roberta Hartl, David Stap, Thomas Schmied, Sebastian Böck, Günter Klambauer, Sepp Hochreiter

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

Key-value (KV) cache growth is a major bottleneck in autoregressive decoding, as memory and bandwidth scale linearly with context length. Existing KV eviction methods often rely on static heuristics or proxy scores, which poorly track future token utility and cause brittle eviction as relevance shifts. To address this, we introduce KVpop, which learns a fixed-budget KV eviction policy by directly supervising the keep-or-drop decision. The scorer is trained against a novel future-attention target, computed efficiently without materializing dense attention maps. We further introduce a delayed memory-based scorer that, uniquely among learned eviction methods, defers scoring for a fixed number of steps to exploit near-future context. On AIME and HMMT mathematical reasoning, KVpop retains 98% of full-attention performance on Qwen3-4B at 75% KV cache compression and 97% at 88% compression, consistently outperforming established eviction baselines. Qwen3-8B shows even stronger results, reaching near-full teacher performance. These results show that supervising eviction with future-attention signals cuts memory costs while maintaining quality.

## 一句话总结

KVpop 通过监督学习未来注意力信号，直接指导 KV cache 的删除决策，实现了 75-88% 压缩率下 95-99% 的性能保持。

## 创新点

1. **边界监督学习**：将 KV 删除问题转化为在删除边界的监督学习，直接学习 token 的保留/删除决策，而非依赖启发式规则。

2. **未来注意力目标**：设计了一种新的未来注意力目标，通过转置注意力高效计算，不需要显式构建完整的注意力矩阵。

3. **延迟状态化评分器**：引入 mLSTM 状态化评分器，延迟评分决策直到 token 即将被删除，利用近未来上下文提升决策准确性。

4. **Fenwick 树高效 top-k**：使用 Fenwick 树实现 O(S log S) 的在线 top-k 选择，避免每步重新计算完整排序。

## 带来什么提升

1. **高压缩率性能保持**：Qwen3-4B 在 75% KV cache 压缩下保留 95% 的全注意力性能，88% 压缩下保留 94%。

2. **更大模型更强效果**：Qwen3-8B 在 75% 压缩下达到 95% 性能保留，88% 压缩下达到 99%，接近全注意力教师性能。

3. **内存近似常数使用**：在 131k token 生成长度下，内存仅增长 19%（18GB→19GB），相比全注意力的 100% 增长（18GB→36GB）。

4. **跨域泛化能力**：虽仅在数学推理数据上训练，在代码生成（LiveCodeBench）和 STEM 推理（GPQA）上也保持竞争力。

## 备注

- **延迟评分优势**：实验显示延迟评分比即时评分提升 0.2% 的 token 准确率，验证了利用近未来上下文的价值。
- **内容感知删除模式**：KVpop 倾向于保留推理结构 token（如 discourse markers、操作词、符号），而非等权重对待所有 token。
- **vs DMS 对比**：KVpop 使用固定预算的均匀删除策略，相比 DMS 的动态门控策略更易于 GPU 执行和编译。
- **局限**：设计为 Transformer 的后训练改造，而非从头训练的压缩架构；蒙 LSTM 以外的记忆模块有待探索。
