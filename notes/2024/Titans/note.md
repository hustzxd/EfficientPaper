# Titans: Learning to Memorize at Test Time

> Ali Behrouz, Peilin Zhong, Vahab Mirrokni

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

论文提出一种能在测试时持续学习和记忆的神经长期记忆模块，并据此构造 Titans 架构。其核心思想是把 attention 视为精确但有限的短期记忆，把可在线更新的神经网络参数视为可压缩历史信息的长期记忆，再用 persistent memory 保存与输入无关的任务知识。长期记忆通过 surprise、momentum 和 adaptive forgetting 学习何时写入、保留或遗忘信息；在语言建模、常识推理、DNA 建模、时间序列和超长上下文任务上，Titans 整体优于多种 Transformer 与现代线性循环模型，并可扩展到超过 2M 的上下文长度。

## 一句话总结

Titans 将测试时学习变成可写入神经网络参数的长期记忆，用短期 attention + 长期 neural memory + persistent memory 组合解决长上下文中的信息压缩与遗忘问题。

## 创新点

1. **测试时可学习的神经长期记忆**：把记忆模块当作 meta in-context learner，在 inner loop 中根据当前数据更新自身权重，在 outer loop 中训练整个架构；因此模型可以在推理过程中把历史上下文的抽象写入参数，而不是只保存在固定 hidden state 或 KV cache 中。
2. **Surprise-driven memory update**：用关联记忆损失对输入的梯度表示 momentary surprise，并引入 momentum 形式的 past surprise，使高信息量事件及其后续相关片段能够共同进入记忆，而不是只记住单个突变点。
3. **自适应遗忘与深层记忆**：通过 data-dependent weight decay/gate 控制记忆清除比例；记忆模块使用多层 MLP，突破线性或矩阵状态只能表达线性 key-value 映射的限制，并可用 chunking、矩阵乘法和 associative scan 并行训练。
4. **三种 Titans 组合方式**：MAC（Memory as a Context）把长期记忆检索结果交给 attention，MAG（Memory as a Gate）用滑动窗口 attention 与长期记忆门控融合，MAL（Memory as a Layer）把记忆作为 attention 前的序列层；三者都可加入 persistent memory 作为任务级、输入无关的知识。

## 带来什么提升

1. 在 340M、15B tokens 的语言建模和常识推理实验中，Titans MAG 的综合平均分达到 **47.54**，高于 Gated DeltaNet 的 45.42、TTT 的 44.51 和 Transformer++ 的 42.92；在 400M 设置中，Titans MAC 达到 48.65。
2. 在 16K 的 S-NIAH 长上下文检索中，Titans MAC 对 pass-key、number 和 UUID 三类任务分别达到 **98.4、97.4、95.2**，显著优于 TTT 的 88.4、4.4、0.0，以及 Mamba2 的 5.4、0.0、0.0。
3. 在 BABILong 跨长文档推理中，Titans MAC 超过 Mamba2、RWKV、RecurrentGemma、Gemma、Llama、GPT-4 和 GPT-4o-mini 等基线；论文报告其参数量显著小于部分大模型/RAG 基线，同时取得更高结果。
4. Neural Memory 在 ETT、ECL、Traffic、Weather 等长期时间序列预测上优于 Mamba-based、Transformer-based 和线性模型；在 GenomicsBenchmarks 上也达到与 SOTA 竞争的 DNA 分类准确率。
5. 记忆深度提升会改善长序列 perplexity 和鲁棒性，但训练吞吐随深度近似线性下降；消融实验显示 weight decay、momentum、convolution 和 persistent memory 都有正贡献，MAC 在长上下文任务上通常优于 MAG/MAL。

## 备注

- Titans 的“测试时记忆”本质上是对模型参数进行在线更新，推理系统必须额外管理状态隔离、并发请求之间的记忆污染、checkpoint/恢复和更新开销；它不能直接等同于传统 KV cache。
- 深层 neural memory 比 Mamba2、Gated DeltaNet 略慢，论文将原因归于更复杂的记忆更新和后者高度优化的 kernel；因此效果、记忆表达力和硬件效率之间存在明确 trade-off。
- 与 Nested Learning 的关系：Titans 是 NL 所描述的多层 context flow / 多时间尺度优化的一种具体实例，可作为“可学习记忆系统”与 linear attention state、KV lifecycle 管理结合时的重要基础工作。
