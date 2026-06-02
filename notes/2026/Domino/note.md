# Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding

> Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, Linfeng Zhang

![cover](../../blank.jpg)

> 注：以下论文笔记由 AI Agent 自动生成，可能存在理解偏差或遗漏，请以原论文为准。

## Abstract

推测解码通过草稿模型一次提出多个候选 token，再由目标模型并行验证，从而加速 LLM 推理。但实际加速受限于草稿质量与草稿成本之间的权衡：自回归草稿模型能显式建模 draft token 之间的因果依赖，因此接受长度更高，但需要多次串行前向传播；并行草稿模型一次生成整段 draft，成本低，但削弱了块内因果依赖建模，导致接受率下降。

Domino 提出将“因果依赖建模”从“昂贵的自回归草稿执行”中解耦：先用并行草稿骨干一次性生成整个 draft block 的初步分布，再用轻量 Domino head 根据已采样 draft token 的前缀因果状态对 logits 做低秩残差修正。为稳定 teacher-forced 因果编码，论文进一步提出 base-anchored curriculum：先强化并行骨干的 base logits，再逐步把优化重心转移到经过因果修正的 final logits。实验表明，在 Qwen3 模型上，Domino 在 Transformers 后端最高实现 5.49× 端到端加速，在 SGLang serving 下最高实现 5.8× 吞吐量加速。

## 一句话总结

Domino 把投机解码中的“高质量因果草稿”拆成“并行骨干 + 轻量因果修正头”：保留 DFlash 式 block-parallel drafting 的低延迟，同时用 GRU causal encoder 和低秩 logit correction 恢复 draft token 间的前缀依赖，从而突破自回归草稿质量高但成本高、并行草稿成本低但质量不足的权衡。

## 背景与问题

标准自回归解码每生成一个 token 都要调用目标大模型一次，过程串行且常常 memory-bound，GPU 并行能力利用不足。推测解码的基本思路是用更便宜的草稿模型生成多个候选 token，再让目标模型在一次前向传播中并行验证这些候选，从而每轮推进多个 token。

论文强调，推测解码的端到端加速由两个因素共同决定：

- **接受长度 τ**：每轮验证能平均推进多少 token。
- **草稿成本 T_draft**：为了得到候选 token 需要付出多少额外计算和延迟。

自回归草稿方法（例如 EAGLE 系列）通过逐 token 生成显式建模 draft token 之间的因果依赖，通常能得到更长接受长度；但如果 draft block 长度为 γ，就需要约 γ 次 draft forward 和 γ 次 LM head 投影，成本随 γ 线性增长。并行草稿方法（例如 DFlash、DART）一次生成整个 draft block，成本低且更适合 GPU 并行，但会弱化块内因果依赖，draft 分布与目标模型自回归分布的对齐程度下降。

论文 Figure 1 给出 Qwen3-8B、16-token budget 下的直观例子：EAGLE-3 接受长度 4.86，但由于串行 draft 和树构建开销，端到端加速为 3.28×；DFlash 把 draft 延迟降到很低，但接受长度降到 4.03，加速为 3.42×。核心问题因此变成：**能否保留并行草稿的低成本，同时恢复自回归草稿的因果建模收益？**

## 核心方法

Domino 的核心设计是“并行生成 + 轻量因果修正”。整体由两部分组成：

### 1. Parallel Draft Backbone

论文使用 DFlash 架构作为并行草稿骨干。给定已验证前缀，Domino 用最后一个已验证 token 作为 anchor，构造一个 masked draft block：第一个位置是 anchor，其余位置是 `[MASK]`。骨干网络接收目标模型上下文特征和 masked block embedding，一次非自回归前向传播生成整个 block 的 hidden states。

随后使用冻结的目标模型 LM head 并行得到每个未来位置的 base logits：

```text
L_base_i = LMHead(H_i)
```

这一步保留了 DFlash 的关键优势：主干 draft computation 对整段 block 并行执行，不需要逐 token 反复调用 draft model。

### 2. Domino Head

Domino head 在 base logits 上注入前缀依赖信息，包含两个子模块：

**Causal Encoder**：使用轻量 GRU 编码当前位置之前已经采样出的 draft token embedding，得到前缀相关的因果状态 `S_{i-1}`。论文实现中 GRU hidden dimension 为 1024。这个状态让后续 draft 位置能够看到前面已采样 token 的信息，但不需要重新执行完整 draft model。

**Low-Rank Correction Head**：将并行骨干 hidden state `H_i` 与因果状态 `S_{i-1}` 拼接后，通过低秩瓶颈生成 logit-space residual correction：

```text
ΔL_i = W2 σ(W1 [H_i; S_{i-1}])
L_i = L_base_i + ΔL_i
```

论文实现中低秩维度 `r=256`。重要的是，修正发生在 **logit space**，而不是 hidden space：如果在 hidden space 修正，每个位置修正后还要重新应用完整 LM head，会把昂贵的 full-vocabulary projection 又带回串行分支；logit-space 低秩残差则把串行因果分支限制在很小的开销内。

## 技术细节

### Teacher-forced causal encoding

Domino head 的 causal encoder 需要输入当前位置之前的 draft token。训练时有两种选择：

1. 用模型自生成 token 模拟测试时 rollout（类似 EAGLE-3 的 training-time testing）。
2. 用 ground-truth token 做 teacher forcing。

论文选择 teacher forcing。理由是：在 speculative verification 中，位置 i 的修正只有在前 i-1 个 draft token 都被目标模型接受时才有意义；也就是说，有效贡献来自“前缀正确”的区域。因此用 ground-truth prefix 训练 causal encoder 更贴近接受前缀场景。实验中 teacher forcing 将平均接受长度从 3.80 提高到 3.96。

### Base-anchored curriculum

直接用 teacher forcing 训练 final logits 会产生另一个问题：修正分支看到干净前缀，可能“偷懒”绕过并行骨干，让 base logits 变弱，导致 backbone collapse。为此 Domino 同时监督 base logits 和 final logits，并使用随训练时间变化的权重：

```text
L = (1 - λ_t) L_final + λ_t L_base
```

`λ_t` 从 1 线性退火到 0：训练早期主要锚定 base logits，迫使并行骨干先学出强 base distribution；之后再逐渐让 Domino head 承担 residual correction。结合 teacher forcing 与 curriculum 后，平均接受长度进一步从 3.96 提高到 4.19。

### 运行时实现

Domino head 虽然有一个小的串行 causal update loop，但论文用 fused Triton kernels 和 CUDA Graphs 降低 kernel launch 与 Python 开销。在 Figure 1 的设置下，Domino-head latency 从 2.64ms 降到 1.20ms。相对 DFlash，Domino 只增加 56M 参数（约 +5.3%）和 2.8% 总 draft-then-verify latency，但平均接受长度提高 16.6%，端到端加速提高 12.3%。

## 实验设置

### 模型与任务

论文在 Qwen3-4B 和 Qwen3-8B 上评估。任务覆盖三类：

- **数学推理**：GSM8K、MATH-500、AIME25
- **代码生成**：HumanEval、MBPP、LiveCodeBench
- **开放对话**：MT-Bench、Alpaca

指标包括平均接受长度 τ 和相对自回归 baseline 的端到端 decoding speedup。

### 训练数据

草稿模块训练使用 `mlabonne/open-perfectblend`，包含 1.42M 指令样本，覆盖聊天、数学、代码和通用指令跟随。论文重新用对应目标模型生成 responses，而不是直接使用原数据集 responses。

### Baselines

主要比较对象包括：

- vanilla autoregressive decoding
- EAGLE-3：自回归 drafting 代表
- DFlash：block-parallel / diffusion-style drafting 代表
- DART：并行 drafting + tree pruning
- FR-Spec：通过 frequency-ranked vocabulary subset 降低 LM head projection 成本

EfficientPaper 中当前已存在的可引用 baseline 为 `2024/Eagle` 和 `2026/DFlash`。

### 实现参数

- draft block size：16
- parallel draft backbone：5 layers
- GRU causal encoder hidden dimension：1024
- low-rank correction hidden dimension：256
- 主要实验硬件：NVIDIA A100-SXM4-80GB

## 主要结果

### Transformers 后端低并发端到端加速

在 greedy decoding（Temperature=0）下，Domino 在 Qwen3-4B 和 Qwen3-8B 上均稳定超过 EAGLE-3、DART、DFlash：

- **Qwen3-4B**：平均 speedup 从 DFlash 的 4.70× 提升到 Domino 的 5.47×；平均接受长度从 6.11 提升到 7.08。
- **Qwen3-8B**：平均 speedup 从 DFlash 的 4.66× 提升到 Domino 的 5.49×；平均接受长度从 6.06 提升到 7.17。
- 在 Qwen3-8B 的 GSM8K 上，Domino 达到 7.92× speedup，相比 DFlash 的 5.21× 明显更高。

在 sampling decoding（Temperature=1）下，趋势一致：

- Qwen3-4B：平均 speedup 从 DFlash 的 4.03× 提升到 Domino 的 4.61×。
- Qwen3-8B：平均 speedup 从 DFlash 的 3.96× 提升到 Domino 的 4.46×。

### SGLang serving 高并发吞吐

论文进一步在 SGLang 下评估不同 concurrency 的 throughput。Domino 在 Qwen3-4B 与 Qwen3-8B、GSM8K 与 MBPP 上均高于 EAGLE-3 和 DFlash。摘要中报告 SGLang serving 下最高达到 **5.8× throughput speedup**。这说明 Domino 的更高接受长度并非只在离线 Transformers benchmark 中有效，也能转化为实际 serving throughput。

### 消融实验

**训练数据控制**：在相同 ShareGPT 数据、相同 16-token drafting budget 下，Domino 在 GSM8K、HumanEval、LiveCodeBench 的 TPS speedup 和平均接受长度上整体优于 EAGLE-3、FR-Spec、DFlash，说明收益主要来自架构设计，而非训练数据差异。

**训练策略**：teacher forcing 将平均接受长度从 3.80 提高到 3.96；加入 base-anchored curriculum 后进一步提高到 4.19。没有 curriculum 时，parallel backbone loss 会保持较高，表明 correction branch 可能 shortcut backbone。

**Domino head 作用**：关闭 Domino head 时平均接受长度为 3.49、平均 speedup 为 2.84×；启用 Domino head 后分别提高到 4.19 和 3.31×。这直接证明轻量 prefix-dependent correction 是相对纯并行 backbone 的主要增益来源。

## 优点与局限

### 优点

1. **抓住了投机解码的关键权衡**：不是单纯增加 draft capacity，而是把 causal modeling 与 expensive autoregressive execution 解耦。
2. **兼顾并行性和质量**：主干保持 block-parallel，一次生成整段 draft；轻量头只做低秩 residual correction。
3. **工程成本低**：相对 DFlash 仅增加 56M 参数、约 5.3%，总 latency 增加 2.8%。
4. **训练策略针对性强**：teacher forcing 与 speculative verification 的 accepted-prefix 机制一致；curriculum 避免 correction branch 过度依赖干净前缀而压垮 base backbone。
5. **已有 serving 框架验证**：在 SGLang 中报告最高 5.8× throughput speedup，说明不只是离线算法指标。

### 局限

1. **框架适配范围有限**：论文明确当前实现主要适配 SGLang，其它 serving 框架兼容性尚未系统评估。
2. **硬件相关性强**：实际 speedup 受 memory bandwidth、compute capability、kernel efficiency 影响，不同平台需要重新优化。
3. **仍有轻量串行分支**：Domino head 的 causal update loop 虽然很小，但不是完全并行；在更长 block size 或更低延迟硬件上可能成为新瓶颈。
4. **验证集中在 Qwen3**：论文主要在 Qwen3-4B/8B 上验证，尚不清楚对更大模型、MoE 模型、长上下文场景的泛化情况。
5. **方法依赖训练 draft 模块**：相比一些 training-free speculative approaches，Domino 需要额外训练并部署 draft backbone 与 Domino head。

## 与 EfficientPaper 主题的关系

Domino 属于 **speculative_decoding**，并且与 EfficientPaper 最近的 DFlash 方向直接相连。DFlash 证明 block-parallel / diffusion-style drafter 可以显著降低 T_draft，但因果依赖弱化会限制接受长度；Domino 则进一步说明，下一阶段的投机解码研究重点不是在“自回归 vs 并行”二选一，而是把两者拆解：

- 主干计算保持并行，最大化 GPU 利用率；
- 因果信息用轻量、低秩、logit-space correction 注入；
- 训练目标必须保护 base backbone，避免 correction branch 破坏并行草稿质量。

这使方向 8 从“扩散驱动投机解码”扩展为更一般的 **parallel drafting + lightweight causal correction** 范式。

## 可复现/实现要点

1. 使用 DFlash 式 5-layer parallel backbone 生成整段 draft hidden states。
2. 对 base logits 只做一次并行 LM head projection，避免在串行分支重复 full-vocabulary projection。
3. 用 GRU causal encoder 编码已采样 draft token prefix，hidden dimension 可参考 1024。
4. 用 rank-256 low-rank correction head 生成 logit residual，而非 hidden residual。
5. 训练时采用 teacher forcing，但必须配合 base-anchored curriculum，先监督 base logits，再逐步转向 final logits。
6. serving 实现中需要 fused Triton kernels + CUDA Graphs，否则小型串行 head 的 kernel launch overhead 可能吃掉收益。
7. 与 SGLang 集成时重点关注 draft rollout loop、tree construction/sampling、verification batch 的流水化。

## 个人备注

- Domino 的关键价值不只是“比 DFlash 更快”，而是给出一个更可扩展的设计原则：把昂贵的自回归路径拆成可并行的 base prediction 与廉价的 causal residual。
- 这个思想可能推广到更长 draft block：随着 block size 增大，纯并行 backbone 的因果缺失更严重，Domino head 的边际价值可能上升；但串行 GRU loop 也可能成为瓶颈，需要研究 scan/parallel prefix 或 chunked causal correction。
- 可以进一步考虑把 Domino 与 KVBuffer 的 speculative verification 结合：draft token 的临时 state/KV 如何缓存、何时 flush，与 Domino 的 parallel draft block 可能存在系统级联合优化空间。
- 与 FR-Spec/SpecVocab 的词表裁剪思路互补：Domino 降低因果建模成本，FR-Spec 降低 LM head projection 成本，两者可能共同压缩 draft overhead。
