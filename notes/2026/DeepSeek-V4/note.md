# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

![111](cover.jpg)

## External Guides

- [NPU DeepSeek-V4 推理优化实践](deepseek_v4_inference_guide.md)：来自本地 guide 的 Ascend/NPU 推理部署、融合 kernel、量化、多流并行与 benchmark 说明。

## Abstract

DeepSeek-V4 是 DeepSeek-AI 发布的百万 token 上下文 MoE 语言模型系列预览版，包括 DeepSeek-V4-Pro（1.6T 总参数，49B 激活）和 DeepSeek-V4-Flash（284B 总参数，13B 激活）。论文的核心目标是突破超长上下文下 vanilla attention 的计算和 KV cache 瓶颈，使百万 token 级上下文能够在训练、推理和 test-time scaling 中更常规地使用。

方法上，DeepSeek-V4 引入混合注意力架构，将 Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA）结合起来：CSA 在序列维压缩 KV cache 后执行稀疏注意力，HCA 采用更激进的 KV 压缩但保持 dense attention。同时，模型使用 Manifold-Constrained Hyper-Connections（mHC）增强残差连接，并采用 Muon optimizer 改善收敛和训练稳定性。系统层面，论文还强调 MoE 通信-计算重叠、TileLang kernel 开发、确定性 kernel、长上下文并行、异构 KV cache 和 on-disk KV cache 复用等基础设施优化。

实验上，论文报告 DeepSeek-V4-Pro 在 1M-token 上下文时，相比 DeepSeek-V3.2 只需要约 27% 的单 token 推理 FLOPs 和约 10% 的 KV cache；DeepSeek-V4-Flash 进一步降低成本。在能力侧，DeepSeek-V4-Pro-Max 在知识、推理、长上下文和 agentic coding 等评测中显著超过先前 DeepSeek open models，并在若干任务上接近或对齐 frontier proprietary models。

## Core Ideas

- **Hybrid long-context attention**：CSA 负责在压缩后做稀疏选择，HCA 负责以更高压缩率保留 dense attention 路径，两者共同降低长上下文注意力成本。
- **KV cache 压缩优先**：不是只优化 attention FLOPs，而是同时把长上下文部署中的 KV cache 容量、复用和存储作为核心问题处理。
- **mHC 替代传统残差增强路径**：通过 manifold-constrained residual mapping 增强表示混合能力，补偿长上下文压缩/稀疏化可能带来的建模损失。
- **Muon optimizer 用于超大 MoE 训练**：论文将 Muon 引入 DeepSeek-V4 训练，强调更快收敛和更好的稳定性。
- **系统与模型协同设计**：长上下文能力依赖架构、优化器、kernel、并行策略、KV cache 管理和 post-training 的组合，而不是单一 attention trick。

## Architecture Notes

### CSA: Compressed Sparse Attention

CSA 的思路是先沿序列维压缩 KV cache，再在压缩后的表示上执行 DeepSeek Sparse Attention 类似的稀疏注意力。它适合处理百万 token 上下文中“只需要访问少量相关片段”的场景，主要收益来自减少 attention 计算和 KV cache 访问。

### HCA: Heavily Compressed Attention

HCA 使用更强的 KV 压缩，但保留 dense attention。它可以看作对 CSA 的补充：CSA 偏向选择关键 token/block，HCA 偏向提供全局但低成本的上下文覆盖，从而缓解纯稀疏注意力可能遗漏信息的问题。

### mHC: Manifold-Constrained Hyper-Connections

mHC 用于增强 conventional residual connections。对 Efficient LLM 研究而言，值得关注的是：当 attention 被压缩、稀疏化后，残差/连接结构是否能弥补信息传递能力下降，并提升深层 MoE 模型稳定性。

### Inference KV Cache

论文设计了异构 KV cache 结构和 on-disk KV cache storage，用于共享前缀复用和长上下文场景下的缓存管理。这说明百万 token context 的实用性不只取决于 attention 算子，还取决于 cache layout、offload/reuse 策略和 I/O 管理。

## Key Results

- **模型规模**：DeepSeek-V4-Pro 为 1.6T 总参数、49B 激活；DeepSeek-V4-Flash 为 284B 总参数、13B 激活。
- **长上下文效率**：在 1M-token context 下，DeepSeek-V4-Pro 相比 DeepSeek-V3.2 约为 27% 单 token FLOPs、10% KV cache。
- **Base model 对比**：DeepSeek-V4-Flash-Base 在更小参数预算下超过 DeepSeek-V3.2-Base 多数评测；DeepSeek-V4-Pro-Base 在知识、推理、代码和长上下文上进一步提升。
- **Long-context evaluation**：DeepSeek-V4-Pro 在 MRCR 和 CorpusQA 等任务上展示 1M-token 上下文能力；MRCR 中 128K 以内表现较稳定，超过 128K 后有下降但仍保持较强检索能力。
- **Agentic coding**：论文内部 R&D Coding Benchmark 中，DeepSeek-V4-Pro-Max pass rate 为 67%，接近 Claude Opus 4.5，并低于 Opus 4.6 Thinking。

## Limitations

- **架构复杂度高**：论文承认 DeepSeek-V4 为追求极致长上下文效率，保留了许多已验证组件和技巧，整体架构相对复杂。
- **训练稳定性机制仍不够原则化**：Anticipatory Routing 和 SwiGLU Clamping 等稳定性技巧有效，但机制解释仍不足。
- **超长上下文仍有退化**：MRCR 结果显示超过 128K 后检索性能开始下降，说明百万上下文可用但并非无损。
- **Agent 能力仍落后部分闭源前沿模型**：公开 agent/coding 评测上仍与 frontier closed models 有差距。

## Research Thoughts

- **CSA/HCA ablation**：可以研究 CSA 与 HCA 在不同任务中的互补性，例如检索型、推理型、多文档聚合型任务分别依赖哪一路注意力。
- **压缩 KV 的误差建模**：将 KV compression 视为有损记忆系统，研究压缩误差如何影响 retrieval、reasoning chain 和 agent trajectory。
- **长上下文 + test-time scaling**：百万 context 可承载更长思维链、工具调用历史和多轮状态，值得研究如何分配 context budget 与 reasoning budget。
- **on-disk KV cache 调度**：结合 KV cache 重要性预测、prefix reuse、SSD/CPU/GPU 分层存储，做面向真实 agent workflow 的 cache scheduler。
- **mHC 与稀疏注意力耦合**：探索 residual connection 设计是否能系统性提升 sparse/compressed attention 的可训练性和长程信息保真度。

## Tags

- Long Context
- Sparse Attention
- KV Cache Management
- MoE
- Efficient Training
- LLM Deployment
