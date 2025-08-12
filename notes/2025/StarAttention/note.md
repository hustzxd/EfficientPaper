# Star Attention: Efficient LLM Inference over Long Sequences

![](fig1.png)

## Abstract

Inference with Transformer-based Large Language Models (LLMs) on long
sequences is both costly and slow due to the quadratic complexity of the
self-attention mechanism. We introduce Star Attention, a two-phase block-sparse
approximation that improves computational efficiency by sharding attention
across multiple hosts while minimizing communication overhead. In the first
phase, the context is processed using blockwise-local attention across hosts,
in parallel. In the second phase, query and response tokens attend to all prior
cached tokens through sequence-global attention. Star Attention integrates
seamlessly with most Transformer-based LLMs trained with global attention,
reducing memory requirements and inference time by up to 11x while preserving
97-100% of accuracy.

RingAttention的改进，将长文档分布在多个node上，但是不进行通信，直接计算kv cache，在decode时，query需要global attention，此时通信量较少。

精度会有下降，速度明显提升。

和KVLink的思想有些类似，KVLink用于RAG领域，StarAttention用于分布式推理领域。
