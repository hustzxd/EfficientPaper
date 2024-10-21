# SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

Attention is the cornerstone of modern Large Language Models (LLMs). Yet its
quadratic complexity limits the efficiency and scalability of LLMs, especially
for those with a long-context window. A promising approach addressing this
limitation is to leverage the sparsity in attention. However, existing
sparsity-based solutions predominantly rely on predefined patterns or
heuristics to approximate sparsity. This practice falls short to fully capture
the dynamic nature of attention sparsity in language-based tasks. This paper
argues that attention sparsity should be learned rather than predefined. To
this end, we design SeerAttention, a new Attention mechanism that augments the
conventional attention with a learnable gate that adaptively selects
significant blocks in an attention map and deems the rest blocks sparse. Such
block-level sparsity effectively balances accuracy and speedup. To enable
efficient learning of the gating network, we develop a customized
FlashAttention implementation that extracts the block-level ground truth of
attention map with minimum overhead. SeerAttention not only applies to
post-training, but also excels in long-context fine-tuning. Our results show
that at post-training stages, SeerAttention significantly outperforms
state-of-the-art static or heuristic-based sparse attention methods, while also
being more versatile and flexible to adapt to varying context lengths and
sparsity ratios. When applied to long-context fine-tuning with YaRN,
SeerAttention can achieve a remarkable 90% sparsity ratio at a 32k context
length with minimal perplexity loss, offering a 5.67x speedup over
FlashAttention-2.
