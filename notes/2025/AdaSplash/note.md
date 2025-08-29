# AdaSplash: Adaptive Sparse Flash Attention

![](../../blank.jpg)

## Abstract

The computational cost of softmax-based attention in transformers limits
their applicability to long-context tasks. Adaptive sparsity, of which
$\alpha$-entmax attention is an example, offers a flexible data-dependent
alternative, but existing implementations are inefficient and do not leverage
the sparsity to obtain runtime and memory gains. In this work, we propose
AdaSplash, which combines the efficiency of GPU-optimized algorithms with the
sparsity benefits of $\alpha$-entmax. We first introduce a hybrid
Halley-bisection algorithm, resulting in a 7-fold reduction in the number of
iterations needed to compute the $\alpha$-entmax transformation. Then, we
implement custom Triton kernels to efficiently handle adaptive sparsity.
Experiments with RoBERTa and ModernBERT for text classification and
single-vector retrieval, along with GPT-2 for language modeling, show that our
method achieves substantial improvements in runtime and memory efficiency
compared to existing $\alpha$-entmax implementations. It approaches -- and in
some cases surpasses -- the efficiency of highly optimized softmax
implementations like FlashAttention-2, enabling long-context training while
maintaining strong task performance.
