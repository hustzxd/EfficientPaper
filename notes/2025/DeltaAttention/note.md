# Delta Attention: Fast and Accurate Sparse Attention Inference by Delta Correction

<p align="center">
<img src="fig4.png" width="600" title="blank">
</p>

## Abstract

The attention mechanism of a transformer has a quadratic complexity, leading
to high inference costs and latency for long sequences. However, attention
matrices are mostly sparse, which implies that many entries may be omitted from
computation for efficient inference. Sparse attention inference methods aim to
reduce this computational burden; however, they also come with a troublesome
performance degradation. We discover that one reason for this degradation is
that the sparse calculation induces a distributional shift in the attention
outputs. The distributional shift causes decoding-time queries to fail to align
well with the appropriate keys from the prefill stage, leading to a drop in
performance. We propose a simple, novel, and effective procedure for correcting
this distributional shift, bringing the distribution of sparse attention
outputs closer to that of quadratic attention. Our method can be applied on top
of any sparse attention method, and results in an average 36%pt performance
increase, recovering 88% of quadratic attention accuracy on the 131K RULER
benchmark when applied on top of sliding window attention with sink tokens
while only adding a small overhead. Our method can maintain approximately 98.5%
sparsity over full quadratic attention, making our model 32 times faster than
Flash Attention 2 when processing 1M token prefills.

sparse prefill的output和dense prefill的output分布会发生偏移，可以sample一部分query计算dense 和 sparse 之间的偏移量，并复制扩展，对sparse prefill 的output进行修正。

该方法可以在MInference和StreamingLLM基础上有改进，且仅有少量的overhead。