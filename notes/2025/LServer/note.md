# LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention

<p align="center">
<img src="fig5.png" width="600" title="blank">
</p>

## Abstract

Large language models (LLMs) have shown remarkable potential in processing
long sequences, yet efficiently serving these long-context models remains
challenging due to the quadratic computational complexity of attention in the
prefilling stage and the large memory footprint of the KV cache in the decoding
stage. To address these issues, we introduce LServe, an efficient system that
accelerates long-sequence LLM serving via hybrid sparse attention. This method
unifies different hardware-friendly, structured sparsity patterns for both
prefilling and decoding attention into a single framework, where computations
on less important tokens are skipped block-wise. LServe demonstrates the
compatibility of static and dynamic sparsity in long-context LLM attention.
This design enables multiplicative speedups by combining these optimizations.
Specifically, we convert half of the attention heads to nearly free streaming
heads in both the prefilling and decoding stages. Additionally, we find that
only a constant number of KV pages is required to preserve long-context
capabilities, irrespective of context length. We then design a hierarchical KV
page selection policy that dynamically prunes KV pages based on query-centric
similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and
decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is
released at https://github.com/mit-han-lab/omniserve.

An efficient serving system for long sequence LLMs that leverages hybrid sparse attention.