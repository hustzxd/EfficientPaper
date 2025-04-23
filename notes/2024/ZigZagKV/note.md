# ZigZagkv: Dynamic KV Cache Compression for Long-context Modeling based on Layer Uncertainty

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Large Language models (LLMs) have become a research hotspot. To accelerate
the inference of LLMs, storing computed caches in memory has become the
standard technique. However, as the inference length increases, growing KV
caches might lead to out-of-memory issues. Many existing methods address this
issue through KV cache compression, primarily by preserving key tokens
throughout all layers to reduce information loss. Most of them allocate a
uniform budget size for each layer to retain. However, we observe that the
minimum budget sizes needed to retain essential information vary across layers
and models based on the perspectives of attention and hidden state output.
Building on this observation, this paper proposes a simple yet effective KV
cache compression method that leverages layer uncertainty to allocate budget
size for each layer. Experimental results show that the proposed method can
reduce memory usage of the KV caches to only $\sim$20\% when compared to Full
KV inference while achieving nearly lossless performance.
