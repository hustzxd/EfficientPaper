# Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs) exhibit pronounced memory-bound characteristics
during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In
this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching
method to break through the memory bandwidth bottleneck in LLM inference
through computation-load overlap. By strategically scheduling idle memory
bandwidth during active computation windows, our method proactively prefetches
required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for
subsequent accesses and effectively hiding HBM access latency within
computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that
the proposed method achieves 2.15x improvement in attention kernel efficiency
and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art
baseline FlashAttention-3. Notably, our solution maintains orthogonality to
existing optimization techniques and can be integrated with current inference
frameworks, providing a scalable latency-hiding solution for next-generation
LLM inference engines.
