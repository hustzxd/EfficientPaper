# R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

> Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, Junjie Hu

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Reasoning models have demonstrated impressive performance in self-reflection
and chain-of-thought reasoning. However, they often produce excessively long
outputs, leading to prohibitively large key-value (KV) caches during inference.
While chain-of-thought inference significantly improves performance on complex
reasoning tasks, it can also lead to reasoning failures when deployed with
existing KV cache compression approaches. To address this, we propose
Redundancy-aware KV Cache Compression for Reasoning models (R-KV), a novel
method specifically targeting redundant tokens in reasoning models. Our method
preserves nearly 100% of the full KV cache performance using only 10% of the KV
cache, substantially outperforming existing KV cache baselines, which reach
only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV
cache performance with 16% of the KV cache. This KV-cache reduction also leads
to a 90% memory saving and a 6.6X throughput over standard chain-of-thought
reasoning inference. Experimental results show that R-KV consistently
outperforms existing KV cache compression baselines across two mathematical
reasoning datasets.
