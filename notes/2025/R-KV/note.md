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

Snap-KV等方法只考虑长prompt情况下的kv 压缩，没有考虑长generation下的kv 压缩，现在Reasoning模型解决数学问题时，一般会生成很长的推理链。

- Decoding-time Compression
  - 不止关注prefill时的kv 压缩，在decoding时，每次decode特定长度的token后，对kv 进行压缩
- Importance Scoring via Attention Weights
  - 和之前方法相似，根据最近的$\alpha$个token得到的Attention Weight来判断哪些kv cache更重要
- Redundancy Estimation via Semantic Similarity
  - 对K cache去冗余，先对K 取均值，然后求所有K 的余弦相似性，数值较大表示越冗余。
- Joint Selection Strategy for KV Cache Retention
  - 以上两个方法得到的结果综合，对KV 进行压缩