# Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache

<p align="center">
<img src="fig1.png" width="600" title="blank">
</p>

## Abstract

Large Language Models (LLMs) demonstrate substantial potential across a
diverse array of domains via request serving. However, as trends continue to
push for expanding context sizes, the autoregressive nature of LLMs results in
highly dynamic behavior of the attention layers, showcasing significant
differences in computational characteristics and memory requirements from the
non-attention layers. This presents substantial challenges for resource
management and performance optimization in service systems. Existing static
model parallelism and resource allocation strategies fall short when dealing
with this dynamicity. To address the issue, we propose Infinite-LLM, a novel
LLM serving system designed to effectively handle dynamic context lengths.
Infinite-LLM disaggregates attention layers from an LLM's inference process,
facilitating flexible and independent resource scheduling that optimizes
computational performance and enhances memory utilization jointly. By
leveraging a pooled GPU memory strategy across a cluster, Infinite-LLM not only
significantly boosts system throughput but also supports extensive context
lengths. Evaluated on a dataset with context lengths ranging from a few to
2000K tokens across a cluster with 32 A100 GPUs, Infinite-LLM demonstrates
throughput improvement of 1.35-3.4x compared to state-of-the-art methods,
enabling efficient and elastic LLM deployment.
