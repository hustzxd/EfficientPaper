# Sawtooth Wavefront Reordering: Enhanced CuTile FlashAttention on NVIDIA GB10

> Yifan Zhu, Yekai Pan, Chen Ding

![111](../../blank.jpg)

## Abstract

High-performance attention kernels are essential for Large Language Models. This paper presents analysis of CuTile-based Flash Attention memory behavior and a technique to improve its cache performance. In particular, our analysis on the NVIDIA GB10 (Grace Blackwell) identifies the main cause of L2 cache miss. Leveraging this insight, we introduce a new programming technique called Sawtooth Wavefront Reordering that reduces L2 misses. We validate it in both CUDA and CuTile, observing 50\% or greater reduction in L2 misses and up to 60\% increase in throughput on GB10.
