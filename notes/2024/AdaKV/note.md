# Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference

![](fig2.png)

## Abstract

Large Language Models have excelled in various fields but encounter
challenges in memory and time efficiency due to the expanding Key-Value (KV)
cache required for long-sequence inference. Recent efforts try to reduce KV
cache size to a given memory budget by evicting vast non-critical cache
elements during runtime, while preserving generation quality. Our revisiting of
current eviction methods reveals that they fundamentally minimize an upper
bound of the $L_1$ eviction loss between the pre- and post-eviction outputs of
multi-head self-attention mechanisms. Moreover, our analysis indicates that the
common practices of uniformly assigning budgets across attention heads harm
their post-eviction generation quality. In light of these findings, we propose
a simple yet effective adaptive budget allocation algorithm. This algorithm not
only optimizes the theoretical loss upper bound but also reduces the $L_1$
eviction loss in practice by aligning with the varied characteristics across
different heads. By integrating this algorithm into two state-of-the-art
methods, we demonstrate the effectiveness of using adaptive budget allocation
to optimize KV cache eviction. Extensive evaluations on 16 datasets and the
Needle-in-a-Haystack test confirm significant performance improvements across
various tasks.

每个head分配不同的 Budget，区别看待不同的 attention head.