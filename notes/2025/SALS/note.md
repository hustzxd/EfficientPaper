# SALS: Sparse Attention in Latent Space for KV cache Compression

> Junlin Mu, Hantao Huang, Jihang Zhang, Minghui Yu, Tao Wang, Yidong Li

![111](fig3.png)

## Abstract

Large Language Models capable of handling extended contexts are in high demand, yet their inference remains challenging due to substantial Key-Value cache size and high memory bandwidth requirements. Previous research has demonstrated that KV cache exhibits low-rank characteristics within the hidden dimension, suggesting the potential for effective compression. However, due to the widely adopted Rotary Position Embedding mechanism in modern LLMs, naive low-rank compression suffers severe accuracy degradation or creates a new speed bottleneck, as the low-rank cache must first be reconstructed in order to apply RoPE. In this paper, we introduce two key insights: first, the application of RoPE to the key vectors increases their variance, which in turn results in a higher rank; second, after the key vectors are transformed into the latent space, they largely maintain their representation across most layers. Based on these insights, we propose the Sparse Attention in Latent Space framework. SALS projects the KV cache into a compact latent space via low-rank projection, and performs sparse token selection using RoPE-free query-key interactions in this space. By reconstructing only a small subset of important tokens, it avoids the overhead of full KV cache reconstruction. We comprehensively evaluate SALS on various tasks using two large-scale models: LLaMA2-7b-chat and Mistral-7b, and additionally verify its scalability on the RULER-128k benchmark with LLaMA3.1-8B-Instruct. Experimental results demonstrate that SALS achieves SOTA performance by maintaining competitive accuracy. Under different settings, SALS achieves 6.4-fold KV cache compression and 5.7-fold speed-up in the attention operator compared to FlashAttention2 on the 4K sequence. For the end-to-end throughput performance, we achieves 1.4-fold and 4.5-fold improvement compared to GPT-fast on 4k and 32K sequences, respectively.


- Key 加上RoPE后不好压缩，因此保留RoPE之前的版本，用于压缩
- Pre-RoPE kv compression，减少IO和计算