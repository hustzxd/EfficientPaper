# Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention

> Robin Geens, Marian Verhelst

![111](../../blank.jpg)

## Abstract

Multi-Head Latent Attention (MLA), introduced in DeepSeek-V2, improves the efficiency of large language models by projecting query, key, and value tensors into a compact latent space. This architectural change reduces the KV-cache size and significantly lowers memory bandwidth demands, particularly in the autoregressive decode phase. This letter presents the first hardware-centric analysis of MLA, comparing it to conventional Multi-Head Attention (MHA) and evaluating its implications for accelerator performance. We identify two alternative execution schemes of MLA--reusing, resp. recomputing latent projection matrices--which offer distinct trade-offs between compute and memory access. Using the Stream design space exploration framework, we model their throughput and energy cost across a range of hardware platforms and find that MLA can shift attention workloads toward the compute-bound regime.
  Our results show that MLA not only reduces bandwidth usage but also enables adaptable execution strategies aligned with hardware constraints. Compared to MHA, it provides more stable and efficient performance, particularly on bandwidth-limited hardware platforms. These findings emphasize MLA's relevance as a co-design opportunity for future AI accelerators.
