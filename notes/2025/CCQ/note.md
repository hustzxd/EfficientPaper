# CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs

> Zhaojing Zhou, Xunchao Li, Minghao Li, Handi Zhang, Haoshuang Wang, Wenbin Chang, Yiqun Liu, Qingqing Dang, Dianhai Yu, Yanjun Ma, Haifeng Wang

![](../../blank.jpg)

## Abstract

The rapid scaling of Large Language Models (LLMs) elevates inference costs
and compounds substantial deployment barriers. While quantization to 8 or 4
bits mitigates this, sub-3-bit methods face severe accuracy, scalability, and
efficiency degradation. We propose Convolutional Code Quantization (CCQ), an
inference-optimized quantization approach compressing LLMs to 2.0-2.75 bits
with minimal accuracy loss. Departing from error-prone scalar quantization or
slow vector quantization, CCQ integrates a hardware-aware bit-shift encoding
and decoding solution with Convolutional Code, Hybrid Encoding, and Code
Cluster, jointly overcoming accuracy-speed bottlenecks. We construct a
lookup-free encoding space, enabling a linear mapping between the codebook and
weight vectors, thereby optimizing inference performance. Meanwhile, by drawing
on the concept of data mapping from vector quantization, we minimize the
performance degradation of the model under extremely low-bit conditions.
Experiments demonstrate that CCQ achieves outstanding performance on LLMs
across various benchmarks. We compress DeepSeek-V3 (671B total parameters) to
184GB and ERNIE-4.5-300B-A47B to 89GB, enabling single-GPU deployment of ERNIE
4.5 and eliminating inter-card communication. The 2-bit ERNIE-4.5-300B-A47B
model and inference engine have been open-sourced.
