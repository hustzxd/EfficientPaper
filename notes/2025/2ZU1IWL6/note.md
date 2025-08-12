# Fast and Simplex: 2-Simplicial Attention in Triton

> Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil

![](../../blank.jpg)

## Abstract

Recent work has shown that training loss scales as a power law with both
model size and the number of tokens, and that achieving compute-optimal models
requires scaling model size and token count together. However, these scaling
laws assume an infinite supply of data and apply primarily in compute-bound
settings. As modern large language models increasingly rely on massive
internet-scale datasets, the assumption that they are compute-bound is becoming
less valid. This shift highlights the need for architectures that prioritize
token efficiency.
  In this work, we investigate the use of the 2-simplicial Transformer, an
architecture that generalizes standard dot-product attention to trilinear
functions through an efficient Triton kernel implementation. We demonstrate
that the 2-simplicial Transformer achieves better token efficiency than
standard Transformers: for a fixed token budget, similarly sized models
outperform their dot-product counterparts on tasks involving mathematics,
coding, reasoning, and logic. We quantify these gains by demonstrating that
$2$-simplicial attention changes the exponent in the scaling laws for knowledge
and reasoning tasks compared to dot product attention.
