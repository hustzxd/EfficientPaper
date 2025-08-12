# SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

![](fig1.png)

## Abstract

We introduce SeerAttention-R, a sparse attention framework specifically
tailored for the long decoding of reasoning models. Extended from
SeerAttention, SeerAttention-R retains the design of learning attention
sparsity through a self-distilled gating mechanism, while removing query
pooling to accommodate auto-regressive decoding. With a lightweight plug-in
gating, SeerAttention-R is flexible and can be easily integrated into existing
pretrained model without modifying the original parameters. We demonstrate that
SeerAttention-R, trained on just 0.4B tokens, maintains near-lossless reasoning
accuracy with 4K token budget in AIME benchmark under large sparse attention
block sizes (64/128). Using TileLang, we develop a highly optimized sparse
decoding kernel that achieves near-theoretical speedups of up to 9x over
FlashAttention-3 on H100 GPU at 90% sparsity. Code is available at:
https://github.com/microsoft/SeerAttention.
