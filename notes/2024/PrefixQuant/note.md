# PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization

> Mengzhao Chen, Yi Liu, Jiahao Wang, Yi Bin, Wenqi Shao, Ping Luo

![111](fig2.png)

## Abstract

Existing weight-activation quantization methods for Large Language Models
(LLMs) primarily address channel-wise outliers but often neglect token-wise
outliers, which limits the accuracy of quantized models. In this work, we
propose PrefixQuant, a novel quantization method that achieves state-of-the-art
performance across various precision levels (W4A4KV4 and W4A8KV4) and
granularities (dynamic and static quantization) by effectively isolating
token-wise outliers. First, PrefixQuant eliminates token-wise outliers by
prefixing outlier tokens in the KV cache, a process that is training-free and
highly efficient (e.g., 1 minutes for Llama-3-70B). Second, PrefixQuant
introduces new trainable parameters for block-wise training to compensate for
quantization error. Our experiments show that PrefixQuant significantly
outperforms existing dynamic quantization methods, even under coarser static
quantization settings. For instance, PrefixQuant achieves an average accuracy
improvement of +3.08 and +2.85 points over SpinQuant (dynamic quantization) on
five zero-shot reasoning tasks under dynamic and static quantization settings,
respectively, on W4A4KV4 Llama-3-8B. Additionally, we demonstrate up to 2.74x
prefilling speedup and 2.16x decoding speedup for LLMs using W4A4 PrefixQuant.
Our code is available at https://github.com/ChenMnZ/PrefixQuant.

将特定的一些token标记为outlier