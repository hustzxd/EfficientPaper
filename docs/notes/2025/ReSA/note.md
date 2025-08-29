# Rectified Sparse Attention

![](fig2.png)

## Abstract

Efficient long-sequence generation is a critical challenge for Large Language
Models. While recent sparse decoding methods improve efficiency, they suffer
from KV cache misalignment, where approximation errors accumulate and degrade
generation quality. In this work, we propose Rectified Sparse Attention (ReSA),
a simple yet effective method that combines block-sparse attention with
periodic dense rectification. By refreshing the KV cache at fixed intervals
using a dense forward pass, ReSA bounds error accumulation and preserves
alignment with the pretraining distribution. Experiments across math reasoning,
language modeling, and retrieval tasks demonstrate that ReSA achieves
near-lossless generation quality with significantly improved efficiency.
Notably, ReSA delivers up to 2.42$\times$ end-to-end speedup under decoding at
256K sequence length, making it a practical solution for scalable long-context
inference. Code is available at https://aka.ms/ReSA-LM.

使用sparse decoding t次后，使用一次dense prefill修正之前的生成的t个token的kv cache，和speculative decoding类似，但是不进行reject/accept判断。