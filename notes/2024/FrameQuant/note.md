# FrameQuant: Flexible Low-Bit Quantization for Transformers

![](framequant.png)

## Abstract

Transformers are the backbone of powerful foundation models for many Vision
and Natural Language Processing tasks. But their compute and memory/storage
footprint is large, and so, serving such models is expensive often requiring
high-end hardware. To mitigate this difficulty, Post-Training Quantization
seeks to modify a pre-trained model and quantize it to eight bits or lower,
significantly boosting compute/memory/latency efficiency. Such models have been
successfully quantized to four bits with some performance loss. In this work,
we outline a simple scheme to quantize Transformer-based models to just two
bits (plus some overhead) with only a small drop in accuracy. Key to our
formulation is a concept borrowed from Harmonic analysis called Fusion Frames.
Our main finding is that the quantization must take place not in the original
weight space, but instead in the Fusion Frame representations. If quantization
is interpreted as the addition of noise, our casting of the problem allows
invoking an extensive body of known consistent recovery and noise robustness
guarantees. Further, if desired, de-noising filters are known in closed form.
We show empirically, via a variety of experiments, that (almost) two-bit
quantization for Transformer models promises sizable efficiency gains.

> https://arxivtools.blob.core.windows.net/xueshuxiangzipaperhtml/2023_12_22/2312.13488.pdf

直接在original weight space对权重量化不是最优的，论文中提出将weight转化到Fusion Frame空间进行表示，从而能够将weight量化到2bit，比SOTA方法有较大的提升。

与QuIP的区别，如果设置redundancy factor r = 1，且随机设置正交矩阵P,那么就和QuIP一致了。

是否适合更高的bit, 比如3-bit, 4-bit配置，论文没有给出对应的实验结果，并指出OPTQ等方法已经有4-bit的结果了，所以推断这个方法在2-bit上有提升，在3/4-bit上可能提升不明显。
