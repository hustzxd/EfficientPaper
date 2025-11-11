# PureKV: Plug-and-Play KV Cache Optimization with Spatial-Temporal Sparse Attention for Vision-Language Large Models

> Zhonghua Jiang, Kunxi Li, Yiyun Zhou, Sihao Liu, Zhaode Wang, Chengfei lv, Shengyu Zhang

![111](fig1.png)

## Abstract

Vision-Language Large Models (VLLMs) face significant efficiency challenges
when processing high-resolution inputs. The quadratic complexity in attention
and autoregressive generation, as well as the constantly growing key value (KV)
cache size, severely hinder the prefilling and decoding stages. Recent efforts
have attempted to compress KV cache by identifying and pruning KV cache of less
important tokens, but these methods typically rely on attention scores to
estimate token importance, making them incompatible with efficient attention
mechanisms such as FlashAttention and Sparse Attention, which do not explicitly
compute attention matrices. Moreover, existing methods overlook how sparse
attention, while accelerating the prefilling stage, alters the information
structure of the KV cache, thereby compromising the effectiveness of downstream
KV cache compression strategies. To address this issue, we propose PureKV, a
plug-and-play framework for joint optimization of sparse attention and KV cache
compression. We first introduce a KV cache compression strategy that is fully
compatible with efficient attention accelerators. Our method utilizes lower
layer attention scores to estimate the importance of high layers' KV cache,
enabling active pruning without compromising accuracy. In addition, we have
designed a Spatial-Temporal Sparse Attention (ST-SpAttn) module specifically
tailored for video KV cache compression algorithms. This module combines
spatial and temporal attention sparsity to improve the compression efficiency
of KV cache optimization algorithms by purifying spatial noise and temporal
redundancy in KV cache. At the same time, ST-SpAttn also accelerated the
prefilling stage of VLLMs. Extensive experiments on VLLMs (VideoLLaMA2,
Qwen2.5-VL) have shown that PureKV achieves 5.0 times KV cache compression and
3.16 times prefill acceleration, with negligible quality degradation.

使用上一层的attention score，预测下一层的kv重要性