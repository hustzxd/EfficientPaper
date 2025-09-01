# RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations

> Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, Kehong Yuan

![111](fig3.png)

## Abstract

Key-Value (KV) cache facilitates efficient large language models (LLMs)
inference by avoiding recomputation of past KVs. As the batch size and context
length increase, the oversized KV caches become a significant memory
bottleneck, highlighting the need for efficient compression. Existing KV
quantization rely on fine-grained quantization or the retention of a
significant portion of high bit-widths caches, both of which compromise
compression ratio and often fail to maintain robustness at extremely low
average bit-widths. In this work, we explore the potential of rotation
technique for 2-bit KV quantization and propose RotateKV, which achieves
accurate and robust performance through the following innovations: (i)
Outlier-Aware Rotation, which utilizes channel-reordering to adapt the
rotations to varying channel-wise outlier distributions without sacrificing the
computational efficiency of the fast Walsh-Hadamard transform (FWHT); (ii)
Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position
embedding (RoPE) on proposed outlier-aware rotation and further smooths
outliers across heads; (iii) Attention-Sink-Aware Quantization, which leverages
the massive activations to precisely identify and protect attention sinks.
RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit
quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning
and long-context capabilities, with less than 1.7\% degradation on GSM8K,
outperforming existing methods even at lower average bit-widths. RotateKV also
showcases a 3.97x reduction in peak memory usage, supports 5.75x larger batch
sizes, and achieves a 2.32x speedup in decoding stage.


对 kv进行Hadamard变换