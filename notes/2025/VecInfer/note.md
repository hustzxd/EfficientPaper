# VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization

> Dingyu Yao, Chenxu Yang, Zhengyang Tong, Zheng Lin, Wei Liu, Jian Luan, Weiping Wang

![111](fig4.png)

## Abstract

The Key-Value (KV) cache introduces substantial memory overhead during large
language model (LLM) inference. Although existing vector quantization (VQ)
methods reduce KV cache usage and provide flexible representational capacity
across bit-widths, they suffer severe performance degradation at ultra-low
bit-widths due to key cache outliers that hinder effective codebook
utilization. To address this challenge, we propose VecInfer, a novel VQ method
for aggressive KV cache compression while enabling efficient inference. By
applying smooth and Hadamard transformations, VecInfer suppresses outliers in
the key cache, enabling the codebook to comprehensively cover the original data
distribution and thereby reducing quantization difficulty. To facilitate
efficient deployment, we design an optimized CUDA kernel that fuses computation
with dequantization to minimize memory access overhead. Extensive evaluations
demonstrate that VecInfer consistently outperforms existing quantization
baselines across both long-context understanding and mathematical reasoning
tasks. With only 2-bit quantization, VecInfer achieves performance comparable
to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in
large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in
single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length.
