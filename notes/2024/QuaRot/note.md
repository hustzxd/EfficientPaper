# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

> Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman

![111](../../blank.jpg)

## Abstract

We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This computational invariance is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism, and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4 bits, without any channels identified for retention in higher precision. Our 4-bit quantized LLaMa2-70B model has losses of at most 0.47 WikiText-2 perplexity and retains 99% of the zero-shot performance. We also show that QuaRot can provide lossless 6 and 8 bit LLaMa2 models without any calibration data using round-to-nearest quantization. Code is available at: https://github.com/spcl/QuaRot.


---

*以下总结由 MiMo 生成：*

这篇论文旨在解决大语言模型在4位量化中因隐藏状态异常值导致的精度损失问题。研究者提出了QuaRot方法，通过旋转模型隐藏状态来消除异常值，同时保持计算不变性，从而实现端到端的4位量化。实验表明，该方法在LLaMa2-70B模型上仅带来0.47的WikiText-2困惑度损失，并保留了99%的零样本性能，同时支持无校准数据的6位和8位无损量化。

## GPT Summary

> 由 GPT 自动生成，请人工核验。

### 论文动机

QuaRot 关注 LLM 端到端低比特推理中的 activation outlier 问题。传统 4-bit 量化往往需要保留高精度 outlier channel、使用复杂校准，或只能量化部分矩阵乘；这会削弱真正的 INT4 加速和 KV cache 压缩收益。论文的目标是在不改变模型输出函数的前提下，通过旋转消除 hidden state outlier，从而实现 weights、activations 和 KV cache 的统一 4-bit 量化。

### 核心方法

- **计算不变旋转**：利用正交旋转不改变线性层组合输出的性质，将 Hadamard/randomized Hadamard 旋转注入 residual stream、FFN、attention 和 KV cache 相关路径。
- **Outlier-free activation**：旋转后 hidden state 的异常通道被扩散到更多维度，动态范围更均匀，使 round-to-nearest / group-wise 量化更稳定。
- **端到端 A4W4KV4**：不仅量化权重和激活，也量化 KV cache；目标是让所有主要矩阵乘都可以用 4-bit 执行，而不是保留少数高精度通道。
- **旋转开销处理**：可离线融合到权重的旋转尽量预先吸收；必须在线执行的部分用 Hadamard transform 和 CUDA/CUTLASS kernel 降低额外成本。

### 实验结论

- 在 LLaMA2 系列上，QuaRot 可以实现 4-bit weights、activations 和 KV cache 的端到端量化。
- LLaMA2-70B 的 4-bit 版本在 WikiText-2 上最多只带来 0.47 perplexity 损失，并保留约 99% zero-shot 性能。
- 对 6-bit / 8-bit 设置，论文报告无需校准数据即可接近无损量化。
- 系统层面，论文报告 LLaMA2-70B 在 batch size 64、sequence length 2048 的 prefill 场景可获得最高 3.33× 加速，并在 decode 阶段带来约 3.89× memory saving。

### 局限与注意点

- 方法依赖旋转变换与 kernel 支持；实际收益受硬件 INT4 路径、Hadamard transform 开销和 serving 框架集成程度影响。
- 论文主要验证 LLaMA2 风格 dense decoder-only 模型；对更新的 GQA/MQA、MoE、超长上下文和 agent workload 仍需要重新评估。
- QuaRot 本身不是 KV eviction / offload / scheduling 方法，更适合作为 KV cache quantization baseline 或与层级 KV management 组合。

### 与当前研究主线的关系

QuaRot 是 KV cache quantization 方向的重要 baseline。它提示 KV management 不一定只在“保留/丢弃/换入换出”之间选择，还可以增加一个低精度存储层：GPU hot KV、CPU/SSD cold KV 都可以结合 4-bit KV 表示，进一步降低 HBM 和传输压力。后续可以考虑将 QuaRot/KIVI/KVQuant 一类量化方法与 HiSparse、Bidaw、PredictKV、Tutti 等 runtime memory hierarchy 方法组合，研究“quantized hierarchical KV cache”。

