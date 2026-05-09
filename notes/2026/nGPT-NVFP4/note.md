# Normalized Architectures are Natively 4-Bit

> Maxim Fishman, Brian Chmiel, Ron Banner, Daniel Soudry, Boris Ginsburg

![111](../../blank.jpg)

## Abstract

Training large language models at 4-bit precision is critical for efficiency. We show that nGPT, an architecture that constrains weights and hidden representations to the unit hypersphere, is inherently more robust to low-precision arithmetic. This removes the need for interventions-such as applying random Hadamard transforms and performing per-tensor scaling calculations-to preserve model quality, and it enables stable end-to-end NVFP4 training. We validate this approach on both a 1.2B dense model and hybrid (Mamba-Transformer) MoE models of up to 3B/30B parameters. We trace this robustness to the dot product: while quantization noise remains largely uncorrelated in both standard and normalized architectures, the signal behaves differently. In nGPT, the hypersphere constraint enhances weak positive correlations among the element-wise products, leading to a constructive accumulation of the signal across the hidden dimension while the noise continues to average out. This yields a higher effective signal-to-noise ratio and a flatter loss landscape, with the effect strengthening as the hidden dimension grows, suggesting increasing advantages at scale. A reference implementation is available at https://github.com/anonymous452026/ngpt-nvfp4

---

## GPT Summary

> 由 GPT 自动生成，请人工核验。

### 1. 研究背景与动机

4-bit NVFP4 训练对降低大模型训练成本很重要，但标准 Transformer 在低精度训练中容易受到 outlier、scale 波动和量化噪声影响。已有 NVFP4 recipe 往往依赖 Randomized Hadamard Transform、dynamic per-tensor scaling、stochastic rounding 或 mixed-precision exception 来避免发散，这些技巧虽然有效，但会引入额外计算和系统开销。

这篇论文提出的问题是：是否存在一种架构本身就对 4-bit arithmetic 更鲁棒，从而减少这些后处理量化补丁？

### 2. nGPT-NVFP4 核心思想

论文研究 nGPT，即将 hidden representations 和 weights 约束到 unit hypersphere 的 normalized Transformer。核心观点是：nGPT 的低精度鲁棒性不是来自更小的局部量化噪声，而是来自更好的 **signal accumulation**。

在标准 GPT 中，模型可以依赖少数大 coordinate 主导 dot product，这些 outlier 在 4-bit 下很难稳定表示。nGPT 阻止单个 coordinate 任意放大，迫使模型在许多维度上学习 distributed alignment。这样 element-wise products 会出现弱但稳定的正相关，signal 在高维求和中建设性累积，而量化噪声仍近似不相关并被平均掉，因此有效 SNR 更高。

### 3. 关键技术

| 技术 | 说明 |
|------|------|
| Unit hypersphere constraint | 将 hidden states 和 weights 约束到单位球面，减少 outlier coordinate 对输出的支配。 |
| NVFP4 end-to-end training | 面向 NVIDIA Blackwell 原生支持的 NVFP4 训练格式，验证 4-bit 训练稳定性。 |
| 去除 RHT 与动态 scaling | nGPT activation scale 更稳定，可去掉 Randomized Hadamard Transform 和 dynamic per-tensor scaling。 |
| Dot-product SNR analysis | 分析量化前后 dot product 的 signal-to-noise ratio，说明优势来自 signal 更一致地累积。 |
| Loss landscape analysis | nGPT 对权重扰动更不敏感，论文报告 GPT loss 退化速度约为 nGPT 的 3.5 倍。 |

### 4. 实验结果

| 设置 | 结果 |
|------|------|
| 1.2B dense model, 1T tokens | nGPT-NVFP4 在去掉 RHT/per-tensor scaling 后仍取得比标准 GPT-NVFP4 更低 relative error，并在多个下游任务上更好。 |
| Hybrid Mamba-Transformer MoE 400M/600M | nGPT 架构优势可迁移到 hybrid MoE，降低 relative error 并去除 RHT/scaling 开销。 |
| Hybrid Mamba-Transformer MoE 3B/30B | 训练约 500B tokens 后，nGPT-NVFP4 relative error 接近 0%，说明优势在更大模型上仍存在。 |
| Learning rate robustness | nGPT 在 BF16 与 NVFP4 下最佳 LR 基本一致，可直接迁移；标准 GPT 的 NVFP4 最佳 LR 明显偏移。 |
| GB200 layer speed | 大 hidden size 下，nGPT-NVFP4 单层相对 BF16 GPT baseline 达到约 3.3–3.6× speedup。 |

### 5. 核心贡献

1. 证明 normalized architecture 可以天然具备 4-bit quantization robustness。
2. 提出 signal accumulation 解释：nGPT 优势来自 signal 在高维 dot product 中一致累积，而不是局部噪声显著变小。
3. 展示 nGPT 可去掉 RHT 和 dynamic per-tensor scaling，减少低精度训练路径开销。
4. 在 1.2B dense model 和最高 3B/30B hybrid MoE 上验证 NVFP4 训练稳定性。
5. 对我们当前方向而言，该论文偏训练和架构设计，不属于近期 deployment-only 主线，但对理解 4-bit outlier 控制和低精度鲁棒性有参考价值。
