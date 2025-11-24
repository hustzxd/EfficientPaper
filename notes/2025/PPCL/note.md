# Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers

> Jian Ma, Qirong Peng, Xujie Zhu, Peixing Xie, Chen Chen, Haonan Lu

![111](fig3.png)

## Abstract

Diffusion Transformers (DiTs) have shown exceptional performance in image generation, yet their large parameter counts incur high computational costs, impeding deployment in resource-constrained settings. To address this, we propose Pluggable Pruning with Contiguous Layer Distillation (PPCL), a flexible structured pruning framework specifically designed for DiT architectures. First, we identify redundant layer intervals through a linear probing mechanism combined with the first-order differential trend analysis of similarity metrics. Subsequently, we propose a plug-and-play teacher-student alternating distillation scheme tailored to integrate depth-wise and width-wise pruning within a single training phase. This distillation framework enables flexible knowledge transfer across diverse pruning ratios, eliminating the need for per-configuration retraining. Extensive experiments on multiple Multi-Modal Diffusion Transformer architecture models demonstrate that PPCL achieves a 50\% reduction in parameter count compared to the full model, with less than 3\% degradation in key objective metrics. Notably, our method maintains high-quality image generation capabilities while achieving higher compression ratios, rendering it well-suited for resource-constrained environments. The open-source code, checkpoints for PPCL can be found at the following link: https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning.
