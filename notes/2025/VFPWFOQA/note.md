# MoR: Mixture Of Representations For Mixed-Precision Training

> Bor-Yiing Su, Peter Dykas, Mike Chrzanowski, Jatin Chhugani

![111](../../blank.jpg)

## Abstract

Mixed-precision training is a crucial technique for scaling deep learning models, but successful mixedprecision training requires identifying and applying the right combination of training methods. This paper presents our preliminary study on Mixture-of-Representations (MoR), a novel, per-tensor and sub-tensor level quantization framework that dynamically analyzes a tensor's numerical properties to select between a variety of different representations. Based on the framework, we have proposed and experimented concrete algorithms that choose dynamically between FP8 and BF16 representations for both per-tensor and sub-tensor level granularities. Our universal approach is designed to preserve model quality across various quantization partition strategies and datasets. Our initial findings show that this approach can achieve state-of-the-art results with 98.38% of tensors quantized to the FP8 format. This work highlights the potential of dynamic, property-aware quantization while preserving model quality. We believe this approach can generally improve the robustness of low precision training, as demonstrated by achieving FP8 accuracies that are on par with existing approaches without the need for fine-grain partitioning, or can be used in combination with other training methods to improve the leverage of even lower precision number formats such as NVFP4.
