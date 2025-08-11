# Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models

> Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang

<p align="center">
<img src="../../blank.jpg" width="600" title="blank">
</p>

## Abstract

In the era of large language models (LLMs), N:M sparsity has emerged as a
structured compression technique critical for accelerating inference. While
prior work has primarily focused on weight sparsity, it often suffers from
significant accuracy degradation. Activation sparsity, though promising, is
typically training-dependent and faces challenges in generalization. To address
these limitations, we introduce Amber Pruner, a training-free N:M activation
sparsity method designed specifically for the prefill stage, targeting the
acceleration of linear projection layers in LLMs. Extensive experiments across
multiple models and sparsity ratios (2:4, 4:8, and 8:16) demonstrate that Amber
Pruner can effectively sparsify and accelerate more than 55% of linear
computations without requiring model retraining. To further enhance generality
and efficiency, we propose Outstanding-sparse, a unified framework that
integrates Amber Pruner with post-training W8A8 quantization. Our approach
preserves strong performance across a range of downstream tasks, with notable
advantages in generative tasks. This work pioneers a new frontier in activation
sparsity, providing foundational insights that are poised to guide the
co-evolution of algorithms and architectures in the design of next-generation
AI systems.

对activation进行N：M pruning，方法借鉴Wanda，通过weight给activation的重要性进行scale，因为weight是静态的，因此scale也是静态的，可以提前保存，在推理时，动态的激活值与静态的scale相乘，并使用topk得到动态的mask。

方法较为简单，动态的算mask开销较大。