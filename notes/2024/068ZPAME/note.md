# A Survey on Inference Optimization Techniques for Mixture of Experts Models

> Jiacheng Liu, Peng Tang, Wenfeng Wang, Yuhang Ren, Xiaofeng Hou, Pheng-Ann Heng, Minyi Guo, Chao Li

<p align="center">
<img src="tab1.png" width="600" title="blank">
</p>

## Abstract

The emergence of large-scale Mixture of Experts (MoE) models represents a
significant advancement in artificial intelligence, offering enhanced model
capacity and computational efficiency through conditional computation. However,
deploying and running inference on these models presents significant challenges
in computational resources, latency, and energy efficiency. This comprehensive
survey analyzes optimization techniques for MoE models across the entire system
stack. We first establish a taxonomical framework that categorizes optimization
approaches into model-level, system-level, and hardware-level optimizations. At
the model level, we examine architectural innovations including efficient
expert design, attention mechanisms, various compression techniques such as
pruning, quantization, and knowledge distillation, as well as algorithm
improvement including dynamic routing strategies and expert merging methods. At
the system level, we investigate distributed computing approaches, load
balancing mechanisms, and efficient scheduling algorithms that enable scalable
deployment. Furthermore, we delve into hardware-specific optimizations and
co-design strategies that maximize throughput and energy efficiency. This
survey provides both a structured overview of existing solutions and identifies
key challenges and promising research directions in MoE inference optimization.
To facilitate ongoing updates and the sharing of cutting-edge advances in MoE
inference optimization research, we have established a repository accessible at
https://github.com/MoE-Inf/awesome-moe-inference/.
