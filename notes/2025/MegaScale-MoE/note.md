# MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

> Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin Jin, Xin Liu

![](fig8.png)

## Abstract

We present MegaScale-MoE, a production system tailored for the efficient
training of large-scale mixture-of-experts (MoE) models. MoE emerges as a
promising architecture to scale large language models (LLMs) to unprecedented
sizes, thereby enhancing model performance. However, existing MoE training
systems experience a degradation in training efficiency, exacerbated by the
escalating scale of MoE models and the continuous evolution of hardware.
  Recognizing the pivotal role of efficient communication in enhancing MoE
training, MegaScale-MoE customizes communication-efficient parallelism
strategies for attention and FFNs in each MoE layer and adopts a holistic
approach to overlap communication with computation at both inter- and
intra-operator levels. Additionally, MegaScale-MoE applies communication
compression with adjusted communication patterns to lower precision, further
improving training efficiency. When training a 352B MoE model on 1,440 NVIDIA
Hopper GPUs, MegaScale-MoE achieves a training throughput of 1.41M tokens/s,
improving the efficiency by 1.88$\times$ compared to Megatron-LM. We share our
operational experience in accelerating MoE training and hope that by offering
our insights in system design, this work will motivate future research in MoE
systems.
