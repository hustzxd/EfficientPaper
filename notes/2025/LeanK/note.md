# LeanK: Learnable K Cache Channel Pruning for Efficient Decoding

> Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu

![](fig2.png)

## Abstract

Large language models (LLMs) enable long-context tasks but face efficiency
challenges due to the growing key-value (KV) cache. We propose LeanK, a
learning-based method that prunes unimportant key (K) cache channels by
leveraging static channel sparsity. With a novel two-stage training process,
LeanK learns channel-wise static mask that could satisfy specific sparsity
ratio and hardware alignment requirement. LeanK reduces GPU memory and
accelerates decoding without sacrificing accuracy. Experiments demonstrate up
to 70% K cache and 16%-18% V cache memory reduction. Custom decoding kernel
enables 1.3x speedup for attention computation. We also provide insights into
model channels and attention heads during long-context inference by analyzing
the learned importance distribution. Our code is available at
https://aka.ms/LeanK.

- static channel mask
- 保留attention sink和recent tokens
- K cache中间部分按照channel进行sparse，如上图所示，mask训练提前得到并固定
- 每32 个decoding step更新是更新recent tokens，mask位置其实仍然固定
- mask得到方式通过两阶段训练得到，并在推理时固定